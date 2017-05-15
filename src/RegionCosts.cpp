#include "RegionCosts.h"
#include "IRVisitor.h"
#include "IRMutator.h"
#include "FindCalls.h"
#include "PartitionLoops.h"
#include "RealizationOrder.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

using std::string;
using std::map;
using std::set;
using std::vector;

namespace {

// Visitor for keeping track of all input images accessed and their types.
class FindImageInputs : public IRVisitor {
    using IRVisitor::visit;
    set<string> seen_image_param;

    void visit(const Call *call) {
        if (call->call_type == Call::Image) {
            input_type[call->name] = call->type;

            // Call to an ImageParam
            if (call->param.defined() && (seen_image_param.count(call->name) == 0)) {
                for (int i = 0; i < call->param.dimensions(); ++i) {
                    const Expr &min = call->param.min_constraint_estimate(i);
                    const Expr &extent = call->param.extent_constraint_estimate(i);

                    user_assert(min.defined())
                        << "AutoSchedule: Estimate of the min value of ImageParam \""
                        << call->name << "\" in dimension " << i << " is not specified.\n";
                    user_assert(extent.defined())
                        << "AutoSchedule: Estimate of the extent value of ImageParam \""
                        << call->name << "\" in dimension " << i << " is not specified.\n";

                    string min_var = call->param.name() + ".min." + std::to_string(i);
                    string extent_var = call->param.name() + ".extent." + std::to_string(i);

                    input_estimates.emplace(min_var, Interval(min, min));
                    input_estimates.emplace(extent_var, Interval(extent, extent));
                    seen_image_param.insert(call->name);
                }
            }
        }
        for (size_t i = 0; i < call->args.size(); i++) {
            call->args[i].accept(this);
        }
    }
public:
    map<string, Type> input_type;
    map<string, Interval> input_estimates;
};

// Visitor for tracking the arithmetic and memory costs.
class ExprCost : public IRVisitor {
    using IRVisitor::visit;

    // Immediate values and variables do not incur any cost.
    void visit(const IntImm *) {}
    void visit(const UIntImm *) {}
    void visit(const FloatImm *) {}
    void visit(const StringImm *) {}
    void visit(const Variable *) {}

    void visit(const Cast *op) {
        op->value.accept(this);
        cost.arith += 1;
    }

    template<typename T>
    void visit_binary_operator(const T *op, int op_cost) {
        op->a.accept(this);
        op->b.accept(this);
        cost.arith += op_cost;
    }

    // The costs of all the simple binary operations is set to one.
    // TODO: Changing the costs for division and multiplication may be
    // beneficial. Write a test case to validate this and update the costs
    // accordingly.

    void visit(const Add *op) { visit_binary_operator(op, 1); }
    void visit(const Sub *op) { visit_binary_operator(op, 1); }
    void visit(const Mul *op) { visit_binary_operator(op, 1); }
    void visit(const Div *op) { visit_binary_operator(op, 1); }
    void visit(const Mod *op) { visit_binary_operator(op, 1); }
    void visit(const Min *op) { visit_binary_operator(op, 1); }
    void visit(const Max *op) { visit_binary_operator(op, 1); }
    void visit(const EQ *op) { visit_binary_operator(op, 1); }
    void visit(const NE *op) { visit_binary_operator(op, 1); }
    void visit(const LT *op) { visit_binary_operator(op, 1); }
    void visit(const LE *op) { visit_binary_operator(op, 1); }
    void visit(const GT *op) { visit_binary_operator(op, 1); }
    void visit(const GE *op) { visit_binary_operator(op, 1); }
    void visit(const And *op) { visit_binary_operator(op, 1); }
    void visit(const Or *op) { visit_binary_operator(op, 1); }

    void visit(const Not *op) {
        op->a.accept(this);
        cost.arith += 1;
    }

    void visit(const Select *op) {
        op->condition.accept(this);
        op->true_value.accept(this);
        op->false_value.accept(this);
        cost.arith += 1;
    }

    void visit(const Call *call) {
        if (call->call_type == Call::Halide || call->call_type == Call::Image) {
            // Each call also counts as an op since it results in a load instruction.
            cost.arith += 1;
            cost.memory += call->type.bytes();
            detailed_byte_loads[call->name] += call->type.bytes();
        } else if (call->call_type == Call::Extern || call->call_type == Call::PureExtern ||
                   call->call_type == Call::ExternCPlusPlus) {
            // TODO: Suffix based matching is kind of sketchy; but going ahead with
            // it for now. Also not all the PureExtern's are accounted for yet.
            if (ends_with(call->name, "_f64")) {
                cost.arith += 20;
            } else if (ends_with(call->name, "_f32")) {
                cost.arith += 10;
            } else if (ends_with(call->name, "_f16")) {
                cost.arith += 5;
            } else {
                // There is no visibility into an extern stage so there is no
                // way to know the cost of the call statically. Modeling the
                // cost of an extern stage requires profiling or user annotation.
                user_warning << "Unknown extern call " << call->name << '\n';
            }
        } else if (call->call_type == Call::Intrinsic || call->call_type == Call::PureIntrinsic) {
            // TODO: Improve the cost model. In some architectures (e.g. ARM or
            // NEON), count_leading_zeros should be as cheap as bitwise ops.
            // div_round_to_zero and mod_round_to_zero can also get fairly expensive.
            if (call->is_intrinsic(Call::reinterpret) || call->is_intrinsic(Call::bitwise_and) ||
                    call->is_intrinsic(Call::bitwise_not) || call->is_intrinsic(Call::bitwise_xor) ||
                    call->is_intrinsic(Call::bitwise_or) || call->is_intrinsic(Call::shift_left) ||
                    call->is_intrinsic(Call::shift_right) || call->is_intrinsic(Call::div_round_to_zero) ||
                    call->is_intrinsic(Call::mod_round_to_zero) || call->is_intrinsic(Call::undef)) {
                cost.arith += 1;
            } else if (call->is_intrinsic(Call::abs) || call->is_intrinsic(Call::absd) ||
                       call->is_intrinsic(Call::lerp) || call->is_intrinsic(Call::random) ||
                       call->is_intrinsic(Call::count_leading_zeros) ||
                       call->is_intrinsic(Call::count_trailing_zeros)) {
                cost.arith += 5;
            } else if (call->is_intrinsic(Call::likely)) {
                // Likely does not result in actual operations.
            } else {
                internal_error << "Unknown intrinsic call " << call->name << '\n';
            }
        }

        for (size_t i = 0; i < call->args.size(); i++) {
            call->args[i].accept(this);
        }
    }

    void visit(const Shuffle *op) {
        cost.arith += 1;
    }

    void visit(const Let *let) {
        let->value.accept(this);
        let->body.accept(this);
    }

    // None of the following IR nodes should be encountered when traversing the
    // IR at the level at which the auto scheduler operates.
    void visit(const Load *) { internal_assert(false); }
    void visit(const Ramp *) { internal_assert(false); }
    void visit(const Broadcast *) { internal_assert(false); }
    void visit(const LetStmt *) { internal_assert(false); }
    void visit(const AssertStmt *) { internal_assert(false); }
    void visit(const ProducerConsumer *) { internal_assert(false); }
    void visit(const For *) { internal_assert(false); }
    void visit(const Store *) { internal_assert(false); }
    void visit(const Provide *) { internal_assert(false); }
    void visit(const Allocate *) { internal_assert(false); }
    void visit(const Free *) { internal_assert(false); }
    void visit(const Realize *) { internal_assert(false); }
    void visit(const Block *) { internal_assert(false); }
    void visit(const IfThenElse *) { internal_assert(false); }
    void visit(const Evaluate *) { internal_assert(false); }

public:
    Cost cost;
    // Detailed breakdown of bytes loaded by the allocation or function
    // they are loaded from.
    map<string, int64_t> detailed_byte_loads;

    ExprCost() : cost(Cost(0, 0)) {}
};

// Return the number of bytes required to store a single value of the
// function.
int64_t get_func_value_size(const Function &f) {
    int64_t size = 0;
    const vector<Type> &types = f.output_types();
    for (size_t i = 0; i < types.size(); i++) {
        size += types[i].bytes();
    }
    internal_assert(!types.empty());
    return size;
}

// Helper class that only accounts for the likely portion of the expression in
// the case of max, min, and select. This will help costing functions with
// boundary conditions better. The likely intrinsic triggers loop partitioning
// and on average (steady stage) the cost of the expression will be equivalent
// to the likely portion.
//
// TODO: Comment this out for now until we modify the compute expr cost and
// detailed byte loads functions to account for likely exprs.
/*class LikelyExpression : public IRMutator {
    using IRMutator::visit;

    void visit(const Min *op) {
        IRVisitor::visit(op);
        bool likely_a = has_likely_tag(op->a);
        bool likely_b = has_likely_tag(op->b);
        if (likely_a && !likely_b) {
            expr = op->a;
        } else if (likely_b && !likely_a) {
            expr = op->a;
        }
    }

    void visit(const Max *op) {
        IRVisitor::visit(op);
        bool likely_a = has_likely_tag(op->a);
        bool likely_b = has_likely_tag(op->b);
        if (likely_a && !likely_b) {
            expr = op->a;
        } else if (likely_b && !likely_a) {
            expr = op->b;
        }
    }

    void visit(const Select *op) {
        IRVisitor::visit(op);
        bool likely_t = has_likely_tag(op->true_value);
        bool likely_f = has_likely_tag(op->false_value);
        if (likely_t && !likely_f) {
            expr = op->true_value;
        } else if (likely_f && !likely_t) {
            expr = op->false_value;
        }
    }
};*/

Cost compute_expr_cost(Expr expr) {
    // TODO: Handle likely
    //expr = LikelyExpression().mutate(expr);
    ExprCost cost_visitor;
    expr.accept(&cost_visitor);
    return cost_visitor.cost;
}

map<string, int64_t> compute_expr_detailed_byte_loads(Expr expr) {
    // TODO: Handle likely
    //expr = LikelyExpression().mutate(expr);
    ExprCost cost_visitor;
    expr.accept(&cost_visitor);
    return cost_visitor.detailed_byte_loads;
}

} // anonymous namespace

RegionCosts::RegionCosts(const map<string, Function> &_env) : env(_env) {
    for (const auto &kv : env) {
        // Pre-compute the function costs without any inlining.
        func_cost[kv.first] = get_func_cost(kv.second);

        // Get the types of all the image inputs to the pipeline, including
        // their estimated min/extent values if applicable (i.e. if they are
        // ImageParam).
        FindImageInputs find;
        kv.second.accept(&find);
        for (const auto &in : find.input_type) {
            inputs[in.first] = in.second;
        }
        for (const auto &iter : find.input_estimates) {
            input_estimates.push(iter.first, iter.second);
        }
    }
}

Cost RegionCosts::stage_region_cost(string func, int stage, const DimBounds &bounds,
                                    const set<string> &inlines) {
    Function curr_f = get_element(env, func);
    Definition def = get_stage_definition(curr_f, stage);

    Box stage_region;

    const vector<Dim> &dims = def.schedule().dims();
    for (int d = 0; d < (int)dims.size() - 1; d++) {
        stage_region.push_back(get_element(bounds, dims[d].var));
    }

    int64_t size = box_size(stage_region);
    if (size == unknown) {
        // Size could not be determined; therefore, it is not possible to
        // determine the arithmetic and memory costs.
        return Cost();
    }

    // If there is nothing to be inlined, use the pre-computed function cost.
    Cost cost = inlines.empty() ? get_element(func_cost, func)[stage]
                                : get_func_stage_cost(curr_f, stage, inlines);
    return Cost(size * cost.arith, size * cost.memory);
}

Cost RegionCosts::stage_region_cost(string func, int stage, const Box &region,
                                    const set<string> &inlines) {
    Function curr_f = get_element(env, func);

    DimBounds pure_bounds;
    const vector<string> &args = curr_f.args();
    internal_assert(args.size() == region.size());
    for (size_t d = 0; d < args.size(); d++) {
        pure_bounds.emplace(args[d], region[d]);
    }

    DimBounds stage_bounds = get_stage_bounds(curr_f, stage, pure_bounds);
    return stage_region_cost(func, stage, stage_bounds, inlines);
}

Cost RegionCosts::region_cost(string func, const Box &region, const set<string> &inlines) {
    Function curr_f = get_element(env, func);
    Cost region_cost(0, 0);

    int num_stages = curr_f.updates().size() + 1;
    for (int s = 0; s < num_stages; s++) {
        Cost stage_cost = stage_region_cost(func, s, region, inlines);

        if (stage_cost.arith == unknown) {
            return Cost();
        } else {
            region_cost.arith += stage_cost.arith;
            region_cost.memory += stage_cost.memory;
        }
    }

    internal_assert(region_cost.arith != unknown && region_cost.memory != unknown);
    return region_cost;
}

Cost RegionCosts::region_cost(const map<string, Box> &regions, const set<string> &inlines){
    Cost total_cost(0, 0);
    for (const auto &f : regions) {
        // The cost for pure inlined functions will be accounted in the
        // consumer of the inlined function so they should be skipped.
        if (inlines.find(f.first) != inlines.end()) {
            internal_assert(get_element(env, f.first).is_pure());
            continue;
        }

        Cost cost = region_cost(f.first, f.second, inlines);
        if (cost.arith == unknown) {
            return Cost();
        } else {
            total_cost.arith += cost.arith;
            total_cost.memory += cost.memory;
        }
    }

    internal_assert((total_cost.arith != unknown) && (total_cost.memory != unknown));
    return total_cost;
}

map<string, int64_t>
RegionCosts::stage_detailed_load_costs(string func, int stage,
                                       const set<string> &inlines) {
    map<string, int64_t> load_costs;
    Function curr_f = get_element(env, func);
    Definition def = get_stage_definition(curr_f, stage);

    for (const auto &e : def.values()) {
        Expr inlined_expr = perform_inline(e, env, inlines);
        inlined_expr = simplify(inlined_expr);

        map<string, int64_t> expr_load_costs = compute_expr_detailed_byte_loads(inlined_expr);
        combine_load_costs(load_costs, expr_load_costs);
        load_costs[func] += e.type().bytes();
    }

    return load_costs;
}

map<string, int64_t>
RegionCosts::stage_detailed_load_costs(string func, int stage,
                                       DimBounds &bounds,
                                       const set<string> &inlines) {
    Function curr_f = get_element(env, func);
    Definition def = get_stage_definition(curr_f, stage);

    Box stage_region;

    const vector<Dim> &dims = def.schedule().dims();
    for (int d = 0; d < (int)dims.size() - 1; d++) {
        stage_region.push_back(get_element(bounds, dims[d].var));
    }

    map<string, int64_t> load_costs = stage_detailed_load_costs(func, stage, inlines);

    int64_t size = box_size(stage_region);
    for (auto &kv : load_costs) {
        if (kv.second == unknown) {
            continue;
        } else if (size == unknown) {
            kv.second = unknown;
        } else {
            kv.second *= size;
        }
    }

    return load_costs;
}

map<string, int64_t>
RegionCosts::detailed_load_costs(string func, const Box &region,
                                 const set<string> &inlines) {
    Function curr_f = get_element(env, func);
    map<string, int64_t> load_costs;

    int num_stages = curr_f.updates().size() + 1;

    DimBounds pure_bounds;
    const vector<string> &args = curr_f.args();
    internal_assert(args.size() == region.size());
    for (size_t d = 0; d < args.size(); d++) {
        pure_bounds.emplace(args[d], region[d]);
    }

    vector<DimBounds> stage_bounds = get_stage_bounds(curr_f, pure_bounds);

    for (int s = 0; s < num_stages; s++) {
        map<string, int64_t> stage_load_costs = stage_detailed_load_costs(func, s, inlines);

        Definition def = get_stage_definition(curr_f, s);

        Box stage_region;

        const vector<Dim> &dims = def.schedule().dims();
        for (int d = 0; d < (int)dims.size() - 1; d++) {
            stage_region.push_back(get_element(stage_bounds[s], dims[d].var));
        }

        int64_t size = box_size(stage_region);
        for (auto &kv : stage_load_costs) {
            if (kv.second == unknown) {
                continue;
            } else if (size == unknown) {
                kv.second = unknown;
            } else {
                kv.second *= size;
            }
        }

        combine_load_costs(load_costs, stage_load_costs);
    }

    return load_costs;
}

map<string, int64_t>
RegionCosts::detailed_load_costs(const map<string, Box> &regions,
                                 const set<string> &inlines) {
    map<string, int64_t> load_costs;
    for (const auto &r : regions) {
        // The cost for pure inlined functions will be accounted in the
        // consumer of the inlined function so they should be skipped.
        if (inlines.find(r.first) != inlines.end()) {
            internal_assert(get_element(env, r.first).is_pure());
            continue;
        }

        map<string, int64_t> partial_load_costs = detailed_load_costs(r.first, r.second, inlines);
        combine_load_costs(load_costs, partial_load_costs);
    }

    return load_costs;
}

Cost RegionCosts::get_func_stage_cost(const Function &f, int stage, const set<string> &inlines) {
    if (f.has_extern_definition()) {
        return Cost();
    }

    Definition def = get_stage_definition(f, stage);

    Cost cost(0, 0);

    for (const auto &e : def.values()) {
        Expr inlined_expr = perform_inline(e, env, inlines);
        inlined_expr = simplify(inlined_expr);

        Cost expr_cost = compute_expr_cost(inlined_expr);
        cost.arith += expr_cost.arith;
        cost.memory += expr_cost.memory;

        // Accounting for the store
        cost.memory += e.type().bytes();
        cost.arith += 1;
    }

    if (!f.is_pure()) {
        for (const auto &arg : def.args()) {
            Expr inlined_arg = perform_inline(arg, env, inlines);
            inlined_arg = simplify(inlined_arg);

            Cost expr_cost = compute_expr_cost(inlined_arg);
            cost.arith += expr_cost.arith;
            cost.memory += expr_cost.memory;
        }
    }

    return cost;
}

vector<Cost> RegionCosts::get_func_cost(const Function &f, const set<string> &inlines) {
    if (f.has_extern_definition()) {
        return { Cost() };
    }

    vector<Cost> func_costs;
    size_t num_stages = f.updates().size() + 1;
    for (size_t s = 0; s < num_stages; s++) {
        func_costs.push_back(get_func_stage_cost(f, s, inlines));
    }
    return func_costs;
}

int64_t RegionCosts::region_size(string func, const Box &region) {
    const Function &f = get_element(env, func);
    int64_t size = box_size(region);
    if (size == unknown) {
        return unknown;
    }
    int64_t size_per_ele = get_func_value_size(f);
    return size * size_per_ele;
}

int64_t RegionCosts::region_footprint(const map<string, Box> &regions,
                                      const set<string> &inlined) {
    map<string, int> num_consumers;
    for (const auto &f : regions) {
        num_consumers[f.first] = 0;
    }
    for (const auto &f : regions) {
        map<string, Function> prods = find_direct_calls(get_element(env, f.first));
        for (const auto &p : prods) {
            auto iter = num_consumers.find(p.first);
            if (iter != num_consumers.end()) {
                iter->second += 1;
            }
        }
    }

    vector<Function> outs;
    for (const auto &f : num_consumers) {
        if (f.second == 0) {
            outs.push_back(get_element(env, f.first));
        }
    }

    // Realization order
    vector<string> order = realization_order(outs, env);

    int64_t working_set_size = 0;
    int64_t curr_size = 0;

    map<string, int64_t> func_sizes;

    for (const auto &f : regions) {
        // Inlined functions do not have allocations
        bool is_inlined = inlined.find(f.first) != inlined.end();
        int64_t size = is_inlined ? 0 : region_size(f.first, f.second);
        if (size == unknown) {
            return unknown;
        } else {
            func_sizes.emplace(f.first, size);
        }
    }

    for (const auto &f : order) {
        if (regions.find(f) != regions.end()) {
            curr_size += get_element(func_sizes, f);
        }
        working_set_size = std::max(curr_size, working_set_size);
        map<string, Function> prods = find_direct_calls(get_element(env, f));
        for (const auto &p : prods) {
            auto iter = num_consumers.find(p.first);
            if (iter != num_consumers.end()) {
                iter->second -= 1;
                if (iter->second == 0) {
                    curr_size -= get_element(func_sizes, p.first);
                    internal_assert(curr_size >= 0);
                }
            }
        }
    }

    return working_set_size;
}

int64_t RegionCosts::input_region_size(string input, const Box &region) {
    int64_t size = box_size(region);
    if (size == unknown) {
        return unknown;
    }
    int64_t size_per_ele = get_element(inputs, input).bytes();
    return size * size_per_ele;
}

int64_t RegionCosts::input_region_size(const map<string, Box> &input_regions) {
    int64_t total_size = 0;
    for (const auto &reg : input_regions) {
        int64_t size = input_region_size(reg.first, reg.second);
        if (size == unknown) {
            return unknown;
        } else {
            total_size += size;
        }
    }
    return total_size;
}

void RegionCosts::disp_func_costs() {
    debug(0) << "===========================" << '\n';
    debug(0) << "Pipeline per element costs:" << '\n';
    debug(0) << "===========================" << '\n';
    for (const auto &kv : env) {
        int stage = 0;
        for (const auto &cost : func_cost[kv.first]) {
            Definition def = get_stage_definition(kv.second, stage);
            for (const auto &e : def.values()) {
                debug(0) << simplify(e) << '\n';
            }
            debug(0) << "(" << kv.first << ", " << stage << ") -> ("
                     << cost.arith << ", " << cost.memory << ")" << '\n';
            stage++;
        }
    }
    debug(0) << "===========================" << '\n';
}

bool is_func_trivial_to_inline(const Function &func) {
    if (!func.can_be_inlined()) {
        return false;
    }

    // For multi-dimensional tuple, we want to take the max over the arithmetic
    // and memory cost separately for conservative estimate.

    Cost inline_cost(0, 0);
    for (const auto &val : func.values()) {
        Cost cost = compute_expr_cost(val);
        inline_cost.arith = std::max(cost.arith, inline_cost.arith);
        inline_cost.memory = std::max(cost.memory, inline_cost.memory);
    }

    // Compute the cost if we were to call the function instead of inline it
    Cost call_cost(0, 0);
    call_cost.arith = 1;
    for (const auto &type : func.output_types()) {
        call_cost.memory = std::max((int64_t)type.bytes(), call_cost.memory);
    }

    bool is_trivial = (call_cost.arith + call_cost.memory) >= (inline_cost.arith + inline_cost.memory);
    return is_trivial;
}

}
}
