#include "SlidingWindow.h"
#include "Bounds.h"
#include "Debug.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Monotonic.h"
#include "Scope.h"
#include "Simplify.h"
#include "Solve.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::map;
using std::string;

namespace {

// Does an expression depend on a particular variable?
class ExprDependsOnVar : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Variable *op) {
        if (op->name == var) result = true;
    }

    void visit(const Let *op) {
        op->value.accept(this);
        // The name might be hidden within the body of the let, in
        // which case there's no point descending.
        if (op->name != var) {
            op->body.accept(this);
        }
    }
public:

    bool result;
    string var;

    ExprDependsOnVar(string v) : result(false), var(v) {
    }
};

bool expr_depends_on_var(Expr e, string v) {
    ExprDependsOnVar depends(v);
    e.accept(&depends);
    return depends.result;
}

class ExpandExpr : public IRMutator2 {
    using IRMutator2::visit;
    const Scope<Expr> &scope;

    Expr visit(const Variable *var) override {
        if (scope.contains(var->name)) {
            Expr expr = scope.get(var->name);
            debug(4) << "Fully expanded " << var->name << " -> " << expr << "\n";
            return expr;
        } else {
            return var;
        }
    }

public:
    ExpandExpr(const Scope<Expr> &s) : scope(s) {}

};

// Perform all the substitutions in a scope
Expr expand_expr(Expr e, const Scope<Expr> &scope) {
    ExpandExpr ee(scope);
    Expr result = ee.mutate(e);
    debug(4) << "Expanded " << e << " into " << result << "\n";
    return result;
}

// Determines if the visited statement (which should a loop) is safe to extend
// for more iterations while wrapping every Provide statement into an If stmt
// that does nothing on the added iterations. Extending is not safe if any
// impure calls or operations that can fault occur outside Provide (and so
// outside of the guarding condition). It's also not safe if there's more than
// one Produce for some function, since we track the offset for the guards only
// at function granularity. Finally, it's hard to figure out what parts of a
// fused loop need to run on the additional iterations, so for now we don't try.
class LoopSafeToExtend : public IRVisitor {
    map<string, int> produce_count;
    bool inside_provide = false;
    bool all_pure = true;
    bool has_fused_loops = false;

    using IRVisitor::visit;

    void visit(const Provide *op) override {
        ScopedValue<bool> sb(inside_provide, true);
        IRVisitor::visit(op);
    }

    void visit(const ProducerConsumer *op) override {
        if (op->is_producer) {
            ++produce_count[op->name];
            if (const auto *nested = op->body.as<ProducerConsumer>()) {
                if (nested->is_producer) {
                    has_fused_loops = true;
                }
            }
        }
        IRVisitor::visit(op);
    }

    void visit(const Call *op) override {
        if (!inside_provide &&
            (op->call_type == Call::Image || op->call_type == Call::Halide ||
             !op->is_pure())) {
            all_pure = false;
        }
        IRVisitor::visit(op);
    }

    // Detects integer division by a non-constant, to avoid adding iterations
    // that might divide by zero.
    void visit(const Div *op) override {
        if (!inside_provide && !op->type.is_float() && !is_const(op->b)) {
            all_pure = false;
        }
        IRVisitor::visit(op);
    }

    void visit(const Load *op) {
        all_pure = false;
    }

    void visit(const Store *op) {
        all_pure = false;
    }

public:

    bool is_safe() const {
        if (!all_pure || has_fused_loops) {
            return false;
        }
        for (const auto &pc : produce_count) {
            if (pc.second > 1) {
                return false;
            }
        }
        return true;
    }
};

// Describes how to extend a single loop by adding iterations at the start.
struct LoopExtension {
    // Number of iterations we add to the start of the loop. Always positive.
    // This is the maximum across all per-Func increases.
    int64_t max_increase;
    // Map from a Func name to the number of iterations we add to the loop to
    // align the realization of this Func with its consumer.
    map<string, int64_t> increase_per_func;

    LoopExtension() : max_increase(0) {}

    // Records the intent to extend the loop by the given number of iterations
    // to align the specified Func.
    void extend_for_func(const string& func, int64_t increase) {
        internal_assert(increase > 0);
        increase_per_func.insert(std::make_pair(func, increase));
        max_increase = std::max(max_increase, increase);
    }

    // Returns the number of iterations of the extended loop to skip when
    // computing the values of the given Func. If the Func is not one of those
    // we wanted to align, we skip all additional iterations.
    int64_t first_iteration_for_func(const string &func) const {
        auto it = increase_per_func.find(func);
        if (it == increase_per_func.end()) {
            return max_increase;
        } else {
            return max_increase - it->second;
        }
    }
};

// Perform sliding window optimization for a function over a
// particular serial for loop
class SlidingWindowOnFunctionAndLoop : public IRMutator2 {
    Function func;
    string loop_var;
    Expr loop_min;
    bool can_extend;
    const LoopExtension &extension;
    Scope<Expr> &scope;

    map<string, Expr> replacements;

    using IRMutator2::visit;

    // Check if the dimension at index 'dim_idx' is always pure (i.e. equal to 'dim')
    // in the definition (including in its specializations)
    bool is_dim_always_pure(const Definition &def, const string& dim, int dim_idx) {
        const Variable *var = def.args()[dim_idx].as<Variable>();
        if ((!var) || (var->name != dim)) {
            return false;
        }

        for (const auto &s : def.specializations()) {
            bool pure = is_dim_always_pure(s.definition, dim, dim_idx);
            if (!pure) {
                return false;
            }
        }
        return true;
    }

    // Returns positive number of iterations to add to the start of the loop so
    // that those iterations cover the overall region starting at min_required
    // while the inner loop starts at min_computed for every iteration. Returns
    // zero if we cannot extend the loop or cannot compute a constant number of
    // iterations to add.
    int64_t try_to_extend_loop(Expr min_required, Expr min_computed) {
        if (!can_extend) {
            return 0;
        }
        // Handling functions with updates should be possible, but left out
        // for now for simplicity.
        if (!func.updates().empty()) {
            return 0;
        }
        // Find the overall min of the required regions for all iterations;
        // because of the monotonicity, it occurs on the first iteration.
        Expr overall_min_required = substitute(loop_var, loop_min, min_required);
        // Find the largest value of the loop variable that results in
        // the new loop starting at or below the overall min.
        Interval loop_var_low_enough =
            solve_for_inner_interval(min_computed <= overall_min_required, loop_var);
        if (!loop_var_low_enough.has_upper_bound()) {
            return 0;
        }
        auto extra_iterations = simplify(loop_min - loop_var_low_enough.max);
        auto increase = as_const_int(extra_iterations);
        return increase ? *increase : 0;
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (!op->is_producer) {
            // In this scope, we can only use additional iterations to align
            // inner loops if the function we are consuming is actually computed
            // on those iterations.
            bool can_extend_here = can_extend &&
                extension.increase_per_func.count(op->name) != 0;
            ScopedValue<bool> can_extend_scope(can_extend, can_extend_here);
            return IRMutator2::visit(op);
        } else if (op->name != func.name()) {
            return IRMutator2::visit(op);
        } else {
            Stmt stmt = op;

            // We're interested in the case where exactly one of the
            // dimensions of the buffer has a min/extent that depends
            // on the loop_var.
            string dim = "";
            int dim_idx = 0;
            Expr min_required, max_required;

            debug(3) << "Considering sliding " << func.name()
                     << " along loop variable " << loop_var << "\n"
                     << "Region provided:\n";

            string prefix = func.name() + ".s" + std::to_string(func.updates().size()) + ".";
            const std::vector<string> func_args = func.args();
            for (int i = 0; i < func.dimensions(); i++) {
                // Look up the region required of this function's last stage
                string var = prefix + func_args[i];
                internal_assert(scope.contains(var + ".min") && scope.contains(var + ".max"));
                Expr min_req = scope.get(var + ".min");
                Expr max_req = scope.get(var + ".max");
                min_req = expand_expr(min_req, scope);
                max_req = expand_expr(max_req, scope);

                debug(3) << func_args[i] << ":" << min_req << ", " << max_req  << "\n";
                if (expr_depends_on_var(min_req, loop_var) ||
                    expr_depends_on_var(max_req, loop_var)) {
                    if (!dim.empty()) {
                        dim = "";
                        min_required = Expr();
                        max_required = Expr();
                        break;
                    } else {
                        dim = func_args[i];
                        dim_idx = i;
                        min_required = min_req;
                        max_required = max_req;
                    }
                } else if (!min_required.defined() &&
                           i == func.dimensions() - 1 &&
                           is_pure(min_req) &&
                           is_pure(max_req)) {
                    // The footprint doesn't depend on the loop var. Just compute everything on the first loop iteration.
                    dim = func_args[i];
                    dim_idx = i;
                    min_required = min_req;
                    max_required = max_req;
                }
            }

            if (!min_required.defined()) {
                debug(3) << "Could not perform sliding window optimization of "
                         << func.name() << " over " << loop_var << " because multiple "
                         << "dimensions of the function dependended on the loop var\n";
                return stmt;
            }

            // If the function is not pure in the given dimension, give up. We also
            // need to make sure that it is pure in all the specializations
            bool pure = true;
            for (const Definition &def : func.updates()) {
                pure = is_dim_always_pure(def, dim, dim_idx);
                if (!pure) {
                    break;
                }
            }
            if (!pure) {
                debug(3) << "Could not performance sliding window optimization of "
                         << func.name() << " over " << loop_var << " because the function "
                         << "scatters along the related axis.\n";
                return stmt;
            }

            bool can_slide_up = false;
            bool can_slide_down = false;

            Monotonic monotonic_min = is_monotonic(min_required, loop_var);
            Monotonic monotonic_max = is_monotonic(max_required, loop_var);

            if (monotonic_min == Monotonic::Increasing ||
                monotonic_min == Monotonic::Constant) {
                can_slide_up = true;
            }

            if (monotonic_max == Monotonic::Decreasing ||
                monotonic_max == Monotonic::Constant) {
                can_slide_down = true;
            }

            if (!can_slide_up && !can_slide_down) {
                debug(3) << "Not sliding " << func.name()
                         << " over dimension " << dim
                         << " along loop variable " << loop_var
                         << " because I couldn't prove it moved monotonically along that dimension\n"
                         << "Min is " << min_required << "\n"
                         << "Max is " << max_required << "\n";
                return stmt;
            }

            // Ok, we've isolated a function, a dimension to slide
            // along, and loop variable to slide over.
            debug(3) << "Sliding " << func.name()
                     << " over dimension " << dim
                     << " along loop variable " << loop_var << "\n";

            Expr loop_var_expr = Variable::make(Int(32), loop_var);

            Expr prev_max_plus_one = substitute(loop_var, loop_var_expr - 1, max_required) + 1;
            Expr prev_min_minus_one = substitute(loop_var, loop_var_expr - 1, min_required) - 1;

            // If there's no overlap between adjacent iterations, we shouldn't slide.
            if (can_prove(min_required >= prev_max_plus_one) ||
                can_prove(max_required <= prev_min_minus_one)) {
                debug(3) << "Not sliding " << func.name()
                         << " over dimension " << dim
                         << " along loop variable " << loop_var
                         << " there's no overlap in the region computed across iterations\n"
                         << "Min is " << min_required << "\n"
                         << "Max is " << max_required << "\n";
                return stmt;
            }

            // Try to extend outer loop boundary to immediately enter steady state.
            if (monotonic_min == Monotonic::Increasing) {
                int64_t increase = try_to_extend_loop(min_required, prev_max_plus_one);
                if (increase > 0) {
                    // TODO: lower debug level.
                    debug(0) << "Aligning realization of " << func.name()
                             << " over dimension " << dim
                             << " along loop variable " << loop_var
                             << " by pushing loop_min back by: "
                             << increase << "\n";
                    loop_extent_increase = increase;
                    replacements[prefix + dim + ".min"] = prev_max_plus_one;
                    return stmt;
                }
            }

            Expr new_min, new_max;
            if (can_slide_up) {
                new_min = select(loop_var_expr <= loop_min, min_required, likely_if_innermost(prev_max_plus_one));
                new_max = max_required;
            } else {
                new_min = min_required;
                new_max = select(loop_var_expr <= loop_min, max_required, likely_if_innermost(prev_min_minus_one));
            }

            Expr early_stages_min_required = new_min;
            Expr early_stages_max_required = new_max;

            debug(3) << "Sliding " << func.name() << ", " << dim << "\n"
                     << "Pushing min up from " << min_required << " to " << new_min << "\n"
                     << "Shrinking max from " << max_required << " to " << new_max << "\n";

            // Now redefine the appropriate regions required
            if (can_slide_up) {
                replacements[prefix + dim + ".min"] = new_min;
            } else {
                replacements[prefix + dim + ".max"] = new_max;
            }

            for (size_t i = 0; i < func.updates().size(); i++) {
                string n = func.name() + ".s" + std::to_string(i) + "." + dim;
                replacements[n + ".min"] = Variable::make(Int(32), prefix + dim + ".min");
                replacements[n + ".max"] = Variable::make(Int(32), prefix + dim + ".max");
            }

            // Ok, we have a new min/max required and we're going to
            // rewrite all the lets that define bounds required. Now
            // we need to additionally expand the bounds required of
            // the last stage to cover values produced by stages
            // before the last one. Because, e.g., an intermediate
            // stage may be unrolled, expanding its bounds provided.
            if (!func.updates().empty()) {
                Box b = box_provided(op->body, func.name());
                if (can_slide_up) {
                    string n = prefix + dim + ".min";
                    Expr var = Variable::make(Int(32), n);
                    stmt = LetStmt::make(n, min(var, b[dim_idx].min), stmt);
                } else {
                    string n = prefix + dim + ".max";
                    Expr var = Variable::make(Int(32), n);
                    stmt = LetStmt::make(n, max(var, b[dim_idx].max), stmt);
                }
            }
            return stmt;
        }
    }

    Stmt visit(const For *op) override {
        // It's not safe to enter an inner loop whose bounds depend on
        // the var we're sliding over.
        Expr min = expand_expr(op->min, scope);
        Expr extent = expand_expr(op->extent, scope);
        if (is_one(extent)) {
            // Just treat it like a let
            Stmt s = LetStmt::make(op->name, min, op->body);
            s = mutate(s);
            // Unpack it back into the for
            const LetStmt *l = s.as<LetStmt>();
            internal_assert(l);
            return For::make(op->name, op->min, op->extent, op->for_type, op->device_api, l->body);
        } else if (is_monotonic(min, loop_var) != Monotonic::Constant ||
                   is_monotonic(extent, loop_var) != Monotonic::Constant) {
            debug(3) << "Not entering loop over " << op->name
                     << " because the bounds depend on the var we're sliding over: "
                     << min << ", " << extent << "\n";
            return op;
        } else {
            return IRMutator2::visit(op);
        }
    }

    Stmt visit(const LetStmt *op) override {
        ScopedBinding<Expr> bind(scope, op->name, simplify(expand_expr(op->value, scope)));
        Stmt new_body = mutate(op->body);

        Expr value = op->value;

        map<string, Expr>::iterator iter = replacements.find(op->name);
        if (iter != replacements.end()) {
            value = iter->second;
            replacements.erase(iter);
        }

        if (new_body.same_as(op->body) && value.same_as(op->value)) {
            return op;
        } else {
            return LetStmt::make(op->name, value, new_body);
        }
    }

public:
    SlidingWindowOnFunctionAndLoop(Function f, string v, Expr v_min,
                                   bool can_extend, const LoopExtension &ext,
                                   Scope<Expr> &scope)
        : func(f), loop_var(v), loop_min(v_min),
          can_extend(can_extend), extension(ext),
          scope(scope), loop_extent_increase(0) {
    }

    int loop_extent_increase;
};

// Adds guarding If statements so that additional loop iterations skip over
// Provides of Funcs that we don't want to compute on those iterations.
class GuardAddedIterations : public IRMutator2 {
    const LoopExtension extension;
    const string loop_var;
    const string loop_min_var;
    const Expr loop_min;
    int64_t first_iteration;

    using IRMutator2::visit;

    // Tracks the first iteration based on the Func we are producing.
    Stmt visit(const ProducerConsumer *op) override {
        if (!op->is_producer) {
            return IRMutator2::visit(op);
        }
        int64_t new_first_iteration =
            std::min(first_iteration, extension.first_iteration_for_func(op->name));
        ScopedValue<int64_t> first_iter_scope(first_iteration, new_first_iteration);
        return IRMutator2::visit(op);
    }

    // Adds guarding Ifs around Provides.
    Stmt visit(const Provide *op) override {
        Stmt result = IRMutator2::visit(op);
        if (first_iteration == 0) {
            return result;
        }
        // loop_var >= loop_min + first_iteration.
        Expr cond = Variable::make(loop_min.type(), loop_var) >= loop_min +
            IntImm::make(loop_min.type(), first_iteration);
        return IfThenElse::make(cond, result);
    }

    // Adjusts all references to the old loop_min.
    Expr visit(const Variable *op) override {
        if (op->name == loop_min_var) {
            return Add::make(op, IntImm::make(op->type, extension.max_increase));
        }
        return IRMutator2::visit(op);
    }

public:
    GuardAddedIterations(const LoopExtension &e, const string &loop_var,
                         const Expr &loop_min)
       : extension(e),
         loop_var(loop_var),
         loop_min_var(loop_var + ".loop_min"),
         loop_min(loop_min),
         first_iteration(e.max_increase) {}
};

// Perform sliding window optimization
class SlidingWindow : public IRMutator2 {
    const map<string, Function> &env;
    Scope<Expr> scope;
    std::vector<string> funcs;
    map<string, int64_t> adjustments;

    using IRMutator2::visit;

    Stmt visit(const Realize *op) override {
        // Find the args for this function
        map<string, Function>::const_iterator iter = env.find(op->name);

        // If it's not in the environment it's some anonymous
        // realization that we should skip (e.g. an inlined reduction)
        if (iter == env.end()) {
            return IRMutator2::visit(op);
        }

        // If the Function in question has the same compute_at level
        // as its store_at level, skip it.
        const FuncSchedule &sched = iter->second.schedule();
        if (sched.compute_level() == sched.store_level()) {
            return IRMutator2::visit(op);
        }

        funcs.push_back(op->name);
        Stmt s = IRMutator2::visit(op);
        funcs.pop_back();
        return s;
    }

    Stmt visit(const For *op) override {
        Stmt new_body = op->body;
        new_body = mutate(new_body);

        debug(3) << " Doing sliding window analysis over loop: " << op->name << "\n";

        Expr extent = simplify(expand_expr(op->extent, scope));

        if (!is_one(extent) && (op->for_type == ForType::Serial ||
                                op->for_type == ForType::Unrolled)) {
            LoopSafeToExtend safe_to_extend;
            op->accept(&safe_to_extend);

            LoopExtension ext;
            for (const string &fn : funcs) {
                auto it = env.find(fn);
                internal_assert(it != env.end());
                SlidingWindowOnFunctionAndLoop mut(it->second, op->name, op->min,
                                                   safe_to_extend.is_safe(),
                                                   ext,
                                                   scope);
                new_body = mut.mutate(new_body);
                if (mut.loop_extent_increase > 0) {
                    ext.extend_for_func(fn, mut.loop_extent_increase);
                }
            }

            if (ext.max_increase > 0) {
                adjustments[op->name + ".loop_min"] = -ext.max_increase;
                adjustments[op->name + ".loop_extent"] = ext.max_increase;
                new_body = GuardAddedIterations(ext, op->name, op->min).mutate(new_body);
            }
        }

        if (new_body.same_as(op->body)) {
            return op;
        } else {
            return For::make(op->name, op->min, op->extent, op->for_type, op->device_api, new_body);
        }
    }

    Stmt visit(const LetStmt *op) override {
        ScopedBinding<Expr> bind(scope, op->name, simplify(expand_expr(op->value, scope)));
        Stmt new_body = mutate(op->body);
        Expr value = mutate(op->value);

        auto it = adjustments.find(op->name);
        if (it != adjustments.end()) {
            value = value + IntImm::make(value.type(), it->second);
            adjustments.erase(it);
        }

        if (new_body.same_as(op->body) && value.same_as(op->value)) {
            return op;
        } else {
            return LetStmt::make(op->name, value, new_body);
        }
    }

public:
    SlidingWindow(const map<string, Function> &e) : env(e) {}
};

}  // namespace

Stmt sliding_window(Stmt s, const map<string, Function> &env) {
    return SlidingWindow(env).mutate(s);
}

}  // namespace Internal
}  // namespace Halide
