# Ensure that make with no arguments makes the right thing
all:

include ../support/Makefile.inc

# This app requires a separate toolchain to be built from the Android NDK,
# using the make-standalone-toolchain.sh script:
#$ build/tools/make-standalone-toolchain.sh --arch=arm64 --platform=android-21 --install-dir=$ANDROID_ARM64_TOOLCHAIN
#$ build/tools/make-standalone-toolchain.sh --arch=arm --platform=android-21 --install-dir=$ANDROID_ARM_TOOLCHAIN
CXX-host ?= $(CXX)
CXX-arm-64-android ?= $(ANDROID_ARM64_TOOLCHAIN)/bin/aarch64-linux-android-c++
CXX-arm-32-android ?= $(ANDROID_ARM_TOOLCHAIN)/bin/arm-linux-androideabi-c++
CXX-arm-64-profile-android ?= $(CXX-arm-64-android)
CXX-arm-32-profile-android ?= $(CXX-arm-32-android)

CXXFLAGS-host ?=
CXXFLAGS-arm-64-android ?=
CXXFLAGS-arm-32-android ?=

LDFLAGS-host ?= -lpthread -ldl -lm
LDFLAGS-arm-64-android ?= -llog -fPIE -pie
LDFLAGS-arm-32-android ?= -llog -fPIE -pie
LDFLAGS-arm-64-profile-android ?= -llog -fPIE -pie
LDFLAGS-arm-32-profile-android ?= -llog -fPIE -pie

BIN ?= bin

all: $(BIN)/process-arm-64-android 

$(BIN)/pipeline: pipeline.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/%/pipeline_cpu.o: $(BIN)/pipeline
	@mkdir -p $(@D)
	$^ -g blur -o $(BIN)/$* -e o,h,html -f pipeline_cpu  target=$*

$(BIN)/%/pipeline_gpu.o: $(BIN)/pipeline
	@mkdir -p $(@D)
	$^ -g blur -o $(BIN)/$* -e o,h,html -f pipeline_gpu  target=$*-opencl

$(BIN)/process-%: process.cpp $(BIN)/%/pipeline_cpu.o $(BIN)/%/pipeline_gpu.o ion_allocation.cpp
	$(CXX-$*) $(CXXFLAGS) $(CXXFLAGS-$*) -I $(BIN)/$* -Wall -O3 process.cpp  ion_allocation.cpp $(BIN)/$*/pipeline_cpu.o $(BIN)/$*/pipeline_gpu.o -o $(BIN)/process-$* $(LDFLAGS-$*)

clean:
	rm -rf $(BIN)
