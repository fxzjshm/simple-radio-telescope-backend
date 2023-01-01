# Building
## Dependency
* a C++ compiler that supports at least C++20
* SYCL 2020 implementation, such as [illuhad/hipSYCL](https://github.com/illuhad/hipSYCL/) and [intel/llvm](https://github.com/intel/llvm/)
  * if use hipSYCL, refer to [this guide](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md)
  * if use intel/llvm, version newer than `998fd91` (2022.11.07) is needed. Refer to [this guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for installation.
* Boost libraries
  * the version of Boost libraries required may be newer than that provided by system package manager (see below "BOOST_INLINE and HIP conflicts")
* hwloc
* FFTW 3
* Qt 5

if ROCm backend enabled, additional dependencies:
* ROCm
* hipfft
* rocfft

if CUDA backend enabled, additional dependencies:
* CUDA toolkit
* cufft

## Building
This project uses CMake 3. 

Configure options:
* `SRTB_SYCL_IMPLEMENTATION`: switches SYCL implementation used. Default to `hipSYCL`.
  * set to `hipSYCL` to use hipSYCL
  * set to `intel-llvm` to use intel/llvm
    * additionally, `CMAKE_C_COMPILER` & `CMAKE_CXX_COMPILER` should be set to intel/llvm installation (see example below)
* `SRTB_ENABLE_ROCM`: `ON` or `OFF`
* `SRTB_ROCM_ARCH`:
  * the arch of target GPU, e.g. `gfx906` or `gfx1030`, required if `SRTB_ENABLE_ROCM` is `ON`, otherwise no effect
* `SRTB_ENABLE_CUDA`: `ON` or `OFF`
* `SRTB_CUDA_ARCH`:
  * the arch of target GPU, e.g. `sm_86`, required if `SRTB_ENABLE_CUDA` is set `ON`, otherwise no effect

Example configure command: (assuming project path is `$PROJECT_PATH`)

* using hipSYCL:
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=hipSYCL \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
$PROJECT_PATH
```

* using intel/llvm: (note C++ compiler is explicitly set here, assuming intel/llvm is installed at `/opt/intel-llvm`)
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=intel-llvm \
-DCMAKE_C_COMPILER=/opt/intel-llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/intel-llvm/bin/clang++ \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
$PROJECT_PATH
```

### Workarounds
#### 1. BOOST_INLINE and HIP conflicts
See [Boost.Config issue 392](https://github.com/boostorg/config/issues/392) , which means if compile for ROCm, Boost 1.80+ may be needed.

You may use CMake configure option `BOOST_ROOT` to set the Boost library used, i.e. add `-DBOOST_ROOT=<path to installed Boost libraries>` to configure command above.

#### 2. configure error: "clangrt builtins lib not found"
If compile with intel/llvm, ROCm/HIP may search 'clang_rt.builtins' in intel/llvm, but this module isn't built by default. 

A patch to `buildbot/configure.py` is
```diff
diff --git a/buildbot/configure.py b/buildbot/configure.py
index f3a43857b7..08cb75e5e3 100644
--- a/buildbot/configure.py
+++ b/buildbot/configure.py
@@ -13,7 +13,7 @@ def do_configure(args):
     if not os.path.isdir(abs_obj_dir):
       os.makedirs(abs_obj_dir)
 
-    llvm_external_projects = 'sycl;llvm-spirv;opencl;xpti;xptifw'
+    llvm_external_projects = 'sycl;llvm-spirv;opencl;xpti;xptifw;compiler-rt'
 
     # libdevice build requires a working SYCL toolchain, which is not the case
     # with macOS target right now.
```
Also add `openmp` if needed, e.g. use intel/llvm as a compiler for hipSYCL
