# Building
## Dependency
* a C++ compiler that supports at least C++20
* a SYCL 2020 implementation, such as [hipSYCL](https://github.com/illuhad/hipSYCL) and [intel/llvm](https://github.com/intel/llvm/)
  * if use hipSYCL, refer to [this guide](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md)
  * if use intel/llvm, version newer than `998fd91` (2022.11.07) is needed. Refer to [this guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for installation.
* Boost libraries
  * the version of Boost libraries required may be newer than that provided by system package manager (see below "BOOST_INLINE and HIP conflicts")
* hwloc (optional)
* FFTW 3 (optional)
* Qt 5 (optional)
  * Qt 6 may work, but not tested yet
* Python 3 with development headers & matplotlib

if ROCm backend enabled, additional dependencies:
* ROCm
* hipfft
* rocfft

if CUDA backend enabled, additional dependencies:
* CUDA toolkit
* cufft

if MUSA backend enabled, additional dependencies:
* MUSA toolkit
* mufft

## Compiler requirements
SYCL compilers used should support these C++ / SYCL features:
* C++ 20, for concept, `std::identity`, etc.
* unified address part of SYCL USM 
  * i.e. managed / shared memory is not required, but may useful for device with limited VRAM
* `parallel_for` with `nd_range`
  * for emulation of `sycl::reduction` using modified SYCL Parallel STL
* ... maybe others

## Building System
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
* for MUSA and OpenCL, should auto detect; or use `SRTB_ENABLE_MUSA`, `SRTB_ENABLE_OPENCL` to force
* or for hipSYCL the user may define `HIPSYCL_TARGETS` directly, see [doc of this variable](https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/using-hipsycl.md)

Example configure command: (assuming project path is `$PROJECT_PATH`)

* using hipSYCL:
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=hipSYCL \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ENABLE_ROCM=ON -DSRTB_ROCM_ARCH=gfx906 \
-DSRTB_ENABLE_MUSA=OFF -DSRTB_ENABLE_OPENCL=OFF \
$PROJECT_PATH
```

or

```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=hipSYCL -DHIPSYCL_TARGETS="hip:gfx1035;omp" $PROJECT_PATH
```

```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=hipSYCL -DHIPSYCL_TARGETS="generic;omp" $PROJECT_PATH
```

* using intel/llvm: (note C++ compiler is explicitly set here, assuming intel/llvm is installed at `/opt/intel-llvm`)
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=intel-llvm \
-DCMAKE_C_COMPILER=/opt/intel-llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/intel-llvm/bin/clang++ \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
$PROJECT_PATH
```

### Compile-time configs
There are several compile-time configurations in `srtb/config.hpp`, some may be toggled for specific setup, e.g.
* `srtb::real` = `float` or `double`, usually `float` for GPU (for less VRAM usage & avoid restriction of `double` FLOPs by some vendor)
* `SRTB_USE_USM_SHARED_MEMORY`, define this to use `sycl::usm::shared` for device memory, or don't define it to use `sycl::usm::device` for device memory (in case shared memory is not supported)

Refer to this source file for more configurations and their docs.

## Workarounds
### 1. BOOST_INLINE and HIP conflicts
See [Boost.Config issue 392](https://github.com/boostorg/config/issues/392) , which means if compile for ROCm, Boost 1.80+ may be needed.

You may use CMake configure option `BOOST_ROOT` to set the Boost library used, i.e. add `-DBOOST_ROOT=<path to installed Boost libraries>` to configure command above.

### 2. configure error: "clangrt builtins lib not found"
If compile with intel/llvm, ROCm/HIP may search 'clang_rt.builtins' in intel/llvm, but this module isn't built by default. 

To fix this, Add
```bash
--llvm-external-projects "compiler-rt"
```
when executing `buildbot/configure.py`.

Also add `openmp` if needed, e.g. use intel/llvm as a compiler for hipSYCL

### 3. cannot load shared libraries
When executing main program, some shared libraries may not be able to load:
```
simple-radio-telescope-backend: error while loading shared libraries: libsycl.so.6: cannot open shared object file: No such file or directory
```
One cause is that intel/llvm is not installed to a location where `ldconfig` recognizes.

To fix this,
```bash
export LD_LIBRARY_PATH=/opt/intel-llvm/lib:$LD_LIBRARY_PATH
```
where `/opt/intel-llvm` should be substituted by actural intel/llvm installation location.
