# Simple radio telescope backend
Everything working in progress...

This is a simple backend (maybe terminal) of radio telescope. 
It reads raw "baseband" data and should be capable of coherent dedispersion, maybe in real-time.
Future plans include Fast Radio Burst (FRB) detection and maybe pulsar folding.

Due to vendor neutrality and current status of some heterogeneous computing APIs (I mean OpenCL, IMHO),
**[SYCL 2020](https://www.khronos.org/sycl/)** from Khronos Group is chosen as target API.

Although say so, currently only CPU (OpenMP, on amd64), ROCm and CUDA backends are tested, due to limited device type available.

## Dependency
* a C++ compiler that supports at least C++20
* SYCL 2020 implementation, such as [illuhad/hipSYCL](https://github.com/illuhad/hipSYCL/) and [intel/llvm](https://github.com/intel/llvm/)
  * if use hipSYCL, refer to [this guide](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md)
  * if use intel/llvm, version newer than `998fd91` (2022.11.07) is needed. Refer to [this guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for installation.
* Boost libraries
* fftw3
* Qt5

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
    * additionally, `CMAKE_C_COMPILER` & `CMAKE_CXX_COMPILER` should be set to intel/llvm installation
* `SRTB_ENABLE_ROCM`: `ON` or `OFF`
* `SRTB_ROCM_ARCH`:
  * if `SRTB_ENABLE_ROCM` is set `ON`, for both hipSYCL and intel/llvm, `SRTB_ROCM_ARCH` is required, which is the arch of target GPU, e.g. `gfx906` or `gfx1030`.
* `SRTB_ENABLE_CUDA`: `ON` or `OFF`
* `SRTB_CUDA_ARCH`:
  * if `SRTB_ENABLE_CUDA` is set `ON` and using hipSYCL, `SRTB_CUDA_ARCH` is required, which is the arch of target GPU, e.g. `sm_86`
  * if using intel/llvm, `SRTB_CUDA_ARCH` is optional

Example configure command:  

* using hipSYCL:
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=hipSYCL \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
-DBOOST_ROOT=/opt/boost \
~/workspace/simple-radio-telescope-backend
```

* using intel/llvm:
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=intel-llvm \
-DCMAKE_C_COMPILER=/opt/intel-llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/intel-llvm/bin/clang++ \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
-DBOOST_ROOT=/opt/boost \
~/workspace/simple-radio-telescope-backend
```

## Code structure
* `userspace/include/srtb/`
  * `config`: compile-time and runtime configurations
  * `work`: defines input of each pipe
  * `global_variables`: stores *almost* all global variables, mainly work queues of pipes (TODO: better ways?)
  * `pipeline/`: components of the pipeline
    * each pipe defines its input work type in `work.hpp`, reads work from the `work_queue` defined in `global_variables.hpp`, do some transformations on the data, and wrap it as the work type of next pipe.
  * `fft/`: wrappers of FFT libraries like fftw, cufft and hipfft
  * `gui/`: user interface to show spectrum, based on Qt5
  * `io/`: read raw "baseband" data
    * `udp_receiver`: from UDP packets using Boost.Asio
    * `file`: from file
    * `rdma`: (TODO, is this needed?) maybe operate a custom driver to read data from network device, then directly transfer to GPU using Direct Memory Access or PCIe Peer to Peer or something likel this.
  * others function as their name indicates
* `userspace/src/`: `main` starts pipes required.
* `userspace/tests/`: test component shown above.
    

## Workarounds
#### 1. BOOST_INLINE and HIP conflicts
See [Boost.Config issue 392](https://github.com/boostorg/config/issues/392) , which means if compile for ROCm, Boost 1.80+ may be needed.

You may use CMake configure option `BOOST_ROOT` to set the Boost library used.

#### 2. configure error: "clangrt builtins lib not found"
If compile with intel/llvm, HIP may search 'clang_rt.builtins' in intel/llvm, but this module isn't built by default. 

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

## License
Main part of this program is licensed under [Mulan Public License, Version 2](http://license.coscl.org.cn/MulanPubL-2.0/index.html) .  

Please notice that Mulan Public License (MulanPubL) is different from Mulan Permissive License (MulanPSL). The former, which this program uses, is more of GPL-like.

## Credits
This repo also contains some 3rd-party code:
* `exprgrammar.hpp` from [Suzerain](https://bitbucket.org/RhysU/suzerain) by RhysU, licensed under [Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/) .

