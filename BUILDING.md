# Building
## Dependency
* a C++ compiler that supports at least C++20
* a SYCL 2020 implementation, such as [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) and Intel oneAPI DPC++ (alternatively, the open-source [Intel LLVM](https://github.com/intel/llvm/))
  * if use AdaptiveCpp, refer to [this guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md)
  * if use Intel LLVM, refer to [this guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) for installation.
* CMake
* Boost libraries
  * the version of Boost libraries required may be newer than that provided by system package manager (see below "BOOST_INLINE and HIP conflicts")
  * Debian-like: `libboost-all-dev`
  * RHEL-like: `boost-devel`
* hwloc (optional)
* FFTW 3 (optional)
  * Debian-like: `libfftw3-dev`
  * RHEL-like: `fftw-devel`
* Qt 5 (optional)
  * Debian-like: `qtdeclarative5-dev` `qml-module-qtquick2` `qml-module-qtquick-window2` `qml-module-qtquick-controls` `qml-module-qtquick-layouts`
  * RHEL-like: `qt5-qtdeclarative-devel`
  * Qt 6 not working (TODO)

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

The CI config (`.circleci/config.yml`) contains an example of setup procedures 
fron scratch up for CPU-only build.

## Compiler requirements
SYCL compilers used should support these C++ / SYCL features:
* C++ 23, for multi-dimensional subscript, concept, `std::identity`, etc.
* unified address part of SYCL USM 
  * i.e. managed / shared memory is not required, but may useful for device with limited VRAM
* `parallel_for` with `nd_range`
  * for emulation of `sycl::reduction` using modified SYCL Parallel STL
* ... maybe others

## Building System
This project uses CMake 3. 

Configure options:
* `SRTB_SYCL_IMPLEMENTATION`: switches SYCL implementation used. Default to `AdaptiveCpp`.
  * set to `AdaptiveCpp` to use AdaptiveCpp
  * set to `oneAPI` to use Intel oneAPI DPC++ or self-compiled Intel LLVM
    * additionally, `CMAKE_C_COMPILER` & `CMAKE_CXX_COMPILER` should be set to `icx`, `icpx` or `clang`, `clang++`, respectively (see example below)
* `SRTB_ENABLE_ROCM`: `ON` or `OFF`, should auto detect
* `SRTB_ROCM_ARCH`:
  * the arch of target GPU, e.g. `gfx906` or `gfx1030`; if set, will trigger ahead-of-time compile for this device.
* `SRTB_ENABLE_CUDA`: `ON` or `OFF`, should auto detect
* `SRTB_CUDA_ARCH`:
  * the arch of target GPU, e.g. `sm_86`; if set, will trigger ahead-of-time compile for this device.
* for MUSA and OpenCL, should auto detect; or use `SRTB_ENABLE_MUSA`, `SRTB_ENABLE_OPENCL` to force enable/disable
* or for AdaptiveCpp the user may define `ACPP_TARGETS` directly, see [doc of this variable](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-hipsycl.md)

Example configure command: (assuming project path is `$PROJECT_PATH`)

* using AdaptiveCpp:
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=AdaptiveCpp \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ENABLE_ROCM=ON -DSRTB_ROCM_ARCH=gfx906 \
-DSRTB_ENABLE_MUSA=OFF -DSRTB_ENABLE_OPENCL=OFF \
$PROJECT_PATH
```

or

```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=AdaptiveCpp -DACPP_TARGETS="hip:gfx1035;omp" $PROJECT_PATH
```

```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=AdaptiveCpp -DACPP_TARGETS="generic;omp" $PROJECT_PATH
```

* using Intel oneAPI DPC++: (note C++ compiler is explicitly set here)
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=oneAPI \
-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
$PROJECT_PATH
```

* using self-compiled Intel LLVM: (note C++ compiler is explicitly set here, assuming intel/llvm is installed at `/opt/intel-llvm`)
```bash
cmake -DSRTB_SYCL_IMPLEMENTATION=oneAPI \
-DCMAKE_C_COMPILER=/opt/intel-llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/intel-llvm/bin/clang++ \
-DSRTB_ENABLE_CUDA=OFF -DSRTB_ENABLE_ROCM=ON -DSRTB_CUDA_ARCH=sm_86 -DSRTB_ROCM_ARCH=gfx906 \
$PROJECT_PATH
```

### Compile-time configs
There are several compile-time configurations in `srtb/config.hpp`, some may be toggled for specific setup, e.g.
* `SRTB_USE_USM_SHARED_MEMORY`, define this to use `sycl::usm::shared` for device memory, or don't define it to use `sycl::usm::device` for device memory (in case shared memory is not supported)

Refer to this source file for more configurations and their docs.

## Workarounds
### 1. BOOST_INLINE and HIP conflicts
See [Boost.Config issue 392](https://github.com/boostorg/config/issues/392) , which means if compile for ROCm, Boost 1.80+ may be needed.

You may use CMake configure option `BOOST_ROOT` to set the Boost library used, i.e. add `-DBOOST_ROOT=<path to installed Boost libraries>` to configure command above.

### 2. configure error: "clangrt builtins lib not found"
If compile with self-built intel/llvm, ROCm/HIP may search 'clang_rt.builtins' in intel/llvm, but this module isn't built by default. 

To fix this, Add
```bash
--llvm-external-projects "compiler-rt"
```
when executing `buildbot/configure.py`.

Also add `openmp` if needed, e.g. use intel/llvm as a compiler for AdaptiveCpp

### 3. cannot load shared libraries
When executing main program, some shared libraries may not be able to load:
```
simple-radio-telescope-backend: error while loading shared libraries: libsycl.so.6: cannot open shared object file: No such file or directory
```
One cause is that Intel LLVM is not installed to a location where `ldconfig` recognizes.

To fix this, for self-compiled Intel LLVM,
```bash
export LD_LIBRARY_PATH=/opt/intel-llvm/lib:$LD_LIBRARY_PATH
```
where `/opt/intel-llvm` should be substituted by actural Intel LLVM installation location;  
for Intel oneAPI DPC++ installation,
```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. QML compile error on old OS, e.g. Ubuntu 20.04
Patch is needed to support old Qt & C++ standard library:
```diff
diff --git a/userspace/include/srtb/gui/spectrum_image_provider.hpp b/userspace/include/srtb/gui/spectrum_image_provider.hpp
index c54ae76..5836f00 100644
--- a/userspace/include/srtb/gui/spectrum_image_provider.hpp
+++ b/userspace/include/srtb/gui/spectrum_image_provider.hpp
@@ -410,8 +410,8 @@ class SimpleSpectrumImageProvider : public QObject, public QQuickImageProvider {
   void trigger_update(size_t data_stream_id) {
     if (parent) {  // object should be main window
       QMetaObject::invokeMethod(parent, "update_spectrum",
-                                Q_ARG(int, data_stream_id),
-                                Q_ARG(int, spectrum_update_counter));
+                                Q_ARG(QVariant, (int) data_stream_id),
+                                Q_ARG(QVariant, (int) spectrum_update_counter));
       spectrum_update_counter++;
       SRTB_LOGD << " [SimpleSpectrumImageProvider] "
                 << "trigger update, spectrum_update_counter = "
diff --git a/userspace/include/srtb/io/udp/packet_parser.hpp b/userspace/include/srtb/io/udp/packet_parser.hpp
index ebdb29c..3ec6a60 100644
--- a/userspace/include/srtb/io/udp/packet_parser.hpp
+++ b/userspace/include/srtb/io/udp/packet_parser.hpp
@@ -130,7 +130,7 @@ struct gznupsr_a1_packet_parser {
         (static_cast<counter_type>(word[6])) |
         (static_cast<counter_type>(word[7]) << (CHAR_BIT * vdif_word_size));
 
-    const vdif_header vh = std::bit_cast<vdif_header>(word);
+    const vdif_header vh = sycl::bit_cast<vdif_header>(word);
 
     // TODO: timestamp
     return std::make_tuple(/* header_size = */ packet_header_size,
diff --git a/userspace/src/main.qml b/userspace/src/main.qml
index 57d4441..9777da8 100644
--- a/userspace/src/main.qml
+++ b/userspace/src/main.qml
@@ -9,7 +9,7 @@ Window {
     property var spectrum_window: new Map()
     readonly property Component spectrum_window_component: Qt.createComponent("spectrum.qml")
 
-    function update_spectrum(window_id: int, counter: int) {
+    function update_spectrum(window_id, counter) {
         if (!spectrum_window.hasOwnProperty(window_id)) {
             spectrum_window[window_id] = 
                 spectrum_window_component.createObject(/* parent =  */ this, {
```
