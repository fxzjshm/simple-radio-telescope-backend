cmake_minimum_required(VERSION 3.12)
project(simple-radio-telescope-backend_userspace)

# import
include(CheckCXXCompilerFlag)

# select SYCL implemention
if(NOT DEFINED SRTB_SYCL_IMPLEMENTATION)
    message(WARNING "SRTB_SYCL_IMPLEMENTATION not defined, default to AdaptiveCpp")
    set(SRTB_SYCL_IMPLEMENTATION "AdaptiveCpp")
endif()


# detect backends
## CUDA, SRTB_CUDA_ARCH required if with AdaptiveCpp and optional with intal-llvm
find_package(CUDAToolkit QUIET)
option(SRTB_ENABLE_CUDA "option to enable CUDA backend" ${CUDAToolkit_FOUND})
message(STATUS "SRTB_ENABLE_CUDA = ${SRTB_ENABLE_CUDA}")

## ROCm/HIP, SRTB_ROCM_ARCH required
if(DEFINED ROCM_PATH)
    set(SRTB_ROCM_PATH ${ROCM_PATH})
else()
    set(SRTB_ROCM_PATH "/opt/rocm")
endif()
list(APPEND CMAKE_PREFIX_PATH ${SRTB_ROCM_PATH}/hip ${SRTB_ROCM_PATH})
find_package(hip QUIET)
find_package(rocfft QUIET)
find_package(hipfft QUIET)
if(hip_FOUND AND rocfft_FOUND AND hipfft_FOUND)
    set(SRTB_ROCM_BACKEND_REQUIREMENT_MET TRUE)
else()
    set(SRTB_ROCM_BACKEND_REQUIREMENT_MET FALSE)
endif()
option(SRTB_ENABLE_ROCM "option to enable ROCm backend" ${SRTB_ROCM_BACKEND_REQUIREMENT_MET})
message(STATUS "SRTB_ENABLE_ROCM = ${SRTB_ENABLE_ROCM}")
if(SRTB_ENABLE_ROCM AND NOT DEFINED SRTB_ROCM_ARCH)
    message(FATAL_ERROR "ROCm backend enabled, but compiling for ROCm/HIP backend requires SRTB_ROCM_ARCH, e.g. gfx906")
endif()
if(SRTB_ENABLE_ROCM)
    link_directories("${ROCM_PATH}/lib")
endif()

# MUSA
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSAToolkit QUIET)
option(SRTB_ENABLE_MUSA "option to enable MUSA backend" ${MUSA_FOUND})
message(STATUS "SRTB_ENABLE_MUSA = ${SRTB_ENABLE_MUSA}")

# OpenCL
find_package(OpenCL QUIET)
## workaround for compile error with target hip:gfx1035;generic
if(OpenCL_FOUND AND ((SRTB_SYCL_IMPLEMENTATION STREQUAL "intel-llvm") OR SRTB_ENABLE_GENERIC OR NOT (SRTB_ENABLE_ROCM OR SRTB_ENABLE_CUDA)))
  set(SRTB_DEFAULT_ENABLE_OPENCL ON)
else()
  set(SRTB_DEFAULT_ENABLE_OPENCL OFF)
endif()
option(SRTB_ENABLE_OPENCL "option to enable OpenCL backend" ${SRTB_DEFAULT_ENABLE_OPENCL})
message(STATUS "SRTB_ENABLE_OPENCL = ${SRTB_ENABLE_OPENCL}")

# export(?) backend selection to macros
if(SRTB_ENABLE_CUDA)
    add_compile_definitions(SRTB_ENABLE_CUDA=1)
endif()
if(SRTB_ENABLE_ROCM)
    add_compile_definitions(SRTB_ENABLE_ROCM=1)
endif()
if(SRTB_ENABLE_MUSA)
    add_compile_definitions(SRTB_ENABLE_MUSA=1)
endif()
if(SRTB_ENABLE_OPENCL)
    add_compile_definitions(SRTB_ENABLE_OPENCL=1)
endif()

if(SRTB_SYCL_IMPLEMENTATION STREQUAL "hipSYCL")
  set(SRTB_SYCL_IMPLEMENTATION "AdaptiveCpp")
endif()

# setup SYCL compilers
if(SRTB_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
  # e.g. ACPP_TARGETS=cuda:sm_86;hip:gfx906;omp or ACPP_TARGETS=generic;omp
  # using CMake list as it happens to be semi-column separated
  if(NOT DEFINED ACPP_TARGETS)
    set(SRTB_ACPP_TARGETS "")
    if(SRTB_ENABLE_CUDA AND NOT SRTB_ENABLE_GENERIC)
        if(NOT DEFINED SRTB_CUDA_ARCH)
            message(FATAL_ERROR "compiling for CUDA backend with AdaptiveCpp requires SRTB_CUDA_ARCH, e.g. sm_86")
        endif()
        set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} cuda:${SRTB_CUDA_ARCH})
    endif()
    if(SRTB_ENABLE_ROCM AND NOT SRTB_ENABLE_GENERIC)
        set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} hip:${SRTB_ROCM_ARCH})
    endif()
    if(SRTB_ENABLE_MUSA)
        set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} generic)
    endif()
    if(SRTB_ENABLE_OPENCL)
        set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} generic)
    endif()
    if(SRTB_ENABLE_GENERIC)
        set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} generic)
    endif()
    set(SRTB_ACPP_TARGETS ${SRTB_ACPP_TARGETS} omp)
    list(REMOVE_DUPLICATES SRTB_ACPP_TARGETS)
    set(ACPP_TARGETS ${SRTB_ACPP_TARGETS})
  endif()
  message(STATUS "ACPP_TARGETS = ${ACPP_TARGETS}")
  find_package(AdaptiveCpp CONFIG REQUIRED)
elseif(SRTB_SYCL_IMPLEMENTATION STREQUAL "intel-llvm")
    # e.g. 
    # CUDA: clang -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend -cuda-gpu-arch=sm_86
    # ROCm: clang -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906
    # mixed: icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda,spir64 \
    #             -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1010 \
    #             -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86 \
    #             -o sycl-app sycl-app.cpp
    # ref: https://simplecore.intel.com/oneapi-io/wp-content/uploads/sites/98/Rafal-B-Target-NVIDIA-and-AMD-with-oneAPI-and-SYCL.pdf
    check_cxx_compiler_flag("-fsycl" COMPILER_OPT_F_SYCL_SUPPORTED)
    if(NOT COMPILER_OPT_F_SYCL_SUPPORTED)
        message(WARNING "flag '-fsycl' not working, make sure this compiler supports SYCL")
    endif()
    set(SRTB_SYCL_FLAGS " -fsycl ")
    ## do not assume id range fit in MAX_INT because baseband_input_count may be very large
    ## when dealing with very high dispersion measurement
    set(SRTB_SYCL_FLAGS "${SRTB_SYCL_FLAGS} -fno-sycl-id-queries-fit-in-int ")
    set(SRTB_SYCL_TARGETS "")
    set(SRTB_SYCL_ARCH_FLAGS "")
    if(SRTB_ENABLE_CUDA)
        set(SRTB_SYCL_TARGETS ${SRTB_SYCL_TARGETS} nvptx64-nvidia-cuda)
        if(DEFINED SRTB_CUDA_ARCH)
            set(SRTB_SYCL_ARCH_FLAGS "${SRTB_SYCL_ARCH_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${SRTB_CUDA_ARCH} ")
        endif()
    endif()
    if(SRTB_ENABLE_ROCM)
        set(SRTB_SYCL_TARGETS ${SRTB_SYCL_TARGETS} amdgcn-amd-amdhsa)
        set(SRTB_SYCL_ARCH_FLAGS "${SRTB_SYCL_ARCH_FLAGS} -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${SRTB_ROCM_ARCH} ")
    endif()
    if(SRTB_ENABLE_OPENCL)
        set(SRTB_SYCL_TARGETS ${SRTB_SYCL_TARGETS} spir64-unknown-unknown)
    endif()
    # convert to comma separated
    string(REPLACE ";" "," SRTB_SYCL_TARGETS "${SRTB_SYCL_TARGETS}")
    set(SRTB_SYCL_FLAGS "${SRTB_SYCL_FLAGS} -fsycl-targets=${SRTB_SYCL_TARGETS} ${SRTB_SYCL_ARCH_FLAGS}")
    # if sycl-targets not set, use default
    message(STATUS "SRTB_SYCL_FLAGS = ${SRTB_SYCL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SRTB_SYCL_FLAGS}")
else()
    message(FATAL_ERROR "Sorry, unknown/untested/un-adapted SYCL implementation " ${SRTB_SYCL_IMPLEMENTATION})
endif()


set(CMAKE_INCLUDE_CURRENT_DIR ON)

# set C++ language version requirement
set(CMAKE_CXX_STANDARD 20)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# https://cmake.org/cmake/help/v3.9/module/CheckIPOSupported.html
# Optional IPO. Do not use IPO if it's not supported by compiler.
include(CheckIPOSupported)
check_ipo_supported(RESULT SRTB_IPO_SUPPORTED OUTPUT SRTB_IPO_CHECK_OUTPUT)
if(SRTB_IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(WARNING "Inter-Procedural Optimization is not supported, output: ${SRTB_IPO_CHECK_OUTPUT}")
endif()

# Position independent code
## TODO: does this affect performance?
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add extra flags
## add extra warnings
## known SYCL implementations are gcc compatible
## -Wold-style-cast removed due to usage in library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wextra -Wall -Winit-self -Woverloaded-virtual -Wuninitialized -Wmissing-declarations -Winit-self -Wold-style-cast -Wshadow -Wswitch-default ")
## required for backtrace on crash
## why CMAKE_BUILD_TYPE=Debug produces binary without debug info?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic ")
## -ffp-contract=fast will not work for df64
## -ffp-contract=fast-honor-pragmas not supported by CUDA
## https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffp-contract
if(SRTB_ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=on ")
endif()
## optimize flags
if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize ")
  # check whether -march=native is supported
  # https://github.com/weidai11/cryptopp/issues/273#issuecomment-249271818
  check_cxx_compiler_flag("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ")
  else()
    message(WARNING "-march=native seems not suppored, please manually set optimization flag here")
    # e.g.
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdc ")
  endif()
endif()
## sanitizer, for debug usage
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-optimize-sibling-calls ")
## trace compile time
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace ")
## vectorize report
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize")


# dependency
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/3rdparty/cmake")

# pkg-config
find_package(PkgConfig QUIET)
if(NOT PKG_CONFIG_FOUND)
  message(WARNING "pkg-config not found, some dependencies based on pkg-config will not be searched")
endif()

## Boost
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS program_options iostreams REQUIRED)
find_package(Boost COMPONENTS stacktrace_backtrace)
if(Boost_STACKTRACE_BACKTRACE_FOUND)
  message(STATUS "Found boost_stacktrace_backtrace")
else()
  message(WARNING "boost_stacktrace_backtrace not found, code location may absent in stacktrace")
endif()
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
#add_compile_definitions(BOOST_ASIO_HAS_IO_URING)
#add_compile_definitions(BOOST_ASIO_DISABLE_EPOLL)

## Python, for matplotlib-cpp
## ref: https://github.com/lava/matplotlib-cpp/blob/master/CMakeLists.txt
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

## hwloc
if(PKG_CONFIG_FOUND)
  find_package(hwloc)
endif()
if(hwloc_FOUND)
  add_compile_definitions(SRTB_HAS_HWLOC=1)
else()
  message(WARNING "hwloc not found, performance of UDP receivers may be degraded")
endif()

## FFTW3
if(PKG_CONFIG_FOUND)
  find_package(FFTW COMPONENTS FLOAT_LIB DOUBLE_LIB FLOAT_THREADS_LIB DOUBLE_THREADS_LIB)
endif()
if(FFTW_FOUND)
  add_compile_definitions(SRTB_HAS_FFTW=1)
else()
  message(WARNING "FFTW not found, performance of CPU FFT transformations may be degraded")
endif()

## 3rdparty
add_subdirectory("3rdparty")
include_directories(SYSTEM "3rdparty/")
include_directories(SYSTEM "3rdparty/matplotlib-cpp/")
include_directories(SYSTEM "3rdparty/SyclParallelSTL/include/")
include_directories(SYSTEM "3rdparty/dsmath/")
include_directories(SYSTEM "3rdparty/concurrentqueue/")
include_directories(SYSTEM "3rdparty/SyclCPLX/include/")
include_directories(SYSTEM "3rdparty/cnpy/")

## Qt (optional)
### ref: https://doc.qt.io/qt-6/cmake-qt5-and-qt6-compatibility.html
find_package(QT NAMES Qt6 Qt5 COMPONENTS Core)
message(STATUS "QT_VERSION_MAJOR = ${QT_VERSION_MAJOR}")
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Gui)
option(SRTB_ENABLE_GUI "option to enable GUI" ${Qt${QT_VERSION_MAJOR}_FOUND})
message(STATUS "SRTB_ENABLE_GUI = ${SRTB_ENABLE_GUI}")
if(SRTB_ENABLE_GUI)
    add_compile_definitions(SRTB_ENABLE_GUI=true)
    # Qt's keywords like signals and slots are annoying
    add_compile_definitions(QT_NO_KEYWORDS)
    # set cmake config for Qt
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
else()
    add_compile_definitions(SRTB_ENABLE_GUI=false)
endif()


function(srtb_add_dependency target_name)
    # add SYCL to targets
    if(SRTB_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
        add_sycl_to_target(TARGET ${target_name})
    elseif(SRTB_SYCL_IMPLEMENTATION STREQUAL "intel-llvm")
        # maybe nothing to do
    else()

    endif()

    # add dependency
    if(FFTW_FOUND)
      target_link_libraries(${target_name} PUBLIC FFTW::DoubleThreads FFTW::Double FFTW::FloatThreads FFTW::Float)
    endif()
    target_link_libraries(${target_name} PUBLIC Threads::Threads)
    target_link_libraries(${target_name} PUBLIC Boost::program_options Boost::iostreams)
    if(Boost_STACKTRACE_BACKTRACE_FOUND)
      # https://stackoverflow.com/questions/52583544/boost-stack-trace-not-showing-function-names-and-line-numbers/71027489#71027489
      target_compile_definitions(${target_name} PRIVATE BOOST_STACKTRACE_LINK)
      target_link_libraries(${target_name} PUBLIC boost_stacktrace_backtrace)
    endif()
    if(hwloc_FOUND)
      target_link_libraries(${target_name} PUBLIC ${hwloc_LIBRARIES})
    endif()
    target_link_libraries(${target_name} PUBLIC stdc++ m)  # TODO: this shouldn't be needed
    target_link_libraries(${target_name} PUBLIC ${CMAKE_DL_LIBS})
    #target_link_libraries(${target_name} PUBLIC uring)
    if(SRTB_ENABLE_CUDA)
        # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
        target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cufft)
    endif()
    if(SRTB_ENABLE_ROCM)
        # https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
        target_link_libraries(${target_name} PUBLIC hip::hipfft roc::rocfft)
        target_link_libraries(${target_name} PUBLIC amdhip64)
    endif()
    if(SRTB_ENABLE_MUSA)
        target_link_libraries(${target_name} PUBLIC MUSA::musart MUSA::musa_driver MUSA::mufft)
    endif()
    if(SRTB_ENABLE_OPENCL)
        target_link_libraries(${target_name} PUBLIC OpenCL::OpenCL)
    endif()

    if(SRTB_ENABLE_GUI)
        target_link_libraries(${target_name} PRIVATE Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Quick)
    endif()
endfunction()

include_directories("include")

add_subdirectory("src")
add_subdirectory("tests")
