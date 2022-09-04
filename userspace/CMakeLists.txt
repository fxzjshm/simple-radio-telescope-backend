cmake_minimum_required(VERSION 3.12)
project(simple-radio-telescope-backend_userspace)

# select SYCL implemention
if(NOT DEFINED SYCL_IMPLEMENTION)
    message(WARNING "SYCL_IMPLEMENTION not defined, default to hipSYCL")
    set(SYCL_IMPLEMENTION "hipSYCL")
endif()

# detect backends
## CUDA, SRTB_CUDA_ARCH required if with hipSYCL and optional with intal-llvm
find_package(CUDAToolkit QUIET)
option(SRTB_ENABLE_CUDA "option to enable CUDA backend" ${CUDAToolkit_FOUND})
message("SRTB_ENABLE_CUDA=${SRTB_ENABLE_CUDA}")
## ROCm/HIP, SRTB_ROCM_ARCH required
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip QUIET)
find_package(rocfft QUIET)
find_package(hipfft QUIET)
if(hip_FOUND AND rocfft_FOUND AND hipfft_FOUND)
    set(SRTB_ROCM_BACKEND_REQUIREMENT_MET TRUE)
else()
    set(SRTB_ROCM_BACKEND_REQUIREMENT_MET FALSE)
endif()
option(SRTB_ENABLE_ROCM "option to enable ROCm backend" ${SRTB_ROCM_BACKEND_REQUIREMENT_MET})
message("SRTB_ENABLE_ROCM=${SRTB_ENABLE_ROCM}")
if(SRTB_ENABLE_ROCM AND NOT DEFINED SRTB_ROCM_ARCH)
    message(FATAL_ERROR "ROCm backend enabled, but compiling for ROCm/HIP backend requires SRTB_ROCM_ARCH, e.g. gfx906")
endif()

# export(?) backend selection to macros
if(SRTB_ENABLE_CUDA)
    add_compile_definitions(SRTB_ENABLE_CUDA=1)
endif()
if(SRTB_ENABLE_ROCM)
    add_compile_definitions(SRTB_ENABLE_ROCM=1)
endif()

# setup SYCL compilers
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    set(SRTB_HIPSYCL_TARGETS "")
    if(SRTB_ENABLE_CUDA)
        if(NOT DEFINED SRTB_CUDA_ARCH)
            message(FATAL_ERROR "compiling for CUDA backend with hipSYCL requires SRTB_CUDA_ARCH, e.g. sm_86")
        endif()
        set(SRTB_HIPSYCL_TARGETS "${SRTB_HIPSYCL_TARGETS}cuda:${SRTB_CUDA_ARCH};")
    endif()
    if(SRTB_ENABLE_ROCM)
        set(SRTB_HIPSYCL_TARGETS "${SRTB_HIPSYCL_TARGETS}hip:${SRTB_ROCM_ARCH};")
    endif()
    set(SRTB_HIPSYCL_TARGETS "${SRTB_HIPSYCL_TARGETS}omp")
    message("HIPSYCL_TARGETS set to ${SRTB_HIPSYCL_TARGETS}")  # e.g. HIPSYCL_TARGETS=cuda:sm_86;hip:gfx906;omp
    set(HIPSYCL_TARGETS ${SRTB_HIPSYCL_TARGETS})
    find_package(hipSYCL CONFIG REQUIRED)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")
    # e.g. 
    # CUDA: clang -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend -cuda-gpu-arch=sm_86
    # ROCm: clang -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906
    message("Make sure this compiler supports SYCL")
    if(SRTB_ENABLE_CUDA AND SRTB_ENABLE_ROCM)
        message(FATAL_ERROR "don't know how to compile for both CUDA and ROCm on intel/llvm (TODO)")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
    if(SRTB_ENABLE_CUDA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda ")
        if(DEFINED SRTB_CUDA_ARCH)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SRTB_CUDA_ARCH}")
        endif()
    endif()
    if(SRTB_ENABLE_ROCM)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${SRTB_ROCM_ARCH} ")
    endif()
    # if sycl-targets not set, use default
    message("CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
else()
    message(FATAL_ERROR "Sorry, unknown/untested/un-adapted SYCL implementation " ${SYCL_IMPLEMENTION})
endif()

# set cmake config for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# set C++ language version requirement
set(CMAKE_CXX_STANDARD 20)

# add extra warnings
## known SYCL implementations are gcc compatible
## -Wold-style-cast removed due to usage in library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -pedantic-errors -Wextra -Wall -Winit-self -Woverloaded-virtual -Wuninitialized -Wmissing-declarations -Winit-self")
## required for backtrace on crash
## why CMAKE_BUILD_TYPE=Debug produces binary without debug info?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer ")
## remove some warnings that do not directly related to this code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-builtins ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic ")

# dependency
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
if(SRTB_ENABLE_ROCM)
    link_directories("/opt/rocm/lib")
endif()
find_package(Qt5 COMPONENTS Quick Gui REQUIRED)

function(srtb_add_dependency target_name)
    # add SYCL to targets
    if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
        add_sycl_to_target(TARGET ${target_name})
    elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")
        # maybe nothing to do
    else()

    endif()

    # add dependency
    target_link_libraries(${target_name} PUBLIC fftw3_threads fftw3 fftw3f_threads fftw3f Threads::Threads) # TODO maybe findFFTW
    target_link_libraries(${target_name} PUBLIC stdc++ m)  # TODO: this shouldn't be needed
    target_link_libraries(${target_name} PUBLIC dl)
    if(SRTB_ENABLE_CUDA)
        # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
        target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cufft)
    endif()
    if(SRTB_ENABLE_ROCM)
        # https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
        target_link_libraries(${target_name} PUBLIC hip::hipfft roc::rocfft)
        target_link_libraries(${target_name} PUBLIC amdhip64)
        # TODO: hip::host gives CLANGRT_BUILTINS not found error,
        #       possibly related to https://github.com/RadeonOpenCompute/ROCm/issues/1717
        #                           https://github.com/ROCm-Developer-Tools/HIP-autopr/blob/develop/hip-lang-config.cmake.in#L98
    endif()
endfunction()

include_directories("include")

add_subdirectory("src")
add_subdirectory("tests")
