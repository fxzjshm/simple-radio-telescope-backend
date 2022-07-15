cmake_minimum_required(VERSION 3.11)
project(simple-radio-telescope-backend_userspace)

# select SYCL implemention
if(NOT DEFINED SYCL_IMPLEMENTION)
    message(WARNING "SYCL_IMPLEMENTION not defined, default to hipSYCL")
    set(SYCL_IMPLEMENTION "hipSYCL")
endif()

# setup SYCL compilers
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    if(NOT DEFINED HIPSYCL_TARGETS)
        message(WARNING "HIPSYCL_TARGETS not defined, default to OpenMP only")
        set(HIPSYCL_TARGETS "omp")
    endif()
    find_package(hipSYCL CONFIG REQUIRED)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")
    message("Make sure this compiler supports SYCL and set target by CMAKE_CXX_FLAGS")
else()
    message(FATAL_ERROR "Sorry, unknown/untested/un-adapted SYCL implementation " ${SYCL_IMPLEMENTION})
endif()

# set C++ language version requirement
set(CMAKE_CXX_STANDARD 20)

# dependency
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories("include")

add_executable(simple-radio-telescope-backend src/main.cpp)

# add SYCL to targets
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET simple-radio-telescope-backend)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")

else()

endif()

# TODO: SRTB_CUDA, etc
find_package(CUDAToolkit)

add_subdirectory("tests")
