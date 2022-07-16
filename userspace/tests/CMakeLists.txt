# Tests

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (file ${files})
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${name}.cpp)
    # add SYCL to targets
    if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
        add_sycl_to_target(TARGET ${name})
    endif()
    add_test(NAME ${name} COMMAND ${name})
endforeach()

# test naive fft
srtb_add_dependency(test-naive_fft)
foreach(bit RANGE 5 25 7)
add_test(NAME test-naive_fft-${bit} COMMAND test-naive_fft ${bit})
endforeach()

# test FFT wrappers
srtb_add_dependency(test-fft_wrappers)
