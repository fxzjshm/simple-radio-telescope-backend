add_executable(test-fft test-fft.cpp)
target_link_libraries(test-fft PUBLIC fftw3) # TODO maybe findFFTW

# add SYCL to targets
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET test-fft)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")

else()

endif()

foreach(bit RANGE 5 25)
add_test(NAME test-fft-${bit}
         COMMAND test-fft ${bit})
endforeach()
