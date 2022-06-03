# Tests

# test fft
add_executable(test-fft test-fft.cpp)
target_link_libraries(test-fft PUBLIC fftw3) # TODO maybe findFFTW

# test udp_receiver
add_executable(test-udp_receiver test-udp_receiver.cpp)


# add SYCL to targets
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET test-fft)
    add_sycl_to_target(TARGET test-udp_receiver)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")

else()

endif()

foreach(bit RANGE 5 25 7)
add_test(NAME test-fft-${bit} COMMAND test-fft ${bit})
endforeach()
