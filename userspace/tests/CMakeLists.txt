# Tests

# test naive fft
add_executable(test-naive_fft test-naive_fft.cpp)
target_link_libraries(test-naive_fft PUBLIC fftw3) # TODO maybe findFFTW
foreach(bit RANGE 5 25 7)
add_test(NAME test-naive_fft-${bit} COMMAND test-naive_fft ${bit})
endforeach()

# test udp_receiver
add_executable(test-udp_receiver test-udp_receiver.cpp)
add_test(NAME test-udp_receiver COMMAND test-udp_receiver)

# add SYCL to targets
if(SYCL_IMPLEMENTION STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET test-naive_fft)
    add_sycl_to_target(TARGET test-udp_receiver)
elseif(SYCL_IMPLEMENTION STREQUAL "intel-llvm")

else()

endif()

