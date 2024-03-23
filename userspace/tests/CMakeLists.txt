# Tests

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (file ${files})
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${name}.cpp)
    # add dependency to targets
    srtb_add_dependency(${name})
    #target_precompile_headers(${name} REUSE_FROM simple-radio-telescope-backend)
    add_test(NAME ${name} COMMAND ${name})
endforeach()

# test naive fft
foreach(bit RANGE 5 25 7)
add_test(NAME test-naive_fft-${bit} COMMAND test-naive_fft ${bit})
endforeach()

# test fft wrappers
## batch_size must be power of 2?
foreach(batch_size RANGE 8 16 8)
foreach(bit RANGE 2 4 3)
add_test(NAME test-fft_wrappers-${bit}-${batch_size} COMMAND test-fft_wrappers ${bit} ${batch_size})
endforeach()
endforeach()

foreach(batch_size RANGE 2 6 7)
foreach(bit RANGE 21 22 3)
add_test(NAME test-fft_wrappers-${bit}-${batch_size} COMMAND test-fft_wrappers ${bit} ${batch_size})
endforeach()
endforeach()

add_test(NAME test-multi_reduce-1307-21996 COMMAND test-multi_reduce 1307 21996)
add_test(NAME test-multi_reduce-7-1723 COMMAND test-multi_reduce 7 1723)
add_test(NAME test-multi_reduce-2907-9 COMMAND test-multi_reduce 2907 9)

add_subdirectory("misc")
