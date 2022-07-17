# Tests

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (file ${files})
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${name}.cpp)
    # add dependency to targets
    srtb_add_dependency(${name})
    add_test(NAME ${name} COMMAND ${name})
endforeach()

# test naive fft
foreach(bit RANGE 5 25 7)
add_test(NAME test-naive_fft-${bit} COMMAND test-naive_fft ${bit})
endforeach()
