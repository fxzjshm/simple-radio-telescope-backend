# not really tests, but can test some specific functions

include_directories(SYSTEM /opt/udpdk/include)
# include_directories(SYSTEM /usr/include/dpdk)
link_directories(/opt/udpdk/lib)

# https://stackoverflow.com/questions/69912482/compiling-dpdk-application-using-cmake-with-link-static-libraries
pkg_check_modules(DPDK "libdpdk")

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (file ${files})
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${name}.cpp)
    # add dependency to targets
    srtb_add_dependency(${name})
    #target_precompile_headers(${name} REUSE_FROM simple-radio-telescope-backend)
endforeach()

add_definitions(${DPDK_CFLAGS})
set(MYDPDK_LIBRARIES -Wl,--whole-archive ${DPDK_LINK_LIBRARIES} -Wl,--no-whole-archive)
include_directories(${DPDK_INCLUDE_DIR})
link_libraries(${MYDPDK_LIBRARIES})
add_definitions(-DHAVE_DPDK)

target_link_libraries(test-udpdk PUBLIC udpdk ${MYDPDK_LIBRARIES})

