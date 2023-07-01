# include paths set on parent level CMakeLists.txt

# avoid cnpy's install commands
find_package(ZLIB REQUIRED)

add_library(cnpy SHARED "cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

add_library(cnpy-static STATIC "cnpy/cnpy.cpp")
set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
target_link_libraries(cnpy-static ${ZLIB_LIBRARIES})
