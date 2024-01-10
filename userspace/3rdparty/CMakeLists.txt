# include paths set on parent level CMakeLists.txt

# avoid cnpy's install commands
find_package(ZLIB REQUIRED)

add_library(cnpy SHARED "cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

add_library(cnpy-static STATIC "cnpy/cnpy.cpp")
set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
target_link_libraries(cnpy-static ${ZLIB_LIBRARIES})

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
set_source_files_properties( "cnpy/cnpy.cpp" PROPERTIES COMPILE_FLAGS " -w " )

add_subdirectory("exprgrammar")
