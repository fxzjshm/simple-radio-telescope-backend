include_directories("include")

add_library(exprgrammar SHARED "src/exprgrammar.cpp")
target_include_directories(exprgrammar SYSTEM PUBLIC "include")

add_library(exprgrammar-static STATIC "src/exprgrammar.cpp")
set_target_properties(exprgrammar-static PROPERTIES OUTPUT_NAME "exprgrammar")
target_include_directories(exprgrammar-static SYSTEM PUBLIC "include")
