add_executable(21cma-make_beam 
  make_beam.cpp
  program_options.cpp
)
set_property(TARGET 21cma-make_beam PROPERTY CXX_STANDARD 23)
srtb_add_dependency(21cma-make_beam)
target_link_libraries(21cma-make_beam PUBLIC exprgrammar)
target_link_libraries(21cma-make_beam PUBLIC starlink_pal erfa)
