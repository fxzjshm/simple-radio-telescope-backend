set(SRTB_MAIN_SOURCE_FILES main.cpp)
if(SRTB_ENABLE_GUI)
  # .hpp files added here because of MOC. Seems that every file that uses Q_Object things have to be MOC-ed
  set(SRTB_MAIN_SOURCE_FILES 
    ${SRTB_MAIN_SOURCE_FILES}
    qml.qrc 
    ../include/srtb/gui/spectrum_image_provider.hpp
  )
endif()
add_executable(simple-radio-telescope-backend ${SRTB_MAIN_SOURCE_FILES})
srtb_add_dependency(simple-radio-telescope-backend)
#target_precompile_headers(simple-radio-telescope-backend
#  PRIVATE
#    [["srtb/sycl.hpp"]]
#    <boost/stacktrace.hpp>
#    <boost/lockfree/spsc_queue.hpp>
#    <boost/asio.hpp>
#)
target_link_libraries(simple-radio-telescope-backend PUBLIC cnpy)
target_link_libraries(simple-radio-telescope-backend PUBLIC exprgrammar)

if(SRTB_ENABLE_GUI)
  add_executable(test-gui
    test-gui.cpp
    qml.qrc
    ../include/srtb/gui/spectrum_image_provider.hpp
  )
  srtb_add_dependency(test-gui)
endif()

add_executable(srtb-correlator 
  correlator.cpp
)
srtb_add_dependency(srtb-correlator)

add_executable(srtb_baseband_receiver baseband_receiver.cpp)
srtb_add_dependency(srtb_baseband_receiver)
