# .hpp files added here because of MOC. Seems that every file that uses Q_Object things have to be MOC-ed
add_executable(simple-radio-telescope-backend main.cpp qml.qrc ../include/srtb/gui/spectrum_image_provider.hpp)
srtb_add_dependency(simple-radio-telescope-backend)
target_link_libraries(simple-radio-telescope-backend PRIVATE Qt5::Gui Qt5::Quick)

add_executable(test-gui test-gui.cpp qml.qrc ../include/srtb/gui/spectrum_image_provider.hpp)
srtb_add_dependency(test-gui)
target_link_libraries(test-gui PRIVATE Qt5::Gui Qt5::Quick)
