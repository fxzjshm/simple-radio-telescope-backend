add_executable(simple-radio-telescope-backend main.cpp qml.qrc)
srtb_add_dependency(simple-radio-telescope-backend)
target_link_libraries(simple-radio-telescope-backend PRIVATE Qt5::Gui Qt5::Quick)
