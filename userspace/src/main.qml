import QtQuick 2.2
import QtQuick.Window 2.2

Window {
    id: main_window
    width: Screen.width * 0.8
    height: Screen.height * 0.8
    visible: true
    title: qsTr("Hello World")

    Image {
        id: spectrum_image
        cache: true
        width: parent.width
        height: parent.height
        source: "image://spectrum-image-provider/spectrum"
        smooth: false
    }

    function update_spectrum(counter) {
        spectrum_image.source = "image://spectrum-image-provider/" + counter;
    }
}
