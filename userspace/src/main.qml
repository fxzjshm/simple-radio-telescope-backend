import QtQuick 2.2
import QtQuick.Window 2.2

Window {
    id: main_window
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    Image {
        id: spectrum_image
        cache: false
        source: "image://spectrum-image-provider/spectrum"
    }

    function update_spectrum(counter) {
        console.log("update_spectrum");
        spectrum_image.source = "image://spectrum-image-provider/" + counter;
    }
}
