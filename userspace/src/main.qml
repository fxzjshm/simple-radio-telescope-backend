import QtQuick 2.2
import QtQuick.Window 2.2

Window {
    id: window
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")

    Image {
        id: image
        cache: false
        source: "image://spectrum-image-provider/spectrum"
    }
}
