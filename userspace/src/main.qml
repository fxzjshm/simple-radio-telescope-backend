import QtQuick 2.9
import QtQuick.Window 2.9

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
