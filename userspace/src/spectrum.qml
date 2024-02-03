import QtQuick 2.2
import QtQuick.Window 2.2

Window {
    id: spectrum_window
    width: Screen.width * 0.8
    height: Screen.height * 0.8
    visible: true
    property string spectrum_window_id : "[unset]"
    title: qsTr("simple radio telescope backend (id = " + spectrum_window_id + ")")
    property alias spectrum_source: spectrum_image.source

    Image {
        id: spectrum_image
        cache: false
        width: parent.width
        height: parent.height
        source: "image://spectrum-image-provider/" + spectrum_window_id
        smooth: false
    }
}
