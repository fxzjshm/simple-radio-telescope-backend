import QtQuick 2.2
import QtQuick.Window 2.2

Window {
    id: main_window
    visible: true
    title: qsTr("simple radio telescope backend (main window)")

    property var spectrum_window: new Map()
    readonly property Component spectrum_window_component: Qt.createComponent("spectrum.qml")

    function update_spectrum(window_id: int, counter: int) {
        if (!spectrum_window.hasOwnProperty(window_id)) {
            spectrum_window[window_id] = 
                spectrum_window_component.createObject(/* parent =  */ this, {
                    spectrum_window_id: window_id
                });
        }

        spectrum_window[window_id].spectrum_source =
            "image://spectrum-image-provider/" + window_id + "/" + counter;
        return window_id;
    }
}
