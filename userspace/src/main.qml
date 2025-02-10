import QtQuick 2.2
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.15
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

        // always visible
        main_window.visible = true;

        spectrum_window[window_id].spectrum_source =
            "image://spectrum-image-provider/" + window_id + "/" + counter;
        return window_id;
    }

    ColumnLayout {
        property var margin: Math.sqrt(parent.width * parent.height) * 0.02
        anchors.fill: parent
        anchors.leftMargin: margin
        anchors.rightMargin: margin
        anchors.topMargin: margin
        anchors.bottomMargin: margin

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }

        RowLayout {
            id: inputRow
            width: parent.width

            Text {
                text: "Window ID: "
            }

            TextField {
                id: window_id_input
                Layout.fillWidth: true
            }

            Button {
                id: window_toggle
                height: parent.height
                text: "Toggle"
                onClicked: toggle_window(window_id_input.text)

                function toggle_window(window_id_input: string) {
                    var window_id = parseInt(window_id_input);
                    if (spectrum_window.hasOwnProperty(window_id)) {
                        spectrum_window[window_id].visible = !spectrum_window[window_id].visible;
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
        
    }
}
