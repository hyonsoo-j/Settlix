def set_widget_state(ui, state):
    ui.openfile_pushbutton.setEnabled(state)
    ui.datasetting_pushbutton.setEnabled(state)
    ui.height_pushbutton.setEnabled(state)
    ui.config_pushbutton.setEnabled(state)
    ui.prediction_pushbutton.setEnabled(state)
    ui.img_pushbutton.setEnabled(state)
    ui.csv_pushbutton.setEnabled(state)
    ui.help_pushbutton.setEnabled(state)

def enable_widgets(ui):
    set_widget_state(ui, True)

def disable_widgets(ui):
    set_widget_state(ui, False)