import os
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSignal
from source.utils import enable_widgets
from source.config import Config
import traceback

pg.mkQApp()
pg.setConfigOption('background', 'w')

ui_path = os.path.join(os.path.dirname(__file__), Config.UI_PATH, 'height_window.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(ui_path)

class GraphWindow(QMainWindow):
    data_generated = pyqtSignal(pd.DataFrame)

    def __init__(self, before_dataframe, after_dataframe, present_date, predict_date, parent=None):
        super().__init__(parent)
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.setWindowTitle("Fill Height Settings")

        # Set the fixed size to prevent window resizing
        self.setFixedSize(self.size())

        self.ui.apply_pushbutton.clicked.connect(self.apply)
        self.ui.reset_pushbutton.clicked.connect(self.reset)
        self.ui.cancel_pushbutton.clicked.connect(self.cancel)
        self.ui.confirm_pushbutton.clicked.connect(self.confirm)

        self.present_date = present_date
        self.predict_date = predict_date
        self.before_df = before_dataframe.copy()
        self.origin_after_df = after_dataframe.copy()
        self.after_df = after_dataframe.copy()
        self.parent = parent

        self.height_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
        self.height_text = pg.TextItem(color='r', anchor=(1, 0.5))  # Anchor set to middle right

        self.draw_graph()

        self.ui.graph_widget.addItem(self.height_scatter)
        self.ui.graph_widget.addItem(self.height_text)

        self.height_text.setVisible(False)

        self.proxy = pg.SignalProxy(self.ui.graph_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def closeEvent(self, event):
        if self.parent:
            enable_widgets(self.parent.ui)
        super().closeEvent(event)

    def draw_graph(self):
        self.ui.graph_widget.clear()
        self.ui.graph_widget.plot(self.before_df.index, self.before_df['fill_height'].values, pen='k', name="Before Data")
        self.ui.graph_widget.plot(self.after_df.index, self.after_df['fill_height'].values, pen='r', name="After Data")

    def apply(self):
        try:
            date = int(self.ui.date_lineedit.text())
            if date <= self.present_date or date >= self.predict_date:
                raise ValueError("현재 시점과 예측 시점 사이의 시점을 입력해주세요.")
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        try:
            value = float(self.ui.value_lineedit.text())
            if value <= 0 or (date - 1) in self.after_df.index and value < self.after_df.loc[date - 1, 'fill_height']:
                raise ValueError("이전 성토 높이보다 낮은 높이로 설정할 수 없습니다.")
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.after_df.loc[date:, 'fill_height'] = value
        self.draw_graph()

    def reset(self):
        last_height = self.before_df['fill_height'].iloc[-1]
        self.after_df = pd.DataFrame(
            {'fill_height': [last_height] * (self.predict_date - self.present_date)},
            index=pd.RangeIndex(start=self.present_date, stop=self.predict_date)
        )
        self.draw_graph()

    def cancel(self):
        self.data_generated.emit(self.origin_after_df)
        self.close()

    def confirm(self):
        self.data_generated.emit(self.after_df)
        self.close()

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.ui.graph_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.ui.graph_widget.plotItem.vb.mapSceneToView(pos)
            x_value = int(mousePoint.x())

            valid_index = False
            height_text_content = ""

            try:
                if x_value in self.before_df.index:
                    height_value_before = self.before_df.loc[x_value, 'fill_height']
                    self.height_scatter.setData([x_value], [height_value_before])
                    height_text_content = f"X: {x_value}, Before: {height_value_before:.2f}"
                    self.height_text.setText(height_text_content)
                    self.height_text.setPos(x_value - 0.5, height_value_before)  # Slightly offset to the left
                    self.height_scatter.setVisible(True)
                    self.height_text.setVisible(True)
                    valid_index = True
                else:
                    self.height_scatter.setVisible(False)
                    self.height_text.setVisible(False)

                if x_value in self.after_df.index:
                    height_value_after = self.after_df.loc[x_value, 'fill_height']
                    self.height_scatter.setData([x_value], [height_value_after])
                    height_text_content += f" X: {x_value}, After: {height_value_after:.2f}"
                    self.height_text.setText(height_text_content)
                    self.height_text.setPos(x_value - 0.5, height_value_after)  # Slightly offset to the left
                    self.height_scatter.setVisible(True)
                    self.height_text.setVisible(True)
                    valid_index = True

            except Exception as e:
                print(f"Error in mouseMoved: {type(e).__name__} - {e}")

            if not valid_index:
                self.height_text.setVisible(False)
                self.height_scatter.setVisible(False)

            return

        self.height_text.setVisible(False)
        self.height_scatter.setVisible(False)
