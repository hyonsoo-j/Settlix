import pyqtgraph as pg

def plot_graphs(ui, graph_data):
    ui.height_graph.clear()
    ui.settlement_graph.clear()

    # Plotting height data
    if 'before_height' in graph_data and not graph_data['before_height'].empty:
        ui.height_graph.plot(graph_data['before_height'].index, graph_data['before_height']['fill_height'].values, pen='k')  # Black for before height
    if 'after_height' in graph_data and not graph_data['after_height'].empty:
        ui.height_graph.plot(graph_data['after_height'].index, graph_data['after_height']['fill_height'].values, pen='r')  # Red for after height

    # Plotting settlement data
    if 'before_settlement' in graph_data and not graph_data['before_settlement'].empty:
        ui.settlement_graph.plot(graph_data['before_settlement'].index, graph_data['before_settlement']['settlement'].values, pen='k')  # Black for before settlement
    if 'after_settlement' in graph_data and not graph_data['after_settlement'].empty:
        ui.settlement_graph.plot(graph_data['after_settlement'].index, graph_data['after_settlement']['settlement'].values, pen='k')  # Black for after settlement
    if 'predicted_settlement' in graph_data and not graph_data['predicted_settlement'].empty:
        ui.settlement_graph.plot(graph_data['predicted_settlement'].index, graph_data['predicted_settlement']['settlement'].values, pen='b')  # Blue for predicted settlement

    # Ensure the entire range is visible
    # ui.settlement_graph.plotItem.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
    ui.height_graph.plotItem.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    height_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
    settlement_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('b'))
    height_text = pg.TextItem(color='r')
    settlement_text = pg.TextItem(color='b')

    ui.height_graph.addItem(height_scatter)
    ui.height_graph.addItem(height_text)
    ui.settlement_graph.addItem(settlement_scatter)
    ui.settlement_graph.addItem(settlement_text)

    height_text.setVisible(False)
    settlement_text.setVisible(False)

    def mouseMoved(evt):
        pos = evt[0]
        for plot_widget in [ui.height_graph, ui.settlement_graph]:
            if plot_widget.sceneBoundingRect().contains(pos):
                mousePoint = plot_widget.plotItem.vb.mapSceneToView(pos)
                x_value = int(mousePoint.x())

                valid_index = False

                height_text_content = ""
                settlement_text_content = ""

                try:
                    if 'before_height' in graph_data and not graph_data['before_height'].empty and x_value in graph_data['before_height'].index:
                        height_value_before = graph_data['before_height'].loc[x_value, 'fill_height']
                        height_scatter.setData([x_value], [height_value_before])
                        height_text_content = f"X: {x_value}, Before: {height_value_before:.2f}"
                        height_text.setText(height_text_content)
                        height_text.setPos(x_value, height_value_before)
                        height_scatter.setVisible(True)
                        height_text.setVisible(True)
                        valid_index = True
                    else:
                        height_scatter.setVisible(False)
                        height_text.setVisible(False)

                    if 'after_height' in graph_data and not graph_data['after_height'].empty and x_value in graph_data['after_height'].index:
                        height_value_after = graph_data['after_height'].loc[x_value, 'fill_height']
                        height_scatter.setData([x_value], [height_value_after])
                        height_text_content += f" X: {x_value}, After: {height_value_after:.2f}"
                        height_text.setText(height_text_content)
                        height_text.setPos(x_value, height_value_after)
                        height_scatter.setVisible(True)
                        height_text.setVisible(True)
                        valid_index = True

                    if 'before_settlement' in graph_data and not graph_data['before_settlement'].empty and x_value in graph_data['before_settlement'].index:
                        settlement_value_before = graph_data['before_settlement'].loc[x_value, 'settlement']
                        settlement_scatter.setData([x_value], [settlement_value_before])
                        settlement_text_content = f"X: {x_value}, Before: {settlement_value_before:.2f}"
                        settlement_text.setText(settlement_text_content)
                        settlement_text.setPos(x_value, settlement_value_before)
                        settlement_scatter.setVisible(True)
                        settlement_text.setVisible(True)
                        valid_index = True
                    else:
                        settlement_scatter.setVisible(False)
                        settlement_text.setVisible(False)

                    if 'after_settlement' in graph_data and not graph_data['after_settlement'].empty and x_value in graph_data['after_settlement'].index:
                        settlement_value_after = graph_data['after_settlement'].loc[x_value, 'settlement']
                        settlement_scatter.setData([x_value], [settlement_value_after])
                        settlement_text_content += f" X: {x_value}, After: {settlement_value_after:.2f}"
                        settlement_text.setText(settlement_text_content)
                        settlement_text.setPos(x_value, settlement_value_after)
                        settlement_scatter.setVisible(True)
                        settlement_text.setVisible(True)
                        valid_index = True

                    if 'predicted_settlement' in graph_data and not graph_data['predicted_settlement'].empty and x_value in graph_data['predicted_settlement'].index:
                        settlement_value_predicted = graph_data['predicted_settlement'].loc[x_value, 'settlement']
                        settlement_scatter.setData([x_value], [settlement_value_predicted])
                        settlement_text_content += f" X: {x_value}, Predicted: {settlement_value_predicted:.2f}"
                        settlement_text.setText(settlement_text_content)
                        settlement_text.setPos(x_value, settlement_value_predicted)
                        settlement_scatter.setVisible(True)
                        settlement_text.setVisible(True)
                        valid_index = True

                except Exception as e:
                    print(f"Error in mouseMoved: {type(e).__name__} - {e}")

                if not valid_index:
                    height_text.setVisible(False)
                    settlement_text.setVisible(False)
                    height_scatter.setVisible(False)
                    settlement_scatter.setVisible(False)

                return
        
        height_text.setVisible(False)
        settlement_text.setVisible(False)
        height_scatter.setVisible(False)
        settlement_scatter.setVisible(False)

    proxy = pg.SignalProxy(ui.height_graph.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
    ui.proxies.append(proxy)
    proxy = pg.SignalProxy(ui.settlement_graph.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
    ui.proxies.append(proxy)