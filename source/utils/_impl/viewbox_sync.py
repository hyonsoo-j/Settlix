def link_x_axes(view1, view2):
    def update_view(view):
        x_range, _ = view.viewRange()
        if view is view1:
            view2.setXRange(*x_range, padding=0)
        else:
            view1.setXRange(*x_range, padding=0)

    view1.sigXRangeChanged.connect(lambda: update_view(view1))
    view2.sigXRangeChanged.connect(lambda: update_view(view2))
