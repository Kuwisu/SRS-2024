from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QTabWidget


class TabWidget(QTabWidget):
    """
    This class overrides the sizing method that QTabWidget uses so that
    it will automatically resize to fit the current widget rather than
    fitting the size of the largest widget.

    This code has been derived from user musicamante's implementation
    Link: https://stackoverflow.com/a/66053591
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        return self.sizeHint()

    def sizeHint(self):
        widget_size = self.currentWidget().sizeHint()
        tab_size = self.tabBar().sizeHint()
        size = QSize(
            max(widget_size.width(), tab_size.width()),
            widget_size.height() + tab_size.height()
        )
        return size