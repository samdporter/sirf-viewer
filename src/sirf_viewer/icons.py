# yourpkg/icons.py
import sys
from importlib.resources import files
from PyQt5.QtGui import QIcon


def app_icon() -> QIcon:
    base = files("sirf_viewer.assets")
    if sys.platform.startswith("win"):
        return QIcon(str(base / "app.ico"))
    else:
        return QIcon(str(base / "app_256.png"))
