"""The suite runs with QT reported (ecg.core.models.QT_REPORTED = True): the
QT / QTc code is hidden from the app for now but kept, and its tests keep it
correct for when it is shown again. Set before any other ecg module is
imported, since several read the switch at import time. That the app hides
QT by default is checked in a separate process (tests/core/test_qt_hidden.py).
"""
import ecg.core.models as _models

_models.QT_REPORTED = True
