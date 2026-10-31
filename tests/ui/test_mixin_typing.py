"""ECGApp mixins must stay checkable by pyright.

Each mixin method reads attributes defined on sibling mixins, so an
unannotated ``self`` makes every such read an "unknown attribute" error and
buries real typos. ``self: ECGApp`` is what keeps ``self.xxx`` checked.
"""
import ast
import inspect
from pathlib import Path

from ecg.ui.app import ECGApp


def _mixins():
    return [c for c in ECGApp.__mro__[1:] if c.__module__.startswith("ecg.ui.app_")
            or c.__module__.startswith("ecg.ui.tab_")
            or c.__module__ in ("ecg.ui.params_dialog", "ecg.ui.compare_segments")]


def _functions(cls):
    """Every function defined on cls, including property getters and setters."""
    for name, member in vars(cls).items():
        if isinstance(member, property):
            for f in (member.fget, member.fset, member.fdel):
                if f is not None:
                    yield name, f
        elif inspect.isfunction(member):
            yield name, member


def test_all_fifteen_mixins_are_covered():
    assert len(_mixins()) == 15


def test_every_mixin_method_annotates_self_as_ecgapp():
    missing = [f"{c.__name__}.{name}" for c in _mixins() for name, f in _functions(c)
               if f.__annotations__.get("self") != "ECGApp"]
    assert missing == []


def test_mixins_do_not_import_ecgapp_at_runtime():
    # The import must stay under TYPE_CHECKING: app.py imports every mixin.
    for c in _mixins():
        tree = ast.parse(Path(inspect.getsourcefile(c)).read_text(encoding="utf-8"))
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any(getattr(n, "module", "") == "ecg.ui.app" for n in top_level), c.__name__
