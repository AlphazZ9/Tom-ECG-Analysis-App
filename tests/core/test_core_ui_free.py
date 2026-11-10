"""ecg.core is UI-free, including imports that only exist for type checking."""
import ast
from pathlib import Path

import ecg.core
import ecg.core.models
from ecg.core.models import FilterParams, FilterWidgetHost


def _ui_imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        found += [f"{path.name}:{node.lineno} {n}" for n in names
                  if n == "ecg.ui" or n.startswith("ecg.ui.")]
    return found


def test_no_module_in_core_imports_ecg_ui():
    core = Path(ecg.core.__file__).parent
    assert [hit for p in sorted(core.rglob("*.py")) for hit in _ui_imports(p)] == []


def test_filter_widget_host_declares_the_attributes_from_widgets_requires():
    # Attributes read through getattr() are optional and stay undeclared.
    src = Path(ecg.core.models.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "from_widgets")
    used = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "app"}
    declared = set(FilterWidgetHost.__annotations__) | {"_safe_float"}
    assert used == declared
    assert FilterParams.from_widgets.__func__.__annotations__["app"] == "FilterWidgetHost"
