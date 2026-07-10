# -*- coding: utf-8 -*-
"""
ecg.ui.theme
------------
Colour/font globals, ThemeConfig, and apply_theme_config().

This module owns all mutable UI colour globals (BG, PANEL, BLUE, ...) and
the PLOT dict.  After a theme change, call apply_theme_config(tc) which
updates every global in-place and then calls _sync_to_submodules() to push
the new values into all already-imported ecg.ui.* modules so that their
references (captured at import time) stay current.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import sys
import types as _types
from pathlib import Path
from typing import Optional

import customtkinter as ctk  # type: ignore[import-untyped]
import matplotlib.pyplot as plt

log = logging.getLogger("ecg")

# Optional deps (checked at runtime)
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    nk: Optional[_types.ModuleType] = None
    NK_AVAILABLE = False

try:
    import h5py  # type: ignore[import-untyped]
    H5_AVAILABLE = True
except ImportError:
    h5py: Optional[_types.ModuleType] = None
    H5_AVAILABLE = False

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ════════════════════════════════════════════════════════════
#  THEME SYSTEM  (colours, fonts, presets, persistence)
# ════════════════════════════════════════════════════════════

THEME_PATH:        Path = Path.home() / ".ecg_theme.json"
_THEME_PATH_LEGACY: Path = Path.home() / ".ecg_theme.pkl"   # migrated on first load

# App icon path
APP_ICON_PATH: str | None = None


class ThemeConfig:
    """Full UI colour + font configuration.

    All UI globals (BG, PANEL, CARD etc.) are derived from this object via
    apply_theme_config().  A ThemeConfig is serialised to THEME_PATH so it
    is restored automatically on next launch.
    """

    DEFAULT_FONT_FAMILIES: "list[str]" = [
        "SF Pro Text", "SF Pro Display", ".AppleSystemUIFont", "Helvetica Neue",
        "Segoe UI Variable", "Segoe UI", "Calibri", "Roboto",
        "Ubuntu", "Cantarell", "Inter",
        "Georgia", "Palatino", "Courier New", "Consolas",
    ]

    # Seven carefully designed presets
    PRESETS: "dict[str, dict]" = {
        # -- Light themes --------------------------------------------------
        "Apple": dict(
            is_dark=False,
            BG="#F5F5F7", PANEL="#FAFAFA", CARD="#FFFFFF",
            BORDER="#D2D2D7", BORDER2="#AEAEB2",
            TEXT="#1D1D1F", MUTED="#6E6E73", LIGHT="#AEAEB2",
            RED="#FF3B30",  BLUE="#0071E3", GREEN="#34C759", ORANGE="#FF9500",
            plot_signal="#0071E3", plot_rpeak="#34C759",
            description="Apple Human Interface Guidelines — clean, modern, precise",
        ),
        "Arctic": dict(
            is_dark=False,
            BG="#F8F9FB", PANEL="#EFF1F5", CARD="#FFFFFF",
            BORDER="#DDE1EA", BORDER2="#B0B7C3",
            TEXT="#0F172A", MUTED="#64748B", LIGHT="#94A3B8",
            RED="#C62828",  BLUE="#1A56DB", GREEN="#1B5E20", ORANGE="#D84315",
            plot_signal="#1A56DB", plot_rpeak="#1B5E20",
            description="Crisp off-white with deep cobalt accents",
        ),
        "Warm Lab": dict(
            is_dark=False,
            BG="#FAF7F0", PANEL="#F0EBE0", CARD="#FFFEF8",
            BORDER="#DDD3C0", BORDER2="#BEA882",
            TEXT="#2C1A0E", MUTED="#7D6A55", LIGHT="#A89880",
            RED="#B34700",  BLUE="#1A6FA8", GREEN="#2D6A1A", ORANGE="#C07000",
            plot_signal="#1A6FA8", plot_rpeak="#2D6A1A",
            description="Warm parchment tones for long analysis sessions",
        ),
        "Solarized": dict(
            is_dark=False,
            BG="#FDF6E3", PANEL="#EEE8D5", CARD="#FFFDF5",
            BORDER="#D3C9B3", BORDER2="#B0A892",
            TEXT="#073642", MUTED="#586E75", LIGHT="#839496",
            RED="#DC322F",  BLUE="#268BD2", GREEN="#859900", ORANGE="#CB4B16",
            plot_signal="#268BD2", plot_rpeak="#859900",
            description="Ethan Schoonover's classic readable palette",
        ),
        # -- Dark themes ---------------------------------------------------
        "Apple Dark": dict(
            is_dark=True,
            BG="#1C1C1E", PANEL="#2C2C2E", CARD="#3A3A3C",
            BORDER="#3A3A3C", BORDER2="#48484A",
            TEXT="#F5F5F7", MUTED="#98989D", LIGHT="#636366",
            RED="#FF453A",  BLUE="#0A84FF", GREEN="#30D158", ORANGE="#FF9F0A",
            plot_signal="#0A84FF", plot_rpeak="#30D158",
            description="Apple Dark Mode — macOS Monterey palette",
        ),
        "Nordic": dict(
            is_dark=True,
            BG="#2E3440", PANEL="#3B4252", CARD="#434C5E",
            BORDER="#4C566A", BORDER2="#5E81AC",
            TEXT="#ECEFF4", MUTED="#BEC7D8", LIGHT="#8A97A8",
            RED="#BF616A",  BLUE="#88C0D0", GREEN="#A3BE8C", ORANGE="#EBCB8B",
            plot_signal="#88C0D0", plot_rpeak="#A3BE8C",
            description="Nord palette: muted arctic blues, ultra-readable",
        ),
        "Midnight": dict(
            is_dark=True,
            BG="#0D1117", PANEL="#161B22", CARD="#1C2128",
            BORDER="#30363D", BORDER2="#484F58",
            TEXT="#E6EDF3", MUTED="#7D8590", LIGHT="#545D68",
            RED="#F85149",  BLUE="#388BFD", GREEN="#3FB950", ORANGE="#E3B341",
            plot_signal="#388BFD", plot_rpeak="#3FB950",
            description="GitHub Dark: deep black, vivid signal accents",
        ),
        "Dracula": dict(
            is_dark=True,
            BG="#282A36", PANEL="#1E1F29", CARD="#343746",
            BORDER="#44475A", BORDER2="#6272A4",
            TEXT="#F8F8F2", MUTED="#BD93F9", LIGHT="#6272A4",
            RED="#FF5555",  BLUE="#8BE9FD", GREEN="#50FA7B", ORANGE="#FFB86C",
            plot_signal="#8BE9FD", plot_rpeak="#50FA7B",
            description="High-contrast purple haze, zero eye-strain in dark rooms",
        ),
    }

    def __init__(self) -> None:
        self.preset_name: str   = "Apple"
        self.font_family: str   = _detect_system_font()
        self.font_scale:  float = 1.0
        self.is_dark:     bool  = False
        self._colors:     dict  = dict(ThemeConfig.PRESETS["Apple"])

    def __getattr__(self, name: str) -> str:
        """Forward colour attribute lookups to the internal _colors dict."""
        if name.startswith("_") or name in ("preset_name", "font_family",
                                             "font_scale", "is_dark"):
            raise AttributeError(name)
        try:
            return self._colors[name]
        except KeyError:
            raise AttributeError(name)

    def apply_preset(self, name: str) -> None:
        preset = self.PRESETS.get(name)
        if preset is None:
            return
        self.preset_name = name
        self._colors     = dict(preset)
        self.is_dark     = preset["is_dark"]

    def set_accent(self, color_name: str, hex_val: str) -> None:
        """Override a single accent colour (RED, BLUE, GREEN, or ORANGE)."""
        if color_name in ("RED", "BLUE", "GREEN", "ORANGE"):
            self._colors[color_name] = hex_val
            if not self.preset_name.endswith("*"):
                self.preset_name += "*"

    def font(self, size: int, bold: bool = False) -> "tuple[str, int, str]":
        scaled = max(8, int(round(size * self.font_scale)))
        return (self.font_family, scaled, "bold" if bold else "normal")

    def mono_font(self, size: int = 12) -> "tuple[str, int, str]":
        scaled = max(8, int(round(size * self.font_scale)))
        return ("Consolas", scaled, "normal")

    def save(self) -> None:
        """Persist this theme to ``THEME_PATH`` as a human-readable JSON file.

        JSON replaces the previous pickle format so theme files survive Python
        version upgrades and are inspectable with any text editor.
        """
        data = {
            "preset_name": self.preset_name,
            "font_family": self.font_family,
            "font_scale":  self.font_scale,
            "is_dark":     self.is_dark,
            "_colors":     self._colors,
        }
        with open(THEME_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "ThemeConfig":
        """Load theme from ``THEME_PATH`` (JSON).

        Falls back to defaults on any error.  If only a legacy ``.pkl`` file
        exists (Python upgrade scenario), it is migrated to JSON on first run
        and then deleted so the warning never appears again.
        """
        obj = cls()

        # ── Primary path: JSON ──────────────────────────────────────────────
        if THEME_PATH.exists():
            try:
                with open(THEME_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Only update recognised fields; ignore any unknown keys
                for field in ("preset_name", "font_family", "font_scale",
                              "is_dark", "_colors"):
                    if field in data:
                        setattr(obj, field, data[field])
                return obj
            except Exception as exc:
                log.warning(
                    "Theme file unreadable (%s) — using defaults. "
                    "Delete %s to suppress this message.", exc, THEME_PATH)
                return obj

        # ── Fallback: migrate legacy pickle on first run after upgrade ──────
        if _THEME_PATH_LEGACY.exists():
            try:
                with open(_THEME_PATH_LEGACY, "rb") as fh:
                    data = pickle.load(fh)
                obj.__dict__.update(data)
                obj.save()                    # write JSON
                _THEME_PATH_LEGACY.unlink()   # remove old pickle
                log.info("Theme migrated from pickle → JSON (%s)", THEME_PATH)
            except Exception as exc:
                log.warning("Legacy theme migration failed (%s) — using defaults", exc)

        return obj


def _detect_system_font() -> str:
    """Pick the best available UI font for the current platform."""
    import sys
    import tkinter as _tk
    import tkinter.font as _tkf
    try:
        root = _tk.Tk(); root.withdraw()
        available = set(_tkf.families(root)); root.destroy()
        candidates = (
            ["Segoe UI Variable", "Segoe UI", "Calibri"] if sys.platform == "win32"
            else ["SF Pro Text", "SF Pro Display", ".AppleSystemUIFont",
                  "Helvetica Neue", "Helvetica"] if sys.platform == "darwin"
            else ["Inter", "Ubuntu", "Cantarell", "Noto Sans", "DejaVu Sans"]
        )
        for f in candidates:
            if f in available:
                return f
    except Exception as e:
        log.warning("Font detection failed: %s", e)
    return "Helvetica"


# Active theme singleton (loaded from disk or defaults)
THEME: ThemeConfig = ThemeConfig.load()


# Module-level UI globals mutated by apply_theme_config()
# ─────────────────────────────────────────────────────────
# These are module-level names because matplotlib drawing closures and widget
# constructors throughout ECGApp capture them by name at call time, not at
# definition time.  apply_theme_config() updates all of them atomically.
#
# ThemeColors is a typed snapshot — use it when you need to pass the full
# colour set as a value (e.g. to ExcelExporter or test helpers) without
# depending on the mutable globals directly.
@dataclasses.dataclass(frozen=True)
class ThemeColors:
    """Immutable snapshot of the active UI colour palette.

    Created by ``ThemeColors.from_globals()`` whenever you need a stable
    reference to the current theme (e.g. to pass into a background worker
    or a standalone test).  The module-level globals (BG, CARD, GREEN…)
    remain the live source of truth; this class is a value copy.
    """
    BG:      str = "#F5F5F7"
    PANEL:   str = "#FAFAFA"
    CARD:    str = "#FFFFFF"
    BORDER:  str = "#D2D2D7"
    BORDER2: str = "#AEAEB2"
    RED:     str = "#FF3B30"
    BLUE:    str = "#0071E3"
    GREEN:   str = "#34C759"
    ORANGE:  str = "#FF9500"
    MUTED:   str = "#6E6E73"
    LIGHT:   str = "#AEAEB2"
    TEXT:    str = "#1D1D1F"

    @classmethod
    def from_globals(cls) -> "ThemeColors":
        """Capture the current module-level colour globals into a frozen copy."""
        return cls(
            BG=BG, PANEL=PANEL, CARD=CARD, BORDER=BORDER, BORDER2=BORDER2,
            RED=RED, BLUE=BLUE, GREEN=GREEN, ORANGE=ORANGE,
            MUTED=MUTED, LIGHT=LIGHT, TEXT=TEXT,
        )


BG      = THEME._colors.get("BG",      "#F5F5F7")
PANEL   = THEME._colors.get("PANEL",   "#FAFAFA")
CARD    = THEME._colors.get("CARD",    "#FFFFFF")
BORDER  = THEME._colors.get("BORDER",  "#D2D2D7")
BORDER2 = THEME._colors.get("BORDER2", "#AEAEB2")
RED     = THEME._colors.get("RED",     "#FF3B30")
BLUE    = THEME._colors.get("BLUE",    "#0071E3")
GREEN   = THEME._colors.get("GREEN",   "#34C759")
ORANGE  = THEME._colors.get("ORANGE",  "#FF9500")
MUTED   = THEME._colors.get("MUTED",   "#6E6E73")
LIGHT   = THEME._colors.get("LIGHT",   "#AEAEB2")
TEXT    = THEME._colors.get("TEXT",    "#1D1D1F")

# ── Extended colour palette (hover states, domain-specific accents) ──────
BLUE_DARK    = "#1565C0"   # dark variant of BLUE (titles, active states)
BLUE_HOVER   = "#1344A8"   # hover state for BLUE buttons
BLUE_MID     = "#2196F3"   # mid-tone blue (info badges)
BLUE_DEEP    = "#0D47A1"   # deep navy blue
PURPLE       = "#6A1B9A"   # interpretation / annotation accent
PURPLE_DARK  = "#4A148C"   # darker purple (annotation button)
PINK         = "#AD1457"   # PR / cardiac axis accent
TEAL         = "#26A69A"   # HRV / frequency domain accent
CYAN         = "#00BCD4"   # nonlinear metrics accent
CYAN_BRIGHT  = "#00E5FF"   # Poincaré / scatter accent
GREEN_DARK   = "#2E7D32"   # QRS / normal range accent
GREEN_MID    = "#4CAF50"   # healthy threshold indicator
ORANGE_DARK  = "#E65100"   # RR / heart-rate accent
ORANGE_DEEP  = "#BF360C"   # QTc / deep orange accent
AMBER        = "#FF9800"   # warning / marginal value
AMBER_DARK   = "#FF6F00"   # QT dispersion accent
RED_DARK     = "#B71C1C"   # arrhythmia / danger hover
RED_MID      = "#F44336"   # arrhythmia mid-tone
RED_LIGHT    = "#EF5350"   # arrhythmia light tone
CORAL        = "#FF7043"   # rolling HRV accent
NAVY         = "#1A1A2E"   # dark background accent
GRAY         = "#888888"   # neutral / disabled text
GRAY_LIGHT   = "#E0E0E0"   # light separator / disabled bg



def _make_plot_theme(tc: "ThemeConfig") -> "dict[str, str]":
    if tc.is_dark:
        return dict(
            bg=tc._colors.get("BG",     "#2E3440"),
            axes=tc._colors.get("CARD", "#434C5E"),
            grid=tc._colors.get("BORDER","#4C566A"),
            text=tc._colors.get("TEXT",  "#ECEFF4"),
            muted=tc._colors.get("MUTED","#BEC7D8"),
            border=tc._colors.get("BORDER2","#5E81AC"),
            signal=tc._colors.get("plot_signal", "#88C0D0"),
            raw=tc._colors.get("BORDER2", "#6272A4"),
            rpeak_ok=tc._colors.get("plot_rpeak", "#A3BE8C"),
            rpeak_bad=tc._colors.get("MUTED", "#7D8590"),
            threshold=tc._colors.get("RED", "#BF616A"),
        )
    else:
        return dict(
            bg=tc._colors.get("BG",     "#F8F9FB"),
            axes=tc._colors.get("CARD", "#FFFFFF"),
            grid=tc._colors.get("BORDER","#DDE1EA"),
            text=tc._colors.get("TEXT",  "#0F172A"),
            muted=tc._colors.get("MUTED","#64748B"),
            border=tc._colors.get("BORDER2","#B0B7C3"),
            signal=tc._colors.get("plot_signal", "#1A56DB"),
            raw=tc._colors.get("BORDER2", "#90A4AE"),
            rpeak_ok=tc._colors.get("plot_rpeak", "#1B5E20"),
            rpeak_bad=tc._colors.get("BORDER2", "#B0B7C3"),
            threshold=tc._colors.get("RED", "#C62828"),
        )


PLOT: dict[str, str] = _make_plot_theme(THEME)


def apply_theme_config(tc: "ThemeConfig") -> None:
    """Update all module-level colour globals and matplotlib rcParams from tc.

    Must be followed by ECGApp._rebuild_ui() to repaint the entire widget tree.
    """
    global BG, PANEL, CARD, BORDER, BORDER2, RED, BLUE, GREEN, ORANGE
    global MUTED, LIGHT, TEXT, PLOT
    global FONT_TITLE, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL, FONT_BODY, FONT_MONO
    global FONT_KPI_VALUE, FONT_KPI_LABEL, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR
    global FONT_MICRO, FONT_HINT, FONT_BADGE, FONT_SUBSECTION, FONT_CARD_TITLE

    BG      = tc._colors.get("BG",      BG)
    PANEL   = tc._colors.get("PANEL",   PANEL)
    CARD    = tc._colors.get("CARD",    CARD)
    BORDER  = tc._colors.get("BORDER",  BORDER)
    BORDER2 = tc._colors.get("BORDER2", BORDER2)
    RED     = tc._colors.get("RED",     RED)
    BLUE    = tc._colors.get("BLUE",    BLUE)
    GREEN   = tc._colors.get("GREEN",   GREEN)
    ORANGE  = tc._colors.get("ORANGE",  ORANGE)
    MUTED   = tc._colors.get("MUTED",   MUTED)
    LIGHT   = tc._colors.get("LIGHT",   LIGHT)
    TEXT    = tc._colors.get("TEXT",    TEXT)

    plot_theme = _make_plot_theme(tc)
    PLOT.update(plot_theme)

    plt.rcParams.update({
        "figure.facecolor": plot_theme["bg"],
        "axes.facecolor":   plot_theme["axes"],
        "grid.color":       plot_theme["grid"],
        "axes.labelcolor":  plot_theme["text"],
        "xtick.color":      plot_theme["muted"],
        "ytick.color":      plot_theme["muted"],
        "text.color":       plot_theme["text"],
    })

    ctk.set_appearance_mode("dark" if tc.is_dark else "light")

    FONT_TITLE        = tc.font(15, bold=True)
    FONT_SECTION_HDR  = tc.font(12, bold=True)
    FONT_LABEL        = tc.font(12)
    FONT_SMALL        = tc.font(11)
    FONT_BODY         = tc.font(12)
    FONT_MONO         = tc.mono_font(12)
    # Extended scale-aware font tokens
    FONT_KPI_VALUE    = tc.font(14, bold=True)
    FONT_KPI_LABEL    = tc.font(9)
    FONT_BTN_PRIMARY  = tc.font(13, bold=True)
    FONT_BTN_SEC      = tc.font(11)
    FONT_SIDEBAR_HDR  = tc.font(11, bold=True)
    FONT_MICRO        = tc.font(8)
    FONT_HINT         = tc.font(10)
    FONT_BADGE        = tc.font(9,  bold=True)
    FONT_SUBSECTION   = tc.font(10, bold=True)
    FONT_CARD_TITLE   = tc.font(12, bold=True)


    _sync_to_submodules()


def apply_plot_theme(dark: bool) -> None:
    """Legacy shim used by _toggle_dark -- delegates to apply_theme_config."""
    THEME.is_dark = dark
    apply_theme_config(THEME)


def make_font(size: int = 12, bold: bool = False) -> "tuple[str, int, str]":
    """Return a font tuple scaled by the active THEME.font_scale."""
    return THEME.font(size, bold)


FONT_TITLE        = make_font(15, bold=True)
FONT_SECTION_HDR  = make_font(12, bold=True)
FONT_LABEL        = make_font(12)
FONT_SMALL        = make_font(11)
FONT_BODY         = make_font(12)
FONT_MONO         = THEME.mono_font(12)
# Extended scale-aware font tokens
FONT_KPI_VALUE    = make_font(14, bold=True)
FONT_KPI_LABEL    = make_font(9)
FONT_BTN_PRIMARY  = make_font(13, bold=True)
FONT_BTN_SEC      = make_font(11)
FONT_SIDEBAR_HDR  = make_font(11, bold=True)
# Tokens supplémentaires — couvrent les usages réels de make_font(N) trouvés
# dans app.py qui contournaient la hiérarchie nommée existante (79 appels
# bruts avant cette extension). Tailles choisies pour matcher exactement
# l'usage observé, sans changer aucun rendu visuel existant.
FONT_MICRO        = make_font(8)              # unités d'axes, libellés ultra-discrets
FONT_HINT         = make_font(10)             # texte d'aide / statut sous un bouton
FONT_BADGE        = make_font(9,  bold=True)  # badges courts (sévérité, tag d'état)
FONT_SUBSECTION   = make_font(10, bold=True)  # en-tête de sous-bloc (sous FONT_SIDEBAR_HDR)
FONT_CARD_TITLE   = make_font(12, bold=True)  # titre de carte (alias sémantique de FONT_SECTION_HDR)

def _sync_to_submodules() -> None:
    """Push updated colour/font globals into all already-imported ecg.ui.* modules.

    Python module-level ``from ecg.ui.theme import BG`` copies the value at
    import time.  When apply_theme_config() mutates the globals here, those
    copies go stale.  This function iterates sys.modules and overwrites the
    copied names so every submodule always sees the current theme without
    needing a full reimport.
    """
    _colour_names = (
        "BG", "PANEL", "CARD", "BORDER", "BORDER2",
        "RED", "BLUE", "GREEN", "ORANGE", "MUTED", "LIGHT", "TEXT", "PLOT",
        "BLUE_DARK", "BLUE_HOVER", "BLUE_MID", "BLUE_DEEP",
        "PURPLE", "PURPLE_DARK", "PINK", "TEAL", "CYAN", "CYAN_BRIGHT",
        "GREEN_DARK", "GREEN_MID", "ORANGE_DARK", "ORANGE_DEEP",
        "AMBER", "AMBER_DARK", "RED_DARK", "RED_MID", "RED_LIGHT",
        "CORAL", "NAVY", "GRAY", "GRAY_LIGHT",
    )
    _font_names = (
        "FONT_TITLE", "FONT_SECTION_HDR", "FONT_LABEL", "FONT_SMALL",
        "FONT_BODY", "FONT_MONO", "FONT_KPI_VALUE", "FONT_KPI_LABEL",
        "FONT_BTN_PRIMARY", "FONT_BTN_SEC", "FONT_SIDEBAR_HDR",
        "FONT_MICRO", "FONT_HINT", "FONT_BADGE", "FONT_SUBSECTION", "FONT_CARD_TITLE",
    )
    # This codebase is a flat module layout (app.py, sidebar.py, plots.py,
    # dialogs.py, export.py, models.py, detection.py, loaders.py, analysis.py
    # — all top-level, no "ecg.*" package). The previous filter checked
    # `mod_name.startswith("ecg.")`, which never matched any real module here,
    # so this function was silently a no-op: every module that did
    # `from theme import BG, PANEL, ...` at import time kept its OWN stale
    # copy of those names forever, and only the PLOT dict (mutated in place
    # via .update(), not rebound) ever picked up a theme change. Switching
    # themes therefore left most colours/fonts unchanged everywhere except
    # inside theme.py itself.
    _submodule_names = (
        "app", "sidebar", "plots", "dialogs", "export",
        "models", "detection", "loaders", "analysis",
    )
    _this = sys.modules[__name__]
    _all  = _colour_names + _font_names
    for mod_name, mod in list(sys.modules.items()):
        if mod_name in _submodule_names and mod is not _this and isinstance(mod, type(sys)):
            for name in _all:
                if hasattr(mod, name) and hasattr(_this, name):
                    try:
                        setattr(mod, name, getattr(_this, name))
                    except (AttributeError, TypeError):
                        pass
