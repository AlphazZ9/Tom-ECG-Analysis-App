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

import colorsys
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

    # Exactly two themes -- colours pulled from the Adaptive Workbench mockup
    # (ecg_layout_concepts.html) shown during the redesign brief.
    PRESETS: "dict[str, dict]" = {
        "Light": dict(
            is_dark=False,
            # PANEL == CARD deliberately (both pure white): the app's one
            # unified "neutral surface" token (chrome, workspace, plots,
            # dialogs -- see app.py's self.main/self.tabs and theme.py's
            # _make_plot_theme()) should read as white, not grey. Matching
            # CARD's exact value rather than a "close but different" off-white
            # avoids reintroducing the seam that BG-vs-PANEL used to cause.
            BG="#F4F5F7", PANEL="#FFFFFF", CARD="#FFFFFF",
            BORDER="#DCE0E5", BORDER2="#C7CDD4",
            TEXT="#1B222C", MUTED="#5B6472", LIGHT="#939DA8",
            RED="#D64545",  BLUE="#2D6CDF", GREEN="#2E9E5B", ORANGE="#E08A2A",
            plot_signal="#2D6CDF", plot_rpeak="#2E9E5B", plot_filtered="#6A1B9A",
            description="Clean, modern, restrained — the app's default look",
        ),
        "Dark": dict(
            is_dark=True,
            BG="#14171C", PANEL="#1B1F26", CARD="#1B1F26",
            BORDER="#2B3138", BORDER2="#3A4149",
            TEXT="#E7EAEE", MUTED="#9BA5B2", LIGHT="#656E79",
            RED="#E5695F",  BLUE="#5B8DF5", GREEN="#4CBA7C", ORANGE="#E8A552",
            plot_signal="#5B8DF5", plot_rpeak="#4CBA7C", plot_filtered="#9125D2",
            description="The same design, low-light",
        ),
    }

    def __init__(self) -> None:
        self.preset_name: str   = "Light"
        self.font_family: str   = _detect_system_font()
        self.font_scale:  float = 1.0
        self.is_dark:     bool  = False
        self._colors:     dict  = dict(ThemeConfig.PRESETS["Light"])

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
                # Only update recognised fields; ignore any unknown keys.
                # font_family is deliberately excluded -- it's no longer
                # user-configurable (no picker), so it always stays whatever
                # _detect_system_font() gave the fresh obj = cls() above,
                # rather than carrying forward a value someone picked back
                # when the now-removed font-family combo still existed.
                for field in ("preset_name", "font_scale", "is_dark", "_colors"):
                    if field in data:
                        setattr(obj, field, data[field])
                # Migrate a preset removed in the 2-theme consolidation (e.g.
                # a pre-existing ~/.ecg_theme.json still says "Apple" or
                # "Nordic*") -- snap to whichever current preset matches the
                # saved is_dark, re-deriving _colors cleanly instead of
                # carrying the old palette forward under a name that no
                # longer exists.
                if obj.preset_name.rstrip("*") not in cls.PRESETS:
                    obj.apply_preset("Dark" if obj.is_dark else "Light")
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
# Base (light-mode) hex values -- dark-mode variants are derived
# algorithmically by _adapt_for_mode()/_apply_extended_palette() below,
# computed once at import time and again in apply_theme_config(), rather
# than hand-tuned per preset: none of these historically varied with
# light/dark at all (unlike BG/PANEL/BLUE/... above, which come from
# THEME.PRESETS and already have real per-preset light+dark values).
_EXTENDED_PALETTE_BASE: "dict[str, str]" = {
    "BLUE_DARK":   "#1565C0",   # dark variant of BLUE (titles, active states)
    "BLUE_HOVER":  "#1344A8",   # hover state for BLUE buttons
    "BLUE_MID":    "#2196F3",   # mid-tone blue (info badges)
    "BLUE_DEEP":   "#0D47A1",   # deep navy blue
    "PURPLE":      "#6A1B9A",   # interpretation / annotation accent
    "PURPLE_DARK": "#4A148C",   # darker purple (annotation button)
    "PURPLE_LIGHT": "#AB47BC",  # artifact-review "duplicate" marker accent
    "PINK":        "#AD1457",   # PR / cardiac axis accent
    "TEAL":        "#26A69A",   # HRV / frequency domain accent
    "TEAL_DARK":   "#00695C",   # hover state for TEAL buttons (Compare Segments)
    "CYAN":        "#00BCD4",   # nonlinear metrics accent
    "CYAN_BRIGHT": "#00E5FF",   # Poincaré / scatter accent
    "GREEN_DARK":  "#2E7D32",   # QRS / normal range accent
    "GREEN_MID":   "#4CAF50",   # healthy threshold indicator
    "ORANGE_DARK": "#E65100",   # RR / heart-rate accent
    "ORANGE_DEEP": "#BF360C",   # QTc / deep orange accent
    "AMBER":       "#FF9800",   # warning / marginal value
    "AMBER_DARK":  "#FF6F00",   # QT dispersion accent
    "RED_DARK":    "#B71C1C",   # arrhythmia / danger hover
    "RED_MID":     "#F44336",   # arrhythmia mid-tone
    "RED_LIGHT":   "#EF5350",   # arrhythmia light tone
    "CORAL":       "#FF7043",   # rolling HRV accent
    "NAVY":        "#1A1A2E",   # dark background accent
    "GRAY":        "#888888",   # neutral / disabled text
    "GRAY_LIGHT":  "#E0E0E0",   # light separator / disabled bg
}


# Type-only declarations (no assignment -- a no-op at runtime) so static
# analyzers can resolve `from ecg.ui.theme import BLUE_DARK` etc. across
# every importing file. The real string values are still set exactly as
# before, dynamically, by _apply_extended_palette()'s globals().update()
# below; this just gives that dynamic write a declared type to attach to,
# closing the pyright/Pylance "unknown import symbol" blind spot documented
# on _apply_extended_palette() without changing any runtime behaviour.
BLUE_DARK: str; BLUE_HOVER: str; BLUE_MID: str; BLUE_DEEP: str
PURPLE: str; PURPLE_DARK: str; PURPLE_LIGHT: str
PINK: str; TEAL: str; TEAL_DARK: str; CYAN: str; CYAN_BRIGHT: str
GREEN_DARK: str; GREEN_MID: str
ORANGE_DARK: str; ORANGE_DEEP: str; AMBER: str; AMBER_DARK: str
RED_DARK: str; RED_MID: str; RED_LIGHT: str
CORAL: str; NAVY: str; GRAY: str; GRAY_LIGHT: str


def _adapt_for_mode(hex_val: str, is_dark: bool) -> str:
    """Lighten an extended-palette colour for dark mode; unchanged in light.

    These accents never varied with the theme preset before. Rather than
    hand-tune 24 colours x 8 presets, bump HSL lightness by a fixed amount
    in dark mode so they stay legible/vivid against dark backgrounds
    instead of looking identical to (and often too dark/muddy on) light
    ones. Light mode returns hex_val unchanged, so the default theme's
    rendering is byte-for-byte the same as before this function existed.
    """
    if not is_dark:
        return hex_val
    r = int(hex_val[1:3], 16) / 255.0
    g = int(hex_val[3:5], 16) / 255.0
    b = int(hex_val[5:7], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, min(1.0, l + 0.13), s)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))


def _apply_extended_palette(is_dark: bool) -> "dict[str, str]":
    """(Re)compute every extended-palette global for the given mode.

    Uses globals()[...] rather than a `global` statement + per-name
    assignment -- every one of these 24 names follows the exact same
    "look up base value, adapt for dark, done" recipe, so a loop expresses
    that uniformity directly. globals() here is theme.py's own module
    dict regardless of the caller, so `from ecg.ui.theme import BLUE_DARK`
    and _sync_to_submodules()'s getattr/setattr both see real attributes,
    identical in effect to writing `BLUE_DARK = ...` by hand.

    Also returns the computed dict, so callers that need a specific value
    right away (the semantic COLOR_* aliases below) can subscript it
    instead of reading the bare name -- a bare `BLUE_HOVER` reference has
    no static assignment anywhere in this file (only the dynamic
    globals()[...] write above), which is invisible to pyflakes and would
    read as an undefined name in any function that isn't this one.
    """
    values = {name: _adapt_for_mode(base, is_dark)
              for name, base in _EXTENDED_PALETTE_BASE.items()}
    globals().update(values)
    return values


_EXT = _apply_extended_palette(THEME.is_dark)


def _make_artifact_type_color(ext: "dict[str, str]") -> "dict[str, str]":
    return {"nonphysio": ext["RED_LIGHT"], "ectopic": ext["AMBER"], "duplicate": ext["PURPLE_LIGHT"]}


ARTIFACT_TYPE_COLOR: dict[str, str] = _make_artifact_type_color(_EXT)

# ── Semantic action-colour tokens (rationalised palette, UI redesign) ────
# Aliases onto the already theme-aware core palette above -- not new
# colours, just names that describe INTENT (primary action / success /
# warning / danger / secondary) instead of a raw hue, so call sites stop
# picking colours ad hoc. See _btn()'s variant= parameter in app.py.
COLOR_PRIMARY         = BLUE
COLOR_PRIMARY_HOVER   = _EXT["BLUE_HOVER"]
COLOR_SUCCESS         = GREEN
COLOR_SUCCESS_HOVER   = _EXT["GREEN_DARK"]
COLOR_WARNING         = ORANGE
COLOR_WARNING_HOVER   = _EXT["ORANGE_DARK"]
COLOR_DANGER          = RED
COLOR_DANGER_HOVER    = _EXT["RED_DARK"]
COLOR_SECONDARY       = BORDER
COLOR_SECONDARY_HOVER = BORDER2

# ── Layout spacing tokens (single source of truth) ────────────────────────
# Previously duplicated byte-for-byte in app.py (twice) and re-derived with
# a `_`-prefixed local copy in analysis_controller.py/plots.py/sidebar.py to
# work around an imagined circular import -- theme.py is a leaf module (it
# imports nothing from ecg.ui.*), so every one of those modules can import
# these directly instead.
SPACE_XS = 2
SPACE_S  = 4
SPACE_M  = 8
SPACE_L  = 12


def _make_plot_theme(tc: "ThemeConfig") -> "dict[str, str]":
    if tc.is_dark:
        return dict(
            # bg == axes deliberately: BG and PANEL are two subtly different
            # greys (workspace vs chrome), which read as an unwanted seam
            # between a plot's figure margin and its own axes rectangle, and
            # between the plot and the panel around it. Using PANEL for both
            # makes every plot one flat, consistent grey that matches its
            # surroundings -- see also self.main/self.tabs in app.py, which
            # apply the same fix to the workspace area itself.
            bg=tc._colors.get("PANEL",  "#1B1F26"),
            axes=tc._colors.get("PANEL", "#1B1F26"),
            grid=tc._colors.get("BORDER","#4C566A"),
            text=tc._colors.get("TEXT",  "#ECEFF4"),
            muted=tc._colors.get("MUTED","#BEC7D8"),
            border=tc._colors.get("BORDER2","#5E81AC"),
            signal=tc._colors.get("plot_signal", "#88C0D0"),
            raw=tc._colors.get("BLUE", "#5B8DF5"),
            filtered=tc._colors.get("plot_filtered", "#9125D2"),
            rpeak_ok=tc._colors.get("plot_rpeak", "#A3BE8C"),
            rpeak_bad=tc._colors.get("MUTED", "#7D8590"),
            threshold=tc._colors.get("RED", "#BF616A"),
        )
    else:
        return dict(
            bg=tc._colors.get("PANEL",  "#ECEEF1"),
            axes=tc._colors.get("PANEL", "#ECEEF1"),
            grid=tc._colors.get("BORDER","#DDE1EA"),
            text=tc._colors.get("TEXT",  "#0F172A"),
            muted=tc._colors.get("MUTED","#64748B"),
            border=tc._colors.get("BORDER2","#B0B7C3"),
            signal=tc._colors.get("plot_signal", "#1A56DB"),
            raw=tc._colors.get("BLUE", "#2D6CDF"),
            filtered=tc._colors.get("plot_filtered", "#6A1B9A"),
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
    global COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_SUCCESS, COLOR_SUCCESS_HOVER
    global COLOR_WARNING, COLOR_WARNING_HOVER, COLOR_DANGER, COLOR_DANGER_HOVER
    global COLOR_SECONDARY, COLOR_SECONDARY_HOVER
    global FONT_TITLE, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL, FONT_BODY, FONT_MONO
    global FONT_KPI_VALUE, FONT_KPI_LABEL, FONT_KPI_HERO, FONT_BTN_PRIMARY, FONT_BTN_SEC
    global FONT_SIDEBAR_HDR
    global FONT_MICRO, FONT_HINT, FONT_BADGE, FONT_SUBSECTION, FONT_CARD_TITLE
    global FONT_DIALOG_TITLE

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

    # Extended palette must be recomputed before the semantic aliases below,
    # which reference BLUE_HOVER/GREEN_DARK/ORANGE_DARK/RED_DARK from it.
    _ext = _apply_extended_palette(tc.is_dark)
    ARTIFACT_TYPE_COLOR.update(_make_artifact_type_color(_ext))

    COLOR_PRIMARY         = BLUE
    COLOR_PRIMARY_HOVER   = _ext["BLUE_HOVER"]
    COLOR_SUCCESS         = GREEN
    COLOR_SUCCESS_HOVER   = _ext["GREEN_DARK"]
    COLOR_WARNING         = ORANGE
    COLOR_WARNING_HOVER   = _ext["ORANGE_DARK"]
    COLOR_DANGER          = RED
    COLOR_DANGER_HOVER    = _ext["RED_DARK"]
    COLOR_SECONDARY       = BORDER
    COLOR_SECONDARY_HOVER = BORDER2

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
    FONT_KPI_HERO     = tc.font(20, bold=True)
    FONT_BTN_PRIMARY  = tc.font(13, bold=True)
    FONT_BTN_SEC      = tc.font(11)
    FONT_SIDEBAR_HDR  = tc.font(11, bold=True)
    FONT_MICRO        = tc.font(8)
    FONT_HINT         = tc.font(10)
    FONT_BADGE        = tc.font(9,  bold=True)
    FONT_SUBSECTION   = tc.font(10, bold=True)
    FONT_CARD_TITLE   = tc.font(12, bold=True)
    FONT_DIALOG_TITLE = tc.font(18, bold=True)

    _sync_to_submodules()


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
FONT_KPI_HERO     = make_font(20, bold=True)  # one emphasized number in a stat grid
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
FONT_DIALOG_TITLE = make_font(18, bold=True)  # en-tête de dialogue pleine page (ex. Appearance Settings)

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
        "PURPLE", "PURPLE_DARK", "PURPLE_LIGHT", "PINK", "TEAL", "TEAL_DARK", "CYAN", "CYAN_BRIGHT",
        "GREEN_DARK", "GREEN_MID", "ORANGE_DARK", "ORANGE_DEEP",
        "AMBER", "AMBER_DARK", "RED_DARK", "RED_MID", "RED_LIGHT",
        "CORAL", "NAVY", "GRAY", "GRAY_LIGHT",
        # Semantic action-colour tokens (rationalised palette, UI redesign)
        "COLOR_PRIMARY", "COLOR_PRIMARY_HOVER", "COLOR_SUCCESS", "COLOR_SUCCESS_HOVER",
        "COLOR_WARNING", "COLOR_WARNING_HOVER", "COLOR_DANGER", "COLOR_DANGER_HOVER",
        "COLOR_SECONDARY", "COLOR_SECONDARY_HOVER",
        "ARTIFACT_TYPE_COLOR",
    )
    _font_names = (
        "FONT_TITLE", "FONT_SECTION_HDR", "FONT_LABEL", "FONT_SMALL",
        "FONT_BODY", "FONT_MONO", "FONT_KPI_VALUE", "FONT_KPI_LABEL", "FONT_KPI_HERO",
        "FONT_BTN_PRIMARY", "FONT_BTN_SEC", "FONT_SIDEBAR_HDR",
        "FONT_MICRO", "FONT_HINT", "FONT_BADGE", "FONT_SUBSECTION", "FONT_CARD_TITLE",
        "FONT_DIALOG_TITLE",
    )
    # Every module below does `from ecg.ui.theme import BG, PANEL, ...` at
    # import time, which copies the value once -- when apply_theme_config()
    # rebinds these globals here, those copies go stale unless re-synced.
    # Names are the fully-qualified dotted paths under the ecg.* package
    # (matching sys.modules keys) since the reorg out of the old flat
    # module layout; a bare "sidebar" or "app" would no longer match
    # anything in sys.modules and this loop would silently become a no-op
    # again (that exact bug -- a stale pre-package-move name list -- is why
    # this function was a no-op for a while before the reorg too).
    _submodule_names = (
        "ecg.ui.app", "ecg.ui.sidebar", "ecg.ui.plots", "ecg.ui.dialogs",
        "ecg.io.export", "ecg.core.models", "ecg.core.detection",
        "ecg.io.loaders", "ecg.core.analysis", "ecg.ui.wave_editor",
        # Controller modules extracted from app.py -- each imports its own
        # subset of colour/font names at module scope, so each needs the
        # same re-sync or its colours freeze at whatever theme was active
        # when it was first imported (the same bug this function exists to
        # fix, just introduced by any new sibling module going forward).
        "ecg.ui.navigation_controller", "ecg.ui.export_controller",
        "ecg.ui.plot_controller", "ecg.ui.session_controller",
        "ecg.ui.detection_controller", "ecg.ui.signal_controller",
        "ecg.ui.analysis_controller",
        # New in the UI redesign (Phase 0) -- same requirement as every
        # module above: it imports colour/font names at module scope.
        "ecg.ui.widgets",
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
            # A plain name copy only helps modules that read PLOT/BG/... etc.
            # directly. A module that derives its OWN values from those names
            # once at import time (e.g. plots.py's Y-scale-bar colours, which
            # are computed from PANEL/BG/TEXT/BORDER2/MUTED) needs a chance to
            # recompute those derived values now that the names above were
            # just refreshed. Any submodule that defines a bare
            # `_on_theme_changed()` function gets called here for that.
            hook = getattr(mod, "_on_theme_changed", None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:
                    log.debug("%s._on_theme_changed() failed: %s", mod_name, exc)
