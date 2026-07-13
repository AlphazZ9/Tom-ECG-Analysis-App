# -*- coding: utf-8 -*-
"""
ecg.ui.plots
------------
Matplotlib-to-Tkinter bridge: CanvasSlot and style_axes().

CanvasSlot improvements over the original:
• Optional compact Y-scale bar (yscale_bar=True) with min/max entry fields
  + "Auto" reset.  The bar sits above the canvas and never takes figure space.
• Axes-limit read-back after each redraw so the fields always reflect the
  current view (even after matplotlib toolbar zoom/pan).
• set_ylim() / reset_ylim() public API so callers can programmatically
  set the range without rebuilding the figure.
"""
from __future__ import annotations

import logging
from typing import Optional, Any

import matplotlib
import matplotlib.figure
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk  # noqa: PLC2701
import tkinter as tk

from theme import PLOT, PANEL

log = logging.getLogger("ecg")

# Échelle d'espacement locale — miroir des constantes SPACE_* définies dans
# app.py (non importables ici sans dépendance circulaire). Garder les deux
# synchronisées si la grille d'espacement de l'app évolue.
_SPACE_XS = 2
_SPACE_S  = 4
_SPACE_M  = 8

# Apply matplotlib defaults once at import time
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.frameon":   False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.4,
    "axes.titlepad":    6,
    "axes.labelpad":    4,
    "xtick.major.pad":  4,
    "ytick.major.pad":  4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "grid.linewidth":   0.4,
    "lines.antialiased": True,
})

# Tk colours pulled from theme for the scale bar.
#
# These are DERIVED from PANEL/BG/TEXT/BORDER2/MUTED, not the theme names
# themselves -- theme._sync_to_submodules() refreshes PANEL/BG/etc. on this
# module directly after a theme switch, but has no way to know these four
# derived names exist or need recomputing from them. _on_theme_changed()
# below is called by theme.py right after that refresh (if defined), so the
# Y-scale bar picks up the new theme instead of freezing at whichever one
# was active when this module was first imported.
try:
    from theme import BG, BORDER, BORDER2, TEXT, MUTED
    _BAR_BG     = PANEL
    _ENTRY_BG   = BG
    _ENTRY_FG   = TEXT
    _ENTRY_BD   = BORDER2
    _LABEL_FG   = MUTED
except Exception:
    _BAR_BG = "#1E1E2E"; _ENTRY_BG = "#16161F"
    _ENTRY_FG = "#E0E0E0"; _ENTRY_BD = "#333355"; _LABEL_FG = "#888"


def _on_theme_changed() -> None:
    """Recompute the Y-scale-bar colours from the just-refreshed theme names."""
    global _BAR_BG, _ENTRY_BG, _ENTRY_FG, _ENTRY_BD, _LABEL_FG
    _BAR_BG   = PANEL
    _ENTRY_BG = BG
    _ENTRY_FG = TEXT
    _ENTRY_BD = BORDER2
    _LABEL_FG = MUTED


def style_axes(ax) -> None:
    """Apply the active PLOT theme to a matplotlib Axes object."""
    ax.set_facecolor(PLOT["axes"])
    ax.grid(True, color=PLOT["grid"], lw=0.4, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT["border"])
    ax.spines["bottom"].set_color(PLOT["border"])
    ax.tick_params(colors=PLOT["muted"])
    # y=0.05 laissait un bandeau d'air visible au-dessus des pics les plus
    # hauts (ex. seuil de détection R-peak collé au bord du graphe). 0.03
    # garde assez de marge pour ne rien couper (markers, annotations texte)
    # tout en réduisant l'espace mort entre les données et le cadre.
    ax.margins(x=0.01, y=0.03)
    ax.xaxis.label.set_color(PLOT["text"])
    ax.yaxis.label.set_color(PLOT["text"])
    ax.title.set_color(PLOT["text"])


# ════════════════════════════════════════════════════════════
#  CANVAS MANAGER
# ════════════════════════════════════════════════════════════

class CanvasSlot:
    """Embeds a matplotlib Figure that auto-fits its Tk container.

    Design
    ------
    • self.frame is a plain tk.Frame — gives reliable <Configure> events.
    • Optional Y-scale bar (yscale_bar=True): a thin strip above the canvas
      with Y-min / Y-max entry fields and an "Auto" reset button.
      The bar reads back the actual axes limits after every redraw and after
      matplotlib toolbar zoom/pan (via idle_draw event).
    • Three deferred fits (50 / 200 / 600 ms) after construction.
    """

    _DPI      = 100
    # Espacement constrained_layout (en pouces).
    # _CL_PAD  = marge EXTERNE entre la figure et son cadre Tk (titre, labels
    #            d'axes, légendes en overlay). 0.06 est juste assez pour que
    #            rien ne soit coupé, sans laisser de bordure morte autour
    #            du canvas — la majorité des plots sont à un seul axes et
    #            n'ont pas besoin de plus.
    # _CL_SPACE = espace ENTRE sous-graphiques (hspace/wspace) sur les figures
    #            multi-axes (ex. RR+HR empilés, PSD+radar). Reste à 0.12 :
    #            assez pour que les titres de chaque sous-axe ne se touchent
    #            pas, sans créer un fossé visuel entre graphiques liés.
    _CL_PAD   = 0.06
    _CL_SPACE = 0.12

    # ── helpers ────────────────────────────────────────────────────────────

    def _set_cl_pads(self) -> None:
        try:
            self.fig.set_constrained_layout_pads(  # type: ignore[attr-defined]
                w_pad=self._CL_PAD, h_pad=self._CL_PAD,
                hspace=self._CL_SPACE, wspace=self._CL_SPACE,
            )
        except Exception as _e:
            log.debug("set_constrained_layout_pads: %s", _e)
        # Note: per-plot subplot spacing is applied by callers when needed.

    # ── construction ───────────────────────────────────────────────────────

    def __init__(self, parent: Any, width: float = 6, height: float = 4,
                 toolbar: bool = True, yscale_bar: bool = False) -> None:
        self.frame = tk.Frame(parent, bg=PLOT["bg"], bd=0, highlightthickness=0)
        self.frame.pack(fill="both", expand=True)

        self._draw_fn:          Any              = None
        self._resize_id:        Optional[str]    = None
        self._manual_ylim:      Optional[tuple]  = None   # (lo, hi) or None = auto
        self._yscale_bar:       bool             = yscale_bar
        self._yscale_cid:       Optional[int]    = None   # mpl event id

        # ── Optional Y-scale bar (above canvas) ───────────────────────────
        if yscale_bar:
            self._bar = tk.Frame(self.frame, bg=_BAR_BG, height=26,
                                 bd=0, highlightthickness=0)
            self._bar.pack(side="top", fill="x")
            self._bar.pack_propagate(False)

            tk.Label(self._bar, text="Y:", bg=_BAR_BG, fg=_LABEL_FG,
                     font=("DejaVu Sans", 8)).pack(side="left", padx=(_SPACE_M, _SPACE_XS))

            # Min entry
            self._var_ylo = tk.StringVar(value="auto")
            self._ent_ylo = tk.Entry(
                self._bar, textvariable=self._var_ylo,
                width=7, bg=_ENTRY_BG, fg=_ENTRY_FG, relief="flat",
                highlightbackground=_ENTRY_BD, highlightthickness=1,
                insertbackground=_ENTRY_FG, font=("DejaVu Sans", 8))
            self._ent_ylo.pack(side="left", padx=(0, _SPACE_XS), pady=_SPACE_S)

            tk.Label(self._bar, text="–", bg=_BAR_BG,
                     fg=_LABEL_FG, font=("DejaVu Sans", 8)).pack(side="left")

            # Max entry
            self._var_yhi = tk.StringVar(value="auto")
            self._ent_yhi = tk.Entry(
                self._bar, textvariable=self._var_yhi,
                width=7, bg=_ENTRY_BG, fg=_ENTRY_FG, relief="flat",
                highlightbackground=_ENTRY_BD, highlightthickness=1,
                insertbackground=_ENTRY_FG, font=("DejaVu Sans", 8))
            self._ent_yhi.pack(side="left", padx=(_SPACE_XS, _SPACE_M), pady=_SPACE_S)

            # Auto reset button
            auto_btn = tk.Button(
                self._bar, text="Auto", bg=_BAR_BG, fg=_LABEL_FG,
                relief="flat", cursor="hand2",
                font=("DejaVu Sans", 8), activebackground=_ENTRY_BD,
                command=self._on_yscale_auto)
            auto_btn.pack(side="left", padx=(0, _SPACE_S))

            tk.Label(self._bar,
                     text="(edit Y min/max then press Enter  ·  or use toolbar zoom)",
                     bg=_BAR_BG, fg=_LABEL_FG,
                     font=("DejaVu Sans", 7)).pack(side="left", padx=(_SPACE_S, 0))

            # Bind Enter / FocusOut on both entries
            for ent in (self._ent_ylo, self._ent_yhi):
                ent.bind("<Return>",   self._on_yscale_commit)
                ent.bind("<KP_Enter>", self._on_yscale_commit)
                ent.bind("<FocusOut>", self._on_yscale_commit)
        else:
            self._bar = None  # type: ignore[assignment]

        # ── Canvas area ────────────────────────────────────────────────────
        self._cv_frame = tk.Frame(self.frame, bg=PLOT["bg"], bd=0, highlightthickness=0)
        self._cv_frame.pack(side="top", fill="both", expand=True)

        self.fig = matplotlib.figure.Figure(
            figsize=(width, height), dpi=self._DPI,
            facecolor=PLOT["bg"])
        self._set_cl_pads()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self._cv_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Matplotlib toolbar (below canvas) ──────────────────────────────
        if toolbar:
            tb_frame = tk.Frame(self.frame, bg=PANEL, bd=0, highlightthickness=0)
            tb_frame.pack(side="bottom", fill="x")
            tb = NavigationToolbar2Tk(self.canvas, tb_frame, pack_toolbar=False)
            tb.configure(background=PANEL)
            tb.pack(fill="x")
            # After any toolbar zoom/pan the limits change; read them back
            # into the Y-scale bar so the fields stay in sync.
            if yscale_bar:
                self._yscale_cid = self.canvas.mpl_connect(
                    "draw_event", self._on_mpl_draw)

        self.frame.bind("<Configure>", self._on_configure)

        for delay in (50, 200, 600):
            self.frame.after(delay, self._fit)

        self._show_placeholder()

    # ── Y-scale bar callbacks ──────────────────────────────────────────────

    def _on_yscale_commit(self, _event: Any = None) -> None:
        """Parse Y min/max entries and apply them to all axes in the figure."""
        if not self._yscale_bar:
            return
        try:
            lo_str = self._var_ylo.get().strip().lower()
            hi_str = self._var_yhi.get().strip().lower()
            if lo_str in ("", "auto") or hi_str in ("", "auto"):
                self._on_yscale_auto()
                return
            lo = float(lo_str)
            hi = float(hi_str)
            if lo >= hi:
                return
            self._manual_ylim = (lo, hi)
            self._apply_ylim()
        except (ValueError, AttributeError):
            pass  # non-numeric — ignore silently

    def _on_yscale_auto(self) -> None:
        """Reset y-limits to matplotlib auto-scale on all axes."""
        self._manual_ylim = None
        for ax in self.fig.axes:
            ax.set_ylim(auto=True)
            ax.autoscale(enable=True, axis="y")
        self.canvas.draw_idle()
        self._readback_ylim()

    def _apply_ylim(self) -> None:
        """Apply self._manual_ylim to every axes that has a y-axis."""
        if self._manual_ylim is None:
            return
        lo, hi = self._manual_ylim
        axs = self.fig.axes
        if not axs:
            return
        # Apply only to the primary axes (first one); shared axes follow automatically
        axs[0].set_ylim(lo, hi)
        self.canvas.draw_idle()

    def _readback_ylim(self) -> None:
        """Update Y-scale bar fields from the current axes limits."""
        if not self._yscale_bar:
            return
        axs = self.fig.axes
        if not axs:
            return
        try:
            lo, hi = axs[0].get_ylim()
            self._var_ylo.set(f"{lo:.3g}")
            self._var_yhi.set(f"{hi:.3g}")
        except Exception:
            pass

    def _on_mpl_draw(self, _event: Any = None) -> None:
        """Called after any matplotlib draw (including toolbar zoom/pan)."""
        if self._manual_ylim is None:
            self._readback_ylim()

    # ── Public API ─────────────────────────────────────────────────────────

    def set_ylim(self, lo: float, hi: float) -> None:
        """Programmatically set the y range and update the bar fields."""
        self._manual_ylim = (lo, hi)
        self._apply_ylim()
        if self._yscale_bar:
            self._var_ylo.set(f"{lo:.3g}")
            self._var_yhi.set(f"{hi:.3g}")

    def reset_ylim(self) -> None:
        """Reset to auto-scale and clear the bar fields."""
        self._on_yscale_auto()

    # ── Resize logic ───────────────────────────────────────────────────────

    def _on_configure(self, _event: Any) -> None:
        if self._resize_id is not None:
            try:
                self.frame.after_cancel(self._resize_id)
            except Exception:
                pass
        self._resize_id = self.frame.after(80, self._fit)

    def _fit(self) -> None:
        self._resize_id = None
        try:
            px_w = self._cv_frame.winfo_width()
            px_h = self._cv_frame.winfo_height()
        except Exception as e:
            log.debug("winfo size not ready: %s", e)
            return
        if px_w < 80 or px_h < 50:
            return
        new_w = px_w / self._DPI
        new_h = px_h / self._DPI
        cur_w, cur_h = self.fig.get_size_inches()
        if abs(new_w - cur_w) * self._DPI < 3 and abs(new_h - cur_h) * self._DPI < 3:
            return
        self.fig.set_size_inches(new_w, new_h, forward=True)
        if self._draw_fn is not None:
            self._redraw()
        else:
            self.canvas.draw_idle()

    # ── Drawing ────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self.fig.clear()
        self.fig.patch.set_facecolor(PLOT["bg"])
        self._set_cl_pads()
        if self._draw_fn is None:
            self.canvas.draw_idle()
            return
        try:
            self._draw_fn(self.fig)
        except Exception as exc:
            log.exception("CanvasSlot draw error")
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(PLOT["axes"])
            ax.text(0.5, 0.5, f"Draw error:\n{exc}",
                    ha="center", va="center", color="#D32F2F",
                    transform=ax.transAxes, wrap=True)
        # Re-apply manual ylim if set, then read back the actual limits
        if self._manual_ylim is not None:
            self._apply_ylim()
        else:
            # Small delay — axes may not have final limits yet right after draw
            self.frame.after(50, self._readback_ylim)
        self.canvas.draw_idle()

    def _show_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(PLOT["axes"])
        ax.text(0.5, 0.5, "Run analysis to display",
                ha="center", va="center", color=PLOT["muted"],
                transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(PLOT["border"])
        self.canvas.draw_idle()

    def update(self, draw_fn: Any) -> None:
        """Store *draw_fn* and immediately render it.

        draw_fn(fig) receives the matplotlib Figure and must populate it.
        Replayed automatically on resize.  Must be called from the main thread.
        """
        self._draw_fn = draw_fn
        self._redraw()