"""Entry point — run with:  python -m ecg"""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox


def main() -> None:
    from theme import NK_AVAILABLE
    if not NK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing dependency",
            "Install NeuroKit2:\n  pip install neurokit2",
        )
        root.destroy()
        return
    from app import ECGApp
    ECGApp().mainloop()


if __name__ == "__main__":
    main()
