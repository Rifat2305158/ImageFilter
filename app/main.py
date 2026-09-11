"""
Main application entry point.
"""

import sys
import os

# Ensure project root is in Python path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.gui.main_window import MainWindow


def main():
    """Launch Image Blur, Sharpening & Edge Detection Studio GUI."""
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
