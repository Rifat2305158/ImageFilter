"""Entry point for the application."""
import sys
import os

# Add the parent directory to sys.path if needed, but running from root works too
import src.core.convolution
import src.filters.blur
import src.filters.sharpening
import src.filters.edge
import src.io.image_io
import src.analysis.statistics
import src.gui.app
import src.gui.components

if __name__ == "__main__":
    print("All modules imported successfully.")
