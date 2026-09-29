"""app.gui package initialization."""
from app.gui.header import Header
from app.gui.status_bar import StatusBar
from app.gui.image_view import ImageView
from app.gui.image_area import ImageArea
from app.gui.filter_panel import FilterPanel
from app.gui.kernel_editor import KernelEditorWindow
from app.gui.kernel_info import KernelInfoPanel
from app.gui.filter_comparison import FilterComparisonPanel
from app.gui.history_panel import HistoryPanel
from app.gui.main_window import MainWindow

__all__ = [
    "Header", 
    "StatusBar", 
    "ImageView", 
    "ImageArea",
    "FilterPanel", 
    "KernelEditorWindow", 
    "KernelInfoPanel",
    "FilterComparisonPanel",
    "HistoryPanel",
    "MainWindow"
]


