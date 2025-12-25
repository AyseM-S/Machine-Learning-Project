"""
Result Visualizer Main Module

This module provides the main ResultVisualizer class that coordinates all
visualization components and provides a unified interface for displaying
machine learning model results.

Classes:
    ResultVisualizer: Main coordinator class for visualization components
"""

import tkinter as tk
from .components.ui_utils import UIUtils
from .components.metrics_display import MetricsDisplay
from .components.confusion_matrix import ConfusionMatrixDisplay
from .components.model_comparison import ModelComparisonDisplay
from .components.charts import ChartDisplay


class ResultVisualizer:
    """
    Main coordinator class for all visualization components.
    
    This class provides a unified interface for displaying model performance
    metrics, confusion matrices, model comparisons, and various charts.
    It manages the lifecycle of visualization components and coordinates
    their interactions.
    
    Attributes:
        results_frame: Tkinter frame where visualizations are displayed
        ui_utils: Utility class for UI operations
        metrics_display: Component for displaying metrics
        confusion_matrix: Component for confusion matrix visualization
        model_comparison: Component for model comparison
        charts: Component for chart visualizations
        current_canvas: Currently active matplotlib canvas (if any)
        current_figure: Currently active matplotlib figure (if any)
    """
    
    def __init__(self, results_frame):
        """
        Initialize the ResultVisualizer with a results frame.
        
        Args:
            results_frame: Tkinter Frame widget where visualizations will be displayed
        """
        self.results_frame = results_frame
        self.current_canvas = None
        self.current_figure = None
        
        # Initialize utility and component classes
        self.ui_utils = UIUtils()
        self.metrics_display = MetricsDisplay(self.ui_utils)
        self.confusion_matrix = ConfusionMatrixDisplay(self.ui_utils)
        self.charts = ChartDisplay(self.ui_utils)
        # Pass chart_display to model_comparison so it can create chart tabs
        self.model_comparison = ModelComparisonDisplay(self.ui_utils, self.charts)
        
        # Setup custom colormaps
        self.cm_blues, self.cm_heat = self.ui_utils.setup_custom_colormaps()
    
    def clear_results(self):
        """
        Clears all widgets and visualizations from the results frame.
        
        This method destroys all child widgets and cleans up matplotlib
        resources to prevent memory leaks. Should be called before displaying
        new visualizations.
        """
        # Destroy all child widgets
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Clean up matplotlib resources
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        
        if self.current_figure:
            import matplotlib.pyplot as plt
            plt.close(self.current_figure)
            self.current_figure = None
        
        # Clean up component resources
        self.confusion_matrix.cleanup()
        self.charts.cleanup()
    
    def display_metrics_table(self, metrics, model_name):
        """
        Displays a metrics table for a single model.
        
        Args:
            metrics (dict): Dictionary containing metric values
                Expected keys: 'accuracy', 'precision', 'recall', 'f1_score'
            model_name (str): Name of the model being displayed
        """
        self.clear_results()
        self.metrics_display.display_metrics_table(
            self.results_frame,
            metrics,
            model_name
        )
    
    def display_confusion_matrix(self, cm_data, class_names=None):
        """
        Displays a confusion matrix with statistics and class performance.
        
        Args:
            cm_data (numpy.ndarray): 2D array representing the confusion matrix
            class_names (list, optional): List of class names. If None, generates
                default names like "Class 1", "Class 2", etc.
        """
        self.clear_results()
        self.confusion_matrix.display_confusion_matrix(
            self.results_frame,
            cm_data,
            class_names,
            self.cm_blues
        )
        # Store references for cleanup
        self.current_canvas = self.confusion_matrix.current_canvas
        self.current_figure = self.confusion_matrix.current_figure
    
    def display_model_comparison(self, results_dict):
        """
        Displays a comprehensive model comparison dashboard.
        
        Creates a tabbed interface with comparison tables, analysis, and
        recommendations based on model performance.
        
        Args:
            results_dict (dict): Dictionary mapping model names to their metrics
                Expected format: {model_name: {'accuracy': float, 'precision': float, ...}}
        """
        self.clear_results()
        self.model_comparison.display_model_comparison(
            self.results_frame,
            results_dict
        )
    
    def display_training_message(self, model_name):
        """
        Displays an animated loading message during model training.
        
        Args:
            model_name (str): Name of the model being trained
            
        Returns:
            tk.Frame: The message frame widget (for potential cleanup)
        """
        self.clear_results()
        
        message_frame = tk.Frame(self.results_frame, bg="#e3f2fd")
        message_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Loading animation container
        loading_frame = tk.Frame(message_frame, bg="#e3f2fd")
        loading_frame.pack(expand=True)
        
        # Gear icon
        tk.Label(
            loading_frame,
            text="⚙️",
            font=('Arial', 40),
            fg="#2196F3",
            bg="#e3f2fd"
        ).pack()
        
        # Training message
        tk.Label(
            loading_frame,
            text=f"Training {model_name}",
            font=('Segoe UI', 16, 'bold'),
            fg="#1976D2",
            bg="#e3f2fd"
        ).pack(pady=20)
        
        # Animated dots container
        dots_frame = tk.Frame(loading_frame, bg="#e3f2fd")
        dots_frame.pack()
        
        # Create animated dots
        self.dots = [
            tk.Label(
                dots_frame,
                text="●",
                font=('Arial', 20),
                fg=c,
                bg="#e3f2fd"
            )
            for c in ['#bbdefb', '#64b5f6', '#1976D2']
        ]
        
        for dot in self.dots:
            dot.pack(side="left", padx=2)
        
        # Start animation
        self._animate_dots(0)
        
        return message_frame
    
    def _animate_dots(self, index):
        """
        Animates the loading dots in a cycling pattern.
        
        Args:
            index (int): Current dot index to highlight
        """
        if hasattr(self, 'dots') and self.dots:
            for i, dot in enumerate(self.dots):
                color = '#1976D2' if i == index else '#bbdefb'
                dot.config(fg=color)
            
            next_index = (index + 1) % len(self.dots)
            self.results_frame.after(300, lambda: self._animate_dots(next_index))
    
    def display_error_message(self, error_text):
        """
        Displays an error message in the results frame.
        
        Args:
            error_text (str): Error message to display
        """
        self.clear_results()
        
        error_frame = tk.Frame(self.results_frame, bg="#ffebee")
        error_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Error icon
        tk.Label(
            error_frame,
            text="❌",
            font=('Arial', 40),
            fg="#d32f2f",
            bg="#ffebee"
        ).pack(pady=(0, 20))
        
        # Error title
        tk.Label(
            error_frame,
            text="Training Error",
            font=('Segoe UI', 18, 'bold'),
            fg="#d32f2f",
            bg="#ffebee"
        ).pack(pady=(0, 10))
        
        # Error message
        tk.Label(
            error_frame,
            text=error_text,
            font=('Segoe UI', 11),
            fg="#5d4037",
            bg="#ffebee",
            wraplength=400,
            justify="center"
        ).pack(pady=10)

