"""
Metrics Display Module

This module handles the display of model performance metrics in a user-friendly
card-based format with visual indicators and percentage bars.

Classes:
    MetricsDisplay: Handles creation and display of metric cards and tables
"""

import tkinter as tk


class MetricsDisplay:
    """
    Handles the visualization of model performance metrics.
    
    This class creates and manages metric display components including
    metric cards with visual indicators, percentage bars, and organized
    metric tables for easy comparison.
    """
    
    def __init__(self, ui_utils):
        """
        Initialize the MetricsDisplay component.
        
        Args:
            ui_utils: Instance of UIUtils for accessing utility functions
        """
        self.ui_utils = ui_utils
    
    def display_metrics_table(self, parent, metrics, model_name):
        """
        Displays a comprehensive metrics table with cards for each metric.
        
        Creates a visually appealing grid layout showing accuracy, precision,
        recall, and F1-score with color-coded cards and progress bars.
        
        Args:
            parent: Parent Tkinter widget to display metrics in
            metrics (dict): Dictionary containing metric values
                Expected keys: 'accuracy', 'precision', 'recall', 'f1_score'
            model_name (str): Name of the model being displayed
        """
        # Main container frame with background color
        metrics_container = tk.Frame(parent, bg=self.ui_utils.COLOR_BACKGROUND)
        metrics_container.pack(fill="x", padx=10, pady=10)
        
        # Header section
        header_frame = self.ui_utils.create_header_frame(
            metrics_container,
            f"📊 {model_name} - Performance Metrics",
            self.ui_utils.COLOR_HEADER_BLUE
        )
        
        # Metrics grid container
        metrics_grid = tk.Frame(metrics_container, bg=self.ui_utils.COLOR_BACKGROUND)
        metrics_grid.pack(fill="x", padx=20, pady=10)
        
        # Metric configuration: (key, label, color, icon)
        metric_configs = [
            ("accuracy", "Accuracy", "#4CAF50", "🧠"),
            ("precision", "Precision", "#2196F3", "🎯"),
            ("recall", "Recall", "#FF9800", "🔍"),
            ("f1_score", "F1-Score", "#9C27B0", "⚖️")
        ]
        
        # Create metric cards in a 2x2 grid
        for i, (key, label, color, icon) in enumerate(metric_configs):
            if key in metrics:
                card = self.create_metric_card(
                    metrics_grid, 
                    label, 
                    metrics[key], 
                    color, 
                    icon
                )
                card.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
        
        # Configure grid columns for equal distribution
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
    
    def create_metric_card(self, parent, label, value, color, icon):
        """
        Creates an individual metric card with value and progress bar.
        
        Args:
            parent: Parent Tkinter widget
            label (str): Metric label (e.g., "Accuracy")
            value (float): Metric value between 0.0 and 1.0
            color (str): Color code for the card header
            icon (str): Emoji icon for the metric
            
        Returns:
            tk.Frame: Configured metric card widget
        """
        # Main card frame with border
        card = tk.Frame(parent, bg="white", relief="ridge", borderwidth=1)
        
        # Card header with colored background
        title_frame = tk.Frame(card, bg=color, height=30)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text=f"{icon} {label}",
            font=('Segoe UI', 9, 'bold'),
            fg="white",
            bg=color
        ).pack(pady=5)
        
        # Value display frame
        value_frame = tk.Frame(card, bg="white", height=60)
        value_frame.pack(fill="both", expand=True)
        
        tk.Label(
            value_frame,
            text=f"{value:.4f}",
            font=('Segoe UI', 18, 'bold'),
            fg="#333",
            bg="white"
        ).pack(expand=True)
        
        # Add percentage bar visualization
        self.create_percentage_bar(card, value)
        
        return card
    
    def create_percentage_bar(self, parent, value):
        """
        Creates a visual percentage bar indicator for the metric value.
        
        Args:
            parent: Parent Tkinter widget (metric card)
            value (float): Metric value between 0.0 and 1.0
        """
        # Background bar frame
        bar_frame = tk.Frame(parent, bg="#e0e0e0", height=8)
        bar_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Calculate fill width (0-100 pixels)
        fill_width = int(value * 100)
        
        # Determine fill color based on value
        fill_color = self.ui_utils.COLOR_EXCELLENT if value > 0.7 else self.ui_utils.COLOR_AVERAGE
        
        # Create fill bar
        fill = tk.Frame(
            bar_frame,
            bg=fill_color,
            width=fill_width,
            height=8
        )
        fill.pack(side="left")
        
        # Percentage label
        tk.Label(
            parent,
            text=f"{value*100:.1f}%",
            font=('Segoe UI', 8),
            fg="#666",
            bg="white"
        ).pack()

