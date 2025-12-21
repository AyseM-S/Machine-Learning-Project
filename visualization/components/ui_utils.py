"""
UI Utilities Module

This module provides utility functions for UI styling, color management,
and common UI component creation across the visualization package.

Classes:
    UIUtils: Static utility class for UI-related helper functions
"""

import tkinter as tk
from matplotlib.colors import LinearSegmentedColormap


class UIUtils:
    """
    Utility class providing static methods for UI component creation and styling.
    
    This class centralizes common UI operations such as color management,
    status determination, and reusable component creation to ensure consistency
    across the visualization package.
    """
    
    # Color scheme constants for consistent styling
    COLOR_EXCELLENT = "#4CAF50"      # Green for excellent performance
    COLOR_GOOD = "#8BC34A"           # Light green for good performance
    COLOR_AVERAGE = "#FFC107"        # Yellow for average performance
    COLOR_FAIR = "#FF9800"           # Orange for fair performance
    COLOR_POOR = "#F44336"           # Red for poor performance
    
    COLOR_BACKGROUND = "#f5f7fa"     # Light gray background
    COLOR_HEADER_BLUE = "#4a6fa5"    # Blue header color
    COLOR_HEADER_RED = "#FF5722"     # Red header color
    COLOR_HEADER_PURPLE = "#9C27B0"  # Purple header color
    
    @staticmethod
    def setup_custom_colormaps():
        """
        Creates and returns custom colormaps for visualization.
        
        Returns:
            tuple: A tuple containing (blues_colormap, heat_colormap)
                - blues_colormap: LinearSegmentedColormap for blue tones
                - heat_colormap: LinearSegmentedColormap for red/orange tones
        """
        cm_blues = LinearSegmentedColormap.from_list(
            'custom_blues', 
            ['#f0f8ff', '#87ceeb', '#1e90ff', '#000080']
        )
        cm_heat = LinearSegmentedColormap.from_list(
            'custom_heat',
            ['#ffebee', '#ff8a80', '#ff5252', '#d50000']
        )
        return cm_blues, cm_heat
    
    @staticmethod
    def get_score_color(score):
        """
        Determines the appropriate color based on a performance score.
        
        Args:
            score (float): Performance score between 0.0 and 1.0
            
        Returns:
            str: Hex color code representing the score level
                - "#4CAF50" for score >= 0.9 (excellent)
                - "#8BC34A" for score >= 0.7 (good)
                - "#FFC107" for score >= 0.5 (average)
                - "#FF9800" for score >= 0.3 (fair)
                - "#F44336" for score < 0.3 (poor)
        """
        if score >= 0.9:
            return UIUtils.COLOR_EXCELLENT
        elif score >= 0.7:
            return UIUtils.COLOR_GOOD
        elif score >= 0.5:
            return UIUtils.COLOR_AVERAGE
        elif score >= 0.3:
            return UIUtils.COLOR_FAIR
        else:
            return UIUtils.COLOR_POOR
    
    @staticmethod
    def get_performance_status(score):
        """
        Determines the performance status label and color based on score.
        
        Args:
            score (float): Average performance score between 0.0 and 1.0
            
        Returns:
            tuple: A tuple containing (status_text, status_color)
                - status_text: String label describing performance level
                - status_color: Hex color code for the status badge
        """
        if score >= 0.85:
            return "Excellent", UIUtils.COLOR_EXCELLENT
        elif score >= 0.70:
            return "Good", UIUtils.COLOR_GOOD
        elif score >= 0.55:
            return "Average", UIUtils.COLOR_AVERAGE
        elif score >= 0.40:
            return "Fair", UIUtils.COLOR_FAIR
        else:
            return "Poor", UIUtils.COLOR_POOR
    
    @staticmethod
    def create_header_frame(parent, title, bg_color, height=40):
        """
        Creates a standardized header frame with title.
        
        Args:
            parent: Parent Tkinter widget
            title (str): Header title text
            bg_color (str): Background color for the header
            height (int): Height of the header frame in pixels
            
        Returns:
            tk.Frame: Configured header frame widget
        """
        header = tk.Frame(parent, bg=bg_color, height=height)
        header.pack(fill="x", pady=(0, 10))
        header.pack_propagate(False)
        
        tk.Label(
            header, 
            text=title,
            font=('Segoe UI', 12, 'bold'),
            fg="white",
            bg=bg_color
        ).pack(pady=10)
        
        return header
    
    @staticmethod
    def get_model_icon(model_name):
        """
        Returns an appropriate emoji icon based on model name.
        
        Args:
            model_name (str): Name of the machine learning model
            
        Returns:
            str: Emoji icon representing the model type
        """
        if "MLP" in model_name or "Multilayer" in model_name:
            return "🤖"
        elif "Perceptron" in model_name:
            return "🧠"
        elif "Decision" in model_name or "Tree" in model_name:
            return "🌳"
        else:
            return "📊"

