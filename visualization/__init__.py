"""
Visualization Package for Machine Learning Model Results

This package provides a modular system for visualizing machine learning model
performance metrics, confusion matrices, and model comparisons in a Tkinter GUI.

Main Components:
    - ResultVisualizer: Main coordinator class for all visualization components
    - Components: Modular visualization components for specific features

Usage:
    from visualization import ResultVisualizer
    
    visualizer = ResultVisualizer(results_frame)
    visualizer.display_metrics_table(metrics, model_name)
"""

from .result_visualizer import ResultVisualizer

__all__ = ['ResultVisualizer']

