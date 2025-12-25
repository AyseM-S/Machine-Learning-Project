"""
Confusion Matrix Display Module

This module handles the visualization of confusion matrices with detailed
statistics, class performance analysis, and visual annotations.

Classes:
    ConfusionMatrixDisplay: Handles confusion matrix visualization and statistics
"""

import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ConfusionMatrixDisplay:
    """
    Handles the visualization and analysis of confusion matrices.
    
    This class creates comprehensive confusion matrix visualizations including
    heatmaps, class-wise accuracy bars, and detailed statistics panels.
    """
    
    def __init__(self, ui_utils):
        """
        Initialize the ConfusionMatrixDisplay component.
        
        Args:
            ui_utils: Instance of UIUtils for accessing utility functions
        """
        self.ui_utils = ui_utils
        self.current_canvas = None
        self.current_figure = None
    
    def display_confusion_matrix(self, parent, cm_data, class_names=None, colormap=None):
        """
        Displays a comprehensive confusion matrix with statistics.
        
        Creates a two-panel layout: left side shows the confusion matrix
        heatmap with class performance bars, right side shows detailed statistics.
        
        Args:
            parent: Parent Tkinter widget to display matrix in
            cm_data (numpy.ndarray): 2D array representing the confusion matrix
            class_names (list, optional): List of class names. If None, generates
                default names like "Class 1", "Class 2", etc.
            colormap: Matplotlib colormap for the heatmap visualization
        """
        # Main container frame
        cm_container = tk.Frame(parent, bg=self.ui_utils.COLOR_BACKGROUND)
        cm_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header section
        header = self.ui_utils.create_header_frame(
            cm_container,
            "🎯 Confusion Matrix Analysis",
            self.ui_utils.COLOR_HEADER_RED
        )
        
        # Main content frame for matrix and statistics
        main_content = tk.Frame(cm_container, bg=self.ui_utils.COLOR_BACKGROUND)
        main_content.pack(fill="both", expand=True)
        
        # Left frame: Matrix visualization
        left_frame = tk.Frame(
            main_content,
            bg="white",
            relief="solid",
            borderwidth=1
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=5)
        
        self.create_matrix_visualization(left_frame, cm_data, class_names, colormap)
        
        # Right frame: Statistics panel
        right_frame = tk.Frame(
            main_content,
            bg="white",
            relief="solid",
            borderwidth=1,
            width=250
        )
        right_frame.pack(side="right", fill="y", padx=(5, 0), pady=5)
        right_frame.pack_propagate(False)
        
        self.create_matrix_statistics(right_frame, cm_data)
    
    def create_matrix_visualization(self, parent, cm_data, class_names, colormap):
        """
        Creates the confusion matrix heatmap with class performance visualization.
        
        Args:
            parent: Parent Tkinter widget
            cm_data (numpy.ndarray): 2D confusion matrix array
            class_names (list): List of class names for labeling
            colormap: Matplotlib colormap for heatmap coloring
        """
        # Generate default class names if not provided
        if class_names is None:
            class_names = [f"Class {i+1}" for i in range(len(cm_data))]
        
        # Create figure with two subplots: matrix and class accuracy
        self.current_figure, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(10, 4.5),
            gridspec_kw={'width_ratios': [3, 1]}
        )
        
        # Subplot 1: Main confusion matrix heatmap
        sns.heatmap(
            cm_data,
            annot=True,
            fmt='d',
            cmap=colormap if colormap else 'Blues',
            ax=ax1,
            cbar=False,
            linewidths=1,
            linecolor='white',
            annot_kws={"size": 11, "weight": "bold"}
        )
        
        # Configure axis labels and titles
        ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax1.set_yticklabels(class_names, rotation=0, fontsize=10)
        ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_ylabel('True Label', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
        
        # Add visual annotations: green border for correct predictions (diagonal)
        # and red dashed border for high-error cells
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                if i == j:  # Diagonal elements (correct predictions)
                    ax1.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='green',
                        linewidth=2
                    ))
                elif cm_data[i, j] > cm_data.mean():  # High error cells
                    ax1.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor='red',
                        linewidth=1,
                        linestyle='--'
                    ))
        
        # Subplot 2: Class-wise accuracy bar chart
        accuracy_per_class = np.diag(cm_data) / cm_data.sum(axis=1)
        
        # Color coding based on accuracy level
        colors = [
            self.ui_utils.COLOR_EXCELLENT if acc > 0.7
            else self.ui_utils.COLOR_AVERAGE if acc > 0.5
            else self.ui_utils.COLOR_POOR
            for acc in accuracy_per_class
        ]
        
        y_pos = np.arange(len(class_names))
        ax2.barh(y_pos, accuracy_per_class, color=colors, height=0.6)
        ax2.set_xlim([0, 1])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_names, fontsize=9)
        ax2.set_xlabel('Accuracy per Class', fontsize=10)
        ax2.set_title('Class Performance', fontsize=11, pad=15)
        
        # Add value labels on bars
        for i, v in enumerate(accuracy_per_class):
            ax2.text(
                v + 0.02,
                i,
                f'{v:.1%}',
                va='center',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        # Embed matplotlib figure in Tkinter
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, master=parent)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )
    
    def create_matrix_statistics(self, parent, cm_data):
        """
        Creates a statistics panel showing confusion matrix analysis.
        
        Displays overall accuracy, total samples, correct predictions,
        misclassifications, most confused class pairs, and error rate.
        
        Args:
            parent: Parent Tkinter widget
            cm_data (numpy.ndarray): 2D confusion matrix array
        """
        # Statistics header
        tk.Label(
            parent,
            text="📈 Matrix Statistics",
            font=('Segoe UI', 10, 'bold'),
            bg="#2196F3",
            fg="white"
        ).pack(fill="x", pady=(10, 5))
        
        # Calculate statistics
        total = np.sum(cm_data)
        correct = np.trace(cm_data)
        accuracy = correct / total if total > 0 else 0
        
        # Find most confused class pair (highest off-diagonal value)
        error_matrix = cm_data.copy().astype(float)
        np.fill_diagonal(error_matrix, 0)
        most_confused = np.unravel_index(
            np.argmax(error_matrix),
            error_matrix.shape
        )
        
        # Statistics to display
        stats = [
            ("Overall Accuracy", f"{accuracy:.2%}"),
            ("Total Samples", f"{int(total):,}"),
            ("Correct Predictions", f"{int(correct):,}"),
            ("Misclassifications", f"{int(total-correct):,}"),
            ("Most Confused", f"Class {most_confused[0]+1}→{most_confused[1]+1}"),
            ("Error Rate", f"{(total-correct)/total:.2%}")
        ]
        
        # Display each statistic
        for label, value in stats:
            stat_frame = tk.Frame(parent, bg="white")
            stat_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(
                stat_frame,
                text=label,
                font=('Segoe UI', 9),
                fg="#555",
                bg="white",
                width=20,
                anchor="w"
            ).pack(side="left")
            
            tk.Label(
                stat_frame,
                text=value,
                font=('Segoe UI', 9, 'bold'),
                fg="#2196F3",
                bg="white"
            ).pack(side="right")
    
    def cleanup(self):
        """
        Cleans up matplotlib resources to prevent memory leaks.
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None

