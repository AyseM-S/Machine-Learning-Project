"""
Charts Display Module

This module handles various chart visualizations including radar charts,
bar charts, and ranking displays for model comparison.

Classes:
    ChartDisplay: Handles creation of various chart types
"""

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ChartDisplay:
    """
    Handles creation of various chart visualizations for model comparison.
    
    This class provides methods for creating radar charts, bar charts,
    and ranking displays to visualize model performance from different perspectives.
    """
    
    def __init__(self, ui_utils):
        """
        Initialize the ChartDisplay component.
        
        Args:
            ui_utils: Instance of UIUtils for accessing utility functions
        """
        self.ui_utils = ui_utils
        self.current_figure = None
    
    def create_radar_chart(self, parent, results_dict):
        """
        Creates a radar (spider) chart comparing models across multiple metrics.
        
        Each model is represented as a polygon on a radar chart, making it
        easy to compare strengths and weaknesses across different metrics.
        
        Args:
            parent: Parent Tkinter widget to display chart in
            results_dict (dict): Dictionary mapping model names to their metrics
        """
        self.current_figure = plt.figure(figsize=(6, 5))
        
        # Metric names for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        num_vars = len(metrics)
        
        # Calculate angles for each metric (evenly spaced around circle)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Create polar subplot
        ax = self.current_figure.add_subplot(111, polar=True)
        
        # Color palette for different models
        colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
        
        # Plot each model
        for (model_name, model_metrics), color in zip(results_dict.items(), colors):
            # Extract metric values
            values = [
                model_metrics.get('accuracy', 0),
                model_metrics.get('precision', 0),
                model_metrics.get('recall', 0),
                model_metrics.get('f1_score', 0)
            ]
            values += values[:1]  # Close the polygon
            
            # Plot line and fill area
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Configure axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.title('Model Comparison - Radar Chart', fontsize=12, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(self.current_figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_bar_chart(self, parent, results_dict):
        """
        Creates a 2x2 grid of bar charts, one for each metric.
        
        Each subplot shows all models' performance for a specific metric,
        making it easy to compare models across different performance indicators.
        
        Args:
            parent: Parent Tkinter widget to display chart in
            results_dict (dict): Dictionary mapping model names to their metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        # Metric configurations: (key, display_name, subplot_index)
        metric_configs = [
            ('accuracy', 'Accuracy', 0),
            ('precision', 'Precision', 1),
            ('recall', 'Recall', 2),
            ('f1_score', 'F1-Score', 3)
        ]
        
        models = list(results_dict.keys())
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        
        # Create bar chart for each metric
        for metric_key, metric_name, idx in metric_configs:
            ax = axes[idx]
            
            # Extract values for this metric across all models
            values = [results_dict[model].get(metric_key, 0) for model in models]
            bars = ax.bar(
                models,
                values,
                color=colors[:len(models)],
                edgecolor='black',
                linewidth=1
            )
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for readability
            ax.set_xticklabels(models, rotation=45, ha='right')
        
        plt.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_ranking_display(self, parent, results_dict):
        """
        Creates a visual ranking display showing top-performing models.
        
        Displays models in ranked order with visual indicators (medals for top 3)
        and progress bars showing relative performance.
        
        Args:
            parent: Parent Tkinter widget to display ranking in
            results_dict (dict): Dictionary mapping model names to their metrics
        """
        # Calculate rankings
        rankings = []
        for model, metrics in results_dict.items():
            avg_score = np.mean([
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ])
            rankings.append((model, avg_score))
        
        # Sort by average score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Main ranking frame
        ranking_frame = tk.Frame(parent, bg="white")
        ranking_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        tk.Label(
            ranking_frame,
            text="🏆 Model Ranking",
            font=('Segoe UI', 14, 'bold'),
            fg="#FF9800",
            bg="white"
        ).pack(pady=(0, 20))
        
        # Medal colors and icons for top performers
        medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#5C6BC0', '#9C27B0']
        medal_icons = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        
        # Create ranking cards (top 5)
        for i, (model, score) in enumerate(rankings):
            if i >= 5:
                break
            
            # Card with special styling for top 3
            card = tk.Frame(
                ranking_frame,
                bg=medal_colors[i] if i < 3 else "#f5f5f5",
                relief="ridge",
                borderwidth=2
            )
            card.pack(fill="x", pady=8)
            
            # Left side: Rank and model name
            left_frame = tk.Frame(card, bg=card["bg"])
            left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)
            
            tk.Label(
                left_frame,
                text=f"{medal_icons[i]} #{i+1}: {model}",
                font=('Segoe UI', 11, 'bold'),
                bg=card["bg"]
            ).pack(anchor="w")
            
            # Right side: Score
            right_frame = tk.Frame(card, bg=card["bg"])
            right_frame.pack(side="right", padx=15, pady=10)
            
            tk.Label(
                right_frame,
                text=f"Score: {score:.4f}",
                font=('Segoe UI', 10, 'bold'),
                fg="#2196F3",
                bg=card["bg"]
            ).pack()
            
            # Progress bar
            progress_frame = tk.Frame(card, bg="#e0e0e0", height=8)
            progress_frame.pack(fill="x", padx=15, pady=(0, 10))
            
            fill_width = int(score * 100)
            fill = tk.Frame(
                progress_frame,
                bg=medal_colors[i] if i < 3 else "#5C6BC0",
                width=fill_width,
                height=8
            )
            fill.pack(side="left")
    
    def cleanup(self):
        """
        Cleans up matplotlib resources to prevent memory leaks.
        """
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None

