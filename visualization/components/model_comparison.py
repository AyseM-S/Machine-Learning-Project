"""
Model Comparison Display Module

This module handles comprehensive model comparison visualizations including
detailed comparison tables, performance analysis, and recommendations.

Classes:
    ModelComparisonDisplay: Handles model comparison tables and analysis
"""

import tkinter as tk
from tkinter import ttk
import numpy as np


class ModelComparisonDisplay:
    """
    Handles comprehensive model comparison visualizations.
    
    This class creates detailed comparison tables with scrollable content,
    performance analysis sections, key metrics displays, and actionable
    recommendations based on model performance.
    """
    
    def __init__(self, ui_utils, chart_display):
        """
        Initialize the ModelComparisonDisplay component.
        
        Args:
            ui_utils: Instance of UIUtils for accessing utility functions
            chart_display: Instance of ChartDisplay for creating charts
        """
        self.ui_utils = ui_utils
        self.chart_display = chart_display
    
    def display_model_comparison(self, parent, results_dict):
        """
        Creates a comprehensive model comparison dashboard with multiple tabs.
        
        Args:
            parent: Parent Tkinter widget to display comparison in
            results_dict (dict): Dictionary mapping model names to their metrics
                Expected format: {model_name: {'accuracy': float, 'precision': float, ...}}
        """
        # Main container
        comparison_container = tk.Frame(parent, bg=self.ui_utils.COLOR_BACKGROUND)
        comparison_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header section
        header = self.ui_utils.create_header_frame(
            comparison_container,
            "🏆 Model Comparison Dashboard",
            self.ui_utils.COLOR_HEADER_PURPLE
        )
        header.config(height=40)
        
        # Tabbed interface for different views
        notebook = ttk.Notebook(comparison_container)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Comparison Table
        table_tab = tk.Frame(notebook, bg="white")
        notebook.add(table_tab, text="📋 Comparison Table")
        self.create_comparison_table(table_tab, results_dict)
        
        # Tab 2: Radar Chart
        radar_tab = tk.Frame(notebook, bg="white")
        notebook.add(radar_tab, text="📈 Radar Chart")
        self.chart_display.create_radar_chart(radar_tab, results_dict)
        
        # Tab 3: Bar Chart
        bar_tab = tk.Frame(notebook, bg="white")
        notebook.add(bar_tab, text="📊 Bar Chart")
        self.chart_display.create_bar_chart(bar_tab, results_dict)
    
    def create_comparison_table(self, parent, results_dict):
        """
        Creates a detailed, scrollable comparison table with analysis.
        
        Displays models in a ranked table format with metrics, status badges,
        and comprehensive performance analysis including recommendations.
        
        Args:
            parent: Parent Tkinter widget
            results_dict (dict): Dictionary of model metrics
        """
        # Main container
        main_container = tk.Frame(parent, bg="white")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title header
        title_frame = tk.Frame(main_container, bg="#5C6BC0", height=50)
        title_frame.pack(fill="x", pady=(0, 15))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="📊 MODEL PERFORMANCE COMPARISON",
            font=('Segoe UI', 14, 'bold'),
            fg="white",
            bg="#5C6BC0"
        ).pack(expand=True)
        
        # Scrollable table container
        table_scroll_frame = tk.Frame(main_container, bg="white")
        table_scroll_frame.pack(fill="both", expand=True)
        
        # Canvas and scrollbar setup
        canvas = tk.Canvas(table_scroll_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            table_scroll_frame,
            orient="vertical",
            command=canvas.yview
        )
        
        # Table frame inside canvas
        table_frame = tk.Frame(canvas, bg="white")
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_window = canvas.create_window(
            (0, 0),
            window=table_frame,
            anchor="nw",
            tags="table_frame"
        )
        
        def configure_canvas(event):
            """Update canvas window width when container resizes."""
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", configure_canvas)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Table headers configuration
        headers = [
            ("MODEL", 180),
            ("ACCURACY", 130),
            ("PRECISION", 130),
            ("RECALL", 130),
            ("F1-SCORE", 130),
            ("STATUS", 120),
            ("RANK", 100)
        ]
        
        # Create header row
        header_row = tk.Frame(table_frame, bg="#3F51B5", height=50)
        header_row.pack(fill="x")
        
        # Header cells
        for col, (header_text, width) in enumerate(headers):
            header_cell = tk.Frame(header_row, bg="#3F51B5")
            header_cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            header_row.columnconfigure(col, weight=0, minsize=width)
            
            tk.Label(
                header_cell,
                text=header_text,
                font=('Segoe UI', 11, 'bold'),
                fg="white",
                bg="#3F51B5",
                wraplength=width-10
            ).pack(expand=True, fill="both")
        
        # Process models and calculate scores
        models = list(results_dict.keys())
        
        if not models:
            # Empty state message
            empty_frame = tk.Frame(table_frame, bg="white", height=100)
            empty_frame.pack(fill="x", pady=20)
            
            tk.Label(
                empty_frame,
                text="⚠️ No model data available",
                font=('Segoe UI', 12),
                fg="#666",
                bg="white"
            ).pack(expand=True)
            
            table_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))
            return
        
        # Calculate average scores and sort models
        model_scores = []
        for model in models:
            metrics = results_dict[model]
            avg_score = np.mean([
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ])
            model_scores.append((model, avg_score, metrics))
        
        # Sort by average score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create data rows
        for row_idx, (model, avg_score, metrics) in enumerate(model_scores):
            # Alternating row background
            row_bg = "#FAFAFA" if row_idx % 2 == 0 else "#FFFFFF"
            row_frame = tk.Frame(table_frame, bg=row_bg, height=70)
            row_frame.pack(fill="x")
            
            # Model name cell
            model_cell = tk.Frame(row_frame, bg=row_bg)
            model_cell.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
            row_frame.columnconfigure(0, weight=0, minsize=180)
            
            # Get model icon
            model_icon = self.ui_utils.get_model_icon(model)
            
            tk.Label(
                model_cell,
                text=f"{model_icon} {model}",
                font=('Segoe UI', 11, 'bold'),
                fg="#333",
                bg=row_bg,
                wraplength=170
            ).pack(expand=True, fill="both", padx=5, pady=10)
            
            # Metric columns
            metric_configs = [
                ("accuracy", 1, "ACC"),
                ("precision", 2, "PRE"),
                ("recall", 3, "REC"),
                ("f1_score", 4, "F1")
            ]
            
            for metric_key, col_idx, short_name in metric_configs:
                value = metrics.get(metric_key, 0)
                
                # Metric cell
                metric_cell = tk.Frame(row_frame, bg=row_bg)
                metric_cell.grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1)
                row_frame.columnconfigure(col_idx, weight=0, minsize=130)
                
                # Value label with color coding
                value_label = tk.Label(
                    metric_cell,
                    text=f"{value:.4f}",
                    font=('Segoe UI', 12, 'bold'),
                    fg=self.ui_utils.get_score_color(value),
                    bg=row_bg
                )
                value_label.pack(pady=(10, 5))
                
                # Progress bar
                bar_container = tk.Frame(metric_cell, bg=row_bg)
                bar_container.pack()
                
                bar_bg = tk.Frame(bar_container, bg="#E0E0E0", width=100, height=10)
                bar_bg.pack()
                
                fill_color = self.ui_utils.get_score_color(value)
                fill_width = min(int(value * 100), 100)
                if fill_width > 0:
                    fill = tk.Frame(
                        bar_bg,
                        bg=fill_color,
                        width=fill_width,
                        height=10
                    )
                    fill.place(x=0, y=0, width=fill_width, height=10)
                
                # Percentage label
                tk.Label(
                    metric_cell,
                    text=f"{value*100:.1f}%",
                    font=('Segoe UI', 9),
                    fg="#666",
                    bg=row_bg
                ).pack(pady=(5, 10))
            
            # Status column
            status_cell = tk.Frame(row_frame, bg=row_bg)
            status_cell.grid(row=0, column=5, sticky="nsew", padx=1, pady=1)
            row_frame.columnconfigure(5, weight=0, minsize=120)
            
            status_text, status_color = self.ui_utils.get_performance_status(avg_score)
            tk.Label(
                status_cell,
                text=status_text,
                font=('Segoe UI', 10, 'bold'),
                fg="white",
                bg=status_color,
                width=12,
                height=2
            ).pack(expand=True, fill="both", padx=5, pady=20)
            
            # Rank column
            rank_cell = tk.Frame(row_frame, bg=row_bg)
            rank_cell.grid(row=0, column=6, sticky="nsew", padx=1, pady=1)
            row_frame.columnconfigure(6, weight=0, minsize=100)
            
            rank = row_idx + 1
            rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            rank_icon = rank_icons[rank-1] if rank <= 5 else f"{rank}."
            
            rank_color = "#FF9800" if rank <= 3 else "#666"
            tk.Label(
                rank_cell,
                text=rank_icon,
                font=('Segoe UI', 20 if rank <= 3 else 16),
                fg=rank_color,
                bg=row_bg
            ).pack(expand=True, fill="both")
        
        # Update scroll region
        table_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Mouse wheel scrolling support
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Analysis section
        self._create_analysis_section(main_container, results_dict, model_scores)
    
    def _create_analysis_section(self, parent, results_dict, model_scores):
        """
        Creates the performance analysis and recommendations section.
        
        Args:
            parent: Parent Tkinter widget
            results_dict (dict): Dictionary of model metrics
            model_scores (list): List of tuples (model_name, avg_score, metrics)
        """
        # Analysis frame
        analysis_frame = tk.Frame(
            parent,
            bg="#F5F5F5",
            relief="solid",
            borderwidth=1
        )
        analysis_frame.pack(fill="x", pady=(20, 0))
        
        # Analysis header
        analysis_header = tk.Frame(analysis_frame, bg="#4CAF50", height=40)
        analysis_header.pack(fill="x")
        analysis_header.pack_propagate(False)
        
        tk.Label(
            analysis_header,
            text="📈 PERFORMANCE ANALYSIS & INSIGHTS",
            font=('Segoe UI', 12, 'bold'),
            fg="white",
            bg="#4CAF50"
        ).pack(expand=True)
        
        # Analysis content
        analysis_content = tk.Frame(analysis_frame, bg="white")
        analysis_content.pack(fill="both", expand=True, padx=20, pady=15)
        
        # Generate and display analysis text
        analysis_text = self._generate_performance_analysis(results_dict, model_scores)
        
        analysis_label = tk.Label(
            analysis_content,
            text=analysis_text,
            font=('Segoe UI', 10),
            fg="#333",
            bg="white",
            justify="left",
            wraplength=850,
            anchor="w"
        )
        analysis_label.pack(fill="x", anchor="w")
        
        # Key metrics section
        if len(model_scores) >= 1:
            self._create_key_metrics_section(analysis_content, model_scores, results_dict)
        
        # Recommendations section
        recommendations = self._generate_recommendations(results_dict, model_scores)
        if recommendations:
            rec_frame = tk.Frame(
                analysis_content,
                bg="#FFF3E0",
                relief="solid",
                borderwidth=1
            )
            rec_frame.pack(fill="x", pady=(15, 0))
            
            tk.Label(
                rec_frame,
                text="💡 RECOMMENDATIONS",
                font=('Segoe UI', 10, 'bold'),
                fg="#E65100",
                bg="#FFF3E0"
            ).pack(anchor="w", padx=10, pady=5)
            
            rec_label = tk.Label(
                rec_frame,
                text=recommendations,
                font=('Segoe UI', 9),
                fg="#5D4037",
                bg="#FFF3E0",
                justify="left",
                wraplength=850,
                padx=10,
                pady=5
            )
            rec_label.pack(fill="x", anchor="w")
    
    def _create_key_metrics_section(self, parent, model_scores, results_dict):
        """
        Creates a key metrics display section.
        
        Args:
            parent: Parent Tkinter widget
            model_scores (list): List of model score tuples
            results_dict (dict): Dictionary of model metrics
        """
        key_metrics_frame = tk.Frame(
            parent,
            bg="#F8F9FA",
            relief="solid",
            borderwidth=1
        )
        key_metrics_frame.pack(fill="x", pady=(15, 0))
        
        tk.Label(
            key_metrics_frame,
            text="🔑 KEY METRICS",
            font=('Segoe UI', 10, 'bold'),
            fg="#5C6BC0",
            bg="#F8F9FA"
        ).pack(anchor="w", padx=10, pady=5)
        
        # Calculate key metrics
        best_model, best_score, _ = model_scores[0]
        worst_model, worst_score, _ = model_scores[-1] if len(model_scores) > 1 else (best_model, best_score)
        avg_accuracy = np.mean([score for _, score, _ in model_scores])
        performance_gap = best_score - worst_score if len(model_scores) > 1 else 0
        most_consistent = self._find_most_consistent_model(results_dict)
        
        # Metrics grid
        metrics_grid = tk.Frame(key_metrics_frame, bg="#F8F9FA")
        metrics_grid.pack(fill="x", padx=10, pady=(0, 10))
        
        key_metrics = [
            ("🏆 Best Model", f"{best_model} ({best_score:.2%})"),
            ("📊 Avg. Score", f"{avg_accuracy:.2%}"),
            ("⚡ Performance Gap", f"{performance_gap:.2%}" if len(model_scores) > 1 else "N/A"),
            ("🎯 Most Consistent", most_consistent)
        ]
        
        for i, (label, value) in enumerate(key_metrics):
            metric_item = tk.Frame(metrics_grid, bg="#F8F9FA")
            metric_item.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="w")
            
            tk.Label(
                metric_item,
                text=label,
                font=('Segoe UI', 9, 'bold'),
                fg="#555",
                bg="#F8F9FA"
            ).pack(anchor="w")
            
            tk.Label(
                metric_item,
                text=value,
                font=('Segoe UI', 9),
                fg="#2196F3",
                bg="#F8F9FA"
            ).pack(anchor="w")
    
    def _generate_performance_analysis(self, results_dict, model_scores):
        """
        Generates a comprehensive performance analysis text.
        
        Args:
            results_dict (dict): Dictionary of model metrics
            model_scores (list): List of model score tuples
            
        Returns:
            str: Formatted analysis text
        """
        if not model_scores:
            return "No models to analyze."
        
        analysis_lines = []
        
        # Overall assessment
        best_model, best_score, _ = model_scores[0]
        
        if best_score >= 0.85:
            assessment = "EXCELLENT"
            emoji = "🟢"
        elif best_score >= 0.70:
            assessment = "GOOD"
            emoji = "🟡"
        elif best_score >= 0.55:
            assessment = "MODERATE"
            emoji = "🟠"
        else:
            assessment = "NEEDS IMPROVEMENT"
            emoji = "🔴"
        
        analysis_lines.append(f"{emoji} **OVERALL ASSESSMENT: {assessment}**")
        analysis_lines.append(
            f"   • Best performing model: **{best_model}** with score of **{best_score:.2%}**"
        )
        
        # Model comparison
        if len(model_scores) > 1:
            worst_model, worst_score, _ = model_scores[-1]
            performance_gap = best_score - worst_score
            
            analysis_lines.append(
                f"   • Performance range: **{performance_gap:.2%}** "
                f"({worst_model} → {best_model})"
            )
            
            if performance_gap > 0.15:
                analysis_lines.append("   • ⚠️ Significant performance variation detected")
            elif performance_gap > 0.05:
                analysis_lines.append("   • ⚡ Moderate performance differences")
            else:
                analysis_lines.append("   • ✅ Models perform similarly")
        
        # Metric-based analysis
        analysis_lines.append("\n📊 **METRIC ANALYSIS:**")
        
        for model, score, metrics in model_scores[:min(3, len(model_scores))]:
            strong_metric = max(
                ['accuracy', 'precision', 'recall', 'f1_score'],
                key=lambda m: metrics.get(m, 0)
            )
            weak_metric = min(
                ['accuracy', 'precision', 'recall', 'f1_score'],
                key=lambda m: metrics.get(m, 0)
            )
            
            strong_value = metrics.get(strong_metric, 0)
            weak_value = metrics.get(weak_metric, 0)
            
            analysis_lines.append(
                f"   • **{model}**: Strong in {strong_metric} ({strong_value:.2%}), "
                f"needs work on {weak_metric} ({weak_value:.2%})"
            )
        
        # Key insights
        analysis_lines.append("\n🎯 **KEY INSIGHTS:**")
        
        if best_score >= 0.80:
            analysis_lines.append("   • Models are performing well for this task")
            analysis_lines.append("   • Consider ensemble methods for even better results")
        elif best_score >= 0.65:
            analysis_lines.append("   • Models show acceptable performance")
            analysis_lines.append("   • Hyperparameter tuning could improve results")
        else:
            analysis_lines.append("   • Models need significant improvement")
            analysis_lines.append("   • Consider feature engineering or different algorithms")
        
        return "\n".join(analysis_lines)
    
    def _find_most_consistent_model(self, results_dict):
        """
        Finds the model with the most consistent performance across metrics.
        
        Args:
            results_dict (dict): Dictionary of model metrics
            
        Returns:
            str: Model name with consistency score, or "N/A" if no models
        """
        if not results_dict:
            return "N/A"
        
        consistencies = {}
        for model, metrics in results_dict.items():
            values = [
                metrics.get(m, 0)
                for m in ['accuracy', 'precision', 'recall', 'f1_score']
            ]
            # Lower standard deviation = higher consistency
            if len(values) > 1 and np.std(values) > 0:
                consistency = 1 - np.std(values)
                consistencies[model] = max(consistency, 0)
            else:
                consistencies[model] = 1.0
        
        if consistencies:
            most_consistent = max(consistencies.items(), key=lambda x: x[1])
            return f"{most_consistent[0]} ({most_consistent[1]:.2%})"
        
        return "N/A"
    
    def _generate_recommendations(self, results_dict, model_scores):
        """
        Generates actionable recommendations based on model performance.
        
        Args:
            results_dict (dict): Dictionary of model metrics
            model_scores (list): List of model score tuples
            
        Returns:
            str: Formatted recommendations text, or empty string if no models
        """
        if not model_scores:
            return ""
        
        best_score = model_scores[0][1]
        recommendations = []
        
        if best_score < 0.60:
            recommendations.append(
                "⚠️ Consider using different algorithms or more complex models"
            )
            recommendations.append("📊 Review dataset quality and feature selection")
            recommendations.append("🔧 Try extensive hyperparameter tuning")
        elif best_score < 0.75:
            recommendations.append("⚡ Try ensemble methods (Voting, Stacking)")
            recommendations.append("🎯 Focus on improving the weakest metric")
            recommendations.append("🔍 Consider feature engineering")
        elif best_score < 0.85:
            recommendations.append("✅ Performance is good, consider deployment")
            recommendations.append("⚡ Minor hyperparameter tuning could help")
            recommendations.append("📈 Monitor model performance over time")
        else:
            recommendations.append("🎉 Excellent performance! Ready for production")
            recommendations.append("⚡ Consider model compression for deployment")
            recommendations.append("📊 Implement monitoring and retraining pipeline")
        
        return " • " + "\n • ".join(recommendations)

