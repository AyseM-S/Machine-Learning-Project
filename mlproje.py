import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random # For simulation results
import os

# ==============================================================================
# SECTION 1: BACKEND (Machine Learning Simulation)
# This class takes parameters from the GUI and simulates ML operations.
# In your actual project, you will replace this with real ML code using Scikit-learn, etc.
# ==============================================================================

class MLPipeline:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Loads a CSV file into a DataFrame."""
        try:
            # Check file extension
            if not file_path.lower().endswith('.csv'):
                raise ValueError("Please select a .csv file only.")
                
            self.data = pd.read_csv(file_path)
            # Simple check if data is empty
            if self.data.empty:
                 raise ValueError("The loaded file is empty.")
            return True
        except Exception as e:
            print(f"Data loading error: {e}")
            self.data = None
            return False

    def run_pipeline(self, model_name, normalize, ohe, split_ratio):
        """
        Runs the ML process with parameters from the GUI and returns the results.
        """
        if self.data is None:
            return None, "Please load data first."
        
        # --- 1. Preprocessing Simulation ---
        # In a real application, you would apply StandardScaler/OneHotEncoder here based on 'normalize' and 'ohe'.
        preprocessing_info = f"Preprocessing: Normalize={normalize}, OHE={ohe}"
        print(preprocessing_info)
        
        # --- 2. Model Training and Evaluation Simulation ---
        # In a real application, you would split X_train, X_test, y_train, y_test and train the model here.
        print(f"Training Model: {model_name} (Split Ratio: {split_ratio})")
        
        # Generating Simulated Metrics (Includes project metrics)
        metrics = {
            "model": model_name,
            "accuracy": random.uniform(0.75, 0.99),
            "precision": random.uniform(0.70, 0.95),
            "recall": random.uniform(0.65, 0.90),
            "f1_score": random.uniform(0.70, 0.95),
        }
        
        # Simulated Confusion Matrix (3x3 - assuming Multi-class Classification)
        # In your real code, this should be generated from y_true and y_pred.
        cm_data = np.array([
            [random.randint(100, 200), random.randint(5, 20), random.randint(0, 5)],
            [random.randint(5, 20), random.randint(100, 200), random.randint(0, 5)],
            [random.randint(0, 5), random.randint(5, 20), random.randint(100, 200)]
        ])
        
        return metrics, cm_data

# ==============================================================================
# SECTION 2: GUI (Tkinter Interface)
# ==============================================================================

class MLToolkitGUI:
    def __init__(self, master):
        self.master = master
        master.title("GUI-Based Classification and Evaluation Tool")
        
        # Set initial window size
        master.geometry("1000x700") 
        
        self.ml_pipeline = MLPipeline() # Instance of the Backend class
        
        # --- Tkinter Variables ---
        self.dataset_path = tk.StringVar(value="No data loaded yet.")
        self.normalize_var = tk.IntVar()
        self.ohe_var = tk.IntVar()
        self.model_selection_var = tk.StringVar(value="Perceptron") 
        self.split_ratio = tk.DoubleVar(value=0.7)
        self.current_canvas = None # To clear the Matplotlib canvas
        
        self.create_widgets()

    def create_widgets(self):
        # Main Frame: Two-column layout for controls and results
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # ----------------------------------
        # Left Side: Control Panel
        # ----------------------------------
        control_frame = tk.LabelFrame(main_frame, text="Control Panel (Inputs)", padx=15, pady=15, width=300)
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")

        # 1. Dataset Loading
        tk.Label(control_frame, text="1. Load Dataset (.csv):", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        tk.Button(control_frame, text="📁 Select Dataset", command=self.load_dataset, 
                  bg="#4CAF50", fg="white", activebackground="#45a049", padx=10).pack(fill="x")
        tk.Label(control_frame, textvariable=self.dataset_path, fg="#0056b3", wraplength=250, justify=tk.LEFT).pack(pady=(5, 10))

        # 2. Preprocessing Options
        tk.Label(control_frame, text="2. Preprocessing Options:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        tk.Checkbutton(control_frame, text="Normalization (Scaler)", variable=self.normalize_var, anchor="w").pack(fill="x")
        tk.Checkbutton(control_frame, text="One-Hot Encoding (Categorical)", variable=self.ohe_var, anchor="w").pack(fill="x")

        # 3. Model Selection
        tk.Label(control_frame, text="\n3. Classification Model:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        models = ["Perceptron", "Multilayer Perceptron", "Decision Tree"]
        for model in models:
            tk.Radiobutton(control_frame, text=model, variable=self.model_selection_var, value=model, anchor="w").pack(fill="x")

        # 4. Train/Test Split Slider
        tk.Label(control_frame, text="\n4. Train/Test Split Ratio:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        tk.Scale(control_frame, from_=0.5, to=0.95, resolution=0.01, 
                 orient=tk.HORIZONTAL, variable=self.split_ratio, 
                 label="Train Ratio").pack(fill="x", padx=10)

        # 5. Start Training Button
        tk.Button(control_frame, text="🔥 TRAIN MODELS 🔥", 
                  command=self.start_training, fg="white", bg="#FF5722", 
                  font=('Arial', 11, 'bold'), activebackground="#e64a19").pack(fill="x", pady=20, padx=10)
        
        # ----------------------------------
        # Right Side: Results Area
        # ----------------------------------
        self.results_frame = tk.LabelFrame(main_frame, text="Results and Evaluation", padx=10, pady=10)
        self.results_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # Make the right column expand
        main_frame.columnconfigure(1, weight=1) 
        main_frame.rowconfigure(0, weight=1)
        
        # Initial message
        tk.Label(self.results_frame, text="Use the panel on the left to see training results.", 
                 font=('Arial', 12, 'italic'), fg="gray").pack(expand=True)

    def load_dataset(self):
        """Allows the user to select a CSV file and loads it to the backend."""
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            if self.ml_pipeline.load_data(file_path):
                # If successful, show the file name and number of rows
                file_name = os.path.basename(file_path)
                num_rows = len(self.ml_pipeline.data)
                self.dataset_path.set(f"✅ Loaded: {file_name}\n({num_rows} rows)")
                messagebox.showinfo("Success", f"Dataset loaded successfully:\n{file_name}")
            else:
                # Show error message
                self.dataset_path.set("❌ Loading Failed!")
                messagebox.showerror("Error", "An issue occurred while loading data, or the file is empty/invalid.")

    def clear_results(self):
        """Clears all previous widgets from the results frame."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Also clear the Matplotlib canvas if it exists
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
            
    def display_metrics(self, metrics):
        """Displays metrics (Accuracy, Precision, Recall, F1-Score) in a table format."""
        
        tk.Label(self.results_frame, text=f"Selected Model: {metrics['model']}", 
                 font=('Arial', 14, 'bold'), fg="#FF5722").pack(pady=(5, 10))
                 
        # Create a Treeview table to display metrics
        table = ttk.Treeview(self.results_frame, columns=("Metric", "Score"), show="headings", height=4)
        table.heading("Metric", text="Evaluation Metric")
        table.heading("Score", text="Score")
        
        # Adjust column widths
        table.column("Metric", width=200, anchor=tk.W)
        table.column("Score", width=150, anchor=tk.CENTER)

        # Add data
        metric_labels = {
            "accuracy": "Accuracy", 
            "precision": "Precision", 
            "recall": "Recall", 
            "f1_score": "F1 Score"
        }

        for key, label in metric_labels.items():
            score = f"{metrics[key]:.4f}" # Format to 4 decimal places
            table.insert("", tk.END, values=(label, score), tags=('oddrow' if list(metric_labels.keys()).index(key) % 2 else 'evenrow'))
        
        # Configure row colors (for aesthetics)
        table.tag_configure('oddrow', background='#f0f0f0')
        table.tag_configure('evenrow', background='#ffffff')

        table.pack(pady=10, padx=5, fill="x")
        
        # Description text
        tk.Label(self.results_frame, text="🔥 Confusion Matrix 🔥", 
                 font=('Arial', 12, 'bold')).pack(pady=(15, 5))


    def display_confusion_matrix(self, cm_data):
        """Displays the Confusion Matrix graph using Matplotlib."""
        
        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Draw Heatmap [Image of Confusion Matrix Heatmap]
        cax = ax.matshow(cm_data, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Labels (In a real project, these should be class names)
        classes = ['Class 1', 'Class 2', 'Class 3']
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Add text to matrix cells
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                ax.text(j, i, str(cm_data[i, j]), va='center', ha='center', color='black')

        # Title and axis labels
        ax.set_title("Confusion Matrix Simulation", y=1.05)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        plt.tight_layout() # Prevent layout overlap
        
        # Embed in Tkinter
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        self.current_canvas.draw()
        
        # Pack the Canvas widget, allowing it to fill the frame
        canvas_widget = self.current_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


    def display_results(self, metrics, cm_data):
        """Displays all results sequentially."""
        self.clear_results()
        self.display_metrics(metrics)
        self.display_confusion_matrix(cm_data)


    def start_training(self):
        """Starts the training process, calls the backend, and displays results."""
        
        if self.ml_pipeline.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        # Get parameters
        model_name = self.model_selection_var.get()
        normalize = self.normalize_var.get()
        ohe = self.ohe_var.get()
        split_ratio = self.split_ratio.get()
        
        # Notify the user that training has started
        self.clear_results()
        tk.Label(self.results_frame, text=f"Training '{model_name}' Model...", 
                 font=('Arial', 16, 'bold'), fg="#007bff").pack(pady=50, fill="x")
        
        # Run the backend
        metrics, cm_data = self.ml_pipeline.run_pipeline(model_name, normalize, ohe, split_ratio)
        
        if metrics:
            # Display successful results
            self.display_results(metrics, cm_data)
            messagebox.showinfo("Success", f"Model training completed: {model_name}")
        else:
            # Display error message if returned
            messagebox.showerror("Error", cm_data) 
            self.clear_results()
            tk.Label(self.results_frame, text="An error occurred during training.", 
                     font=('Arial', 14), fg="red").pack(pady=50)


if __name__ == '__main__':
    root = tk.Tk()
    app = MLToolkitGUI(root)
    root.mainloop()