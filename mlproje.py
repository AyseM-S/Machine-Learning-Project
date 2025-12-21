import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import os

# Import your visualization module
from visualization import ResultVisualizer

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

    # --- MODIFIED: Added hidden_layers_str parameter ---
    def run_pipeline(self, model_name, normalize, ohe, split_ratio, hidden_layers_str):
        """
        Runs the ML process with parameters from the GUI and returns the results.
        """
        if self.data is None:
            return None, "Please load data first."
        
        # --- 1. Preprocessing Simulation ---
        preprocessing_info = f"Preprocessing: Normalize={normalize}, OHE={ohe}"
        print(preprocessing_info)
        
        # --- 2. Model Training and Evaluation Simulation ---
        print(f"Training Model: {model_name} (Split Ratio: {split_ratio})")
        
        # --- NEW LOGIC: Use hidden_layers_str for MLP ---
        if model_name == "Multilayer Perceptron":
            print(f"MLP Hidden Layers requested by user: {hidden_layers_str}")
            # Your real ModelManager logic will handle conversion to tuple:
            # layer_tuple = tuple(int(x.strip()) for x in hidden_layers_str.split(","))

        # Generating Simulated Metrics (Includes project metrics)
        metrics = {
            "model": model_name,
            "accuracy": random.uniform(0.75, 0.99),
            "precision": random.uniform(0.70, 0.95),
            "recall": random.uniform(0.65, 0.90),
            "f1_score": random.uniform(0.70, 0.95),
        }
        
        # Simulated Confusion Matrix 
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
        
        master.geometry("1000x700") 
        
        self.ml_pipeline = MLPipeline() # Instance of the Backend class
        
        # --- Tkinter Variables ---
        self.dataset_path = tk.StringVar(value="No data loaded yet.")
        self.normalize_var = tk.IntVar()
        self.ohe_var = tk.IntVar()
        self.model_selection_var = tk.StringVar(value="Perceptron") 
        self.split_ratio = tk.DoubleVar(value=0.7)
        # --- NEW: Variable for Hidden Layers Input ---
        self.hidden_layers_var = tk.StringVar(value="64, 32") 
        
        # Initialize your visualizer
        self.visualizer = None
        
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
            
        # --- NEW: Hidden Layers Input for MLP ---
        tk.Label(control_frame, text="(MLP Only) Hidden Layers (e.g., 128,64):", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        tk.Entry(control_frame, textvariable=self.hidden_layers_var, width=30, justify=tk.CENTER).pack(fill="x", padx=10)
        # --- END NEW WIDGETS ---

        # 4. Train/Test Split Slider
        tk.Label(control_frame, text="\n4. Train/Test Split Ratio:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        tk.Scale(control_frame, from_=0.5, to=0.95, resolution=0.01, 
                  orient=tk.HORIZONTAL, variable=self.split_ratio, 
                  label="Train Ratio").pack(fill="x", padx=10)

        # 5. Start Training Button
        tk.Button(control_frame, text="🔥 TRAIN MODELS 🔥", 
                  command=self.start_training, fg="white", bg="#FF5722", 
                  font=('Arial', 11, 'bold'), activebackground="#e64a19").pack(fill="x", pady=20, padx=10)
        
        # 6. Compare Models Button (Optional)
        tk.Button(control_frame, text="📊 COMPARE ALL MODELS", 
                  command=self.compare_all_models, fg="white", bg="#9C27B0", 
                  font=('Arial', 10, 'bold'), activebackground="#7B1FA2").pack(fill="x", pady=5, padx=10)
        
        # ----------------------------------
        # Right Side: Results Area
        # ----------------------------------
        self.results_frame = tk.LabelFrame(main_frame, text="Results and Evaluation", padx=10, pady=10)
        self.results_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # Make the right column expand
        main_frame.columnconfigure(1, weight=1) 
        main_frame.rowconfigure(0, weight=1)
        
        # Initialize your visualizer with the results frame
        self.visualizer = ResultVisualizer(self.results_frame)
        
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
                file_name = os.path.basename(file_path)
                num_rows = len(self.ml_pipeline.data)
                self.dataset_path.set(f"✅ Loaded: {file_name}\n({num_rows} rows)")
                messagebox.showinfo("Success", f"Dataset loaded successfully:\n{file_name}")
            else:
                self.dataset_path.set("❌ Loading Failed!")
                messagebox.showerror("Error", "An issue occurred while loading data, or the file is empty/invalid.")

    def start_training(self):
        """Starts the training process, calls the backend, and displays results using visualizer."""
        
        if self.ml_pipeline.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        # Get parameters
        model_name = self.model_selection_var.get()
        normalize = self.normalize_var.get()
        ohe = self.ohe_var.get()
        split_ratio = self.split_ratio.get()
        hidden_layers_str = self.hidden_layers_var.get()
        
        # Show training message using visualizer
        self.visualizer.display_training_message(model_name)
        self.master.update()  # Update GUI to show message
        
        # Run the backend (passing the new parameter)
        metrics, cm_data = self.ml_pipeline.run_pipeline(model_name, normalize, ohe, split_ratio, hidden_layers_str)
        
        if metrics:
            # Clear and display results using your visualizer
            self.visualizer.clear_results()
            self.visualizer.display_metrics_table(metrics, model_name)
            self.visualizer.display_confusion_matrix(cm_data)
            
            messagebox.showinfo("Success", f"Model training completed: {model_name}")
        else:
            self.visualizer.display_error_message(cm_data)
    
    def compare_all_models(self):
        """Trains and compares all available models."""
        if self.ml_pipeline.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        # Show training message
        self.visualizer.display_training_message("All Models")
        
        # Simulate training all models
        models = ["Perceptron", "Multilayer Perceptron", "Decision Tree"]
        results_dict = {}
        
        for model_name in models:
            # Simulate metrics for each model
            metrics = {
                "accuracy": random.uniform(0.70, 0.95),
                "precision": random.uniform(0.65, 0.90),
                "recall": random.uniform(0.60, 0.85),
                "f1_score": random.uniform(0.65, 0.90),
            }
            results_dict[model_name] = metrics
        
        # Display comparison
        self.visualizer.clear_results()
        self.visualizer.display_model_comparison(results_dict)
        
        messagebox.showinfo("Comparison Complete", "All models have been trained and compared!")


if __name__ == '__main__':
    root = tk.Tk()
    app = MLToolkitGUI(root)
    root.mainloop()