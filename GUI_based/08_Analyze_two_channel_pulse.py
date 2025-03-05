# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:48:58 2025

@author: Stefan Mucha
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, DoubleVar, Entry, Label, Frame
import os
from scipy.signal import detrend, find_peaks

class AnalysisGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")
        
        # GUI variables
        self.threshold = DoubleVar(value=0.10)
        self.log_data = None
        self.data = None
        self.mean_pulse_form = None  # To store computed mean pulse form

        # Create plot area
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        # Create control area
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="bottom", fill="x")

        # Add controls
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        # Additional controls for pulse analysis:
        Label(self.control_frame, text="Pulse threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)

        # Initialize plot
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 8))
        self.canvas = None
        self.toolbar = None

    def load_file(self):
        """Load the log file and data."""
        filepath = filedialog.askopenfilename(title="Select Log File", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.log_data, self.data = self.load_data(filepath)
            self.refresh_plot()

    def load_data(self, log_filepath):
        """Load log file and associated data file."""
        with open(log_filepath, 'r') as file:
            log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}
    
        base_filepath = log_filepath.split('log_')[0] + log_filepath.split('log_')[-1].split('.')[0]
        feather_filepath = base_filepath + '.feather'
        parquet_filepath = base_filepath + '.parquet'
        
        if os.path.exists(feather_filepath):
            data = pd.read_feather(feather_filepath)
        elif os.path.exists(parquet_filepath):
            data = pd.read_parquet(parquet_filepath)
        else:
            raise FileNotFoundError(
                f"Data file not found. Expected either {feather_filepath} or {parquet_filepath}"
            )
        
        return log_data, data
    
    def refresh_plot(self):
        """Refresh the plots based on current settings and analysis type."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return
        
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.mode = ''
            self.toolbar.update()
            
        for ax in self.axes:
            ax.clear()

        sample_rate = int(self.log_data["Sample Rate"])
        rec_id = self.log_data["Recording ID"]
        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate
        
        # Detrend channels
        channel_1 = detrend(self.data["ch1"])
        channel_2 = detrend(self.data["ch2"])

        # Pulse-type analysis
        # Use channel_1 for pulse detection
        peaks_1, properties_1 = find_peaks(channel_1, height=self.threshold.get(), distance = 50)
        peaks_2, properties_2 = find_peaks(channel_2, height=self.threshold.get(), distance = 50)
        
        # Plot raw data with detected peaks
        self.axes[0].plot(time_axis, channel_1, label="Ch 1")
        self.axes[0].plot(time_axis[peaks_1], channel_1[peaks_1], "rx", label="Peaks")
        
        self.axes[0].set_title(f"Raw Data with Detected Peaks - Recording ID: {rec_id} Ch 1")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].legend(loc="lower left")
        
        self.axes[1].plot(time_axis, channel_2, label="Ch 2")
        self.axes[1].plot(time_axis[peaks_2], channel_2[peaks_2], "rx", label="Peaks")

        self.axes[1].set_title("Ch 2")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].legend(loc="lower left")
        
        # Compute inter-pulse intervals and instantaneous rate
        if len(peaks_1) > 1:
            peak_times_1 = time_axis[peaks_1]
            dt_1 = np.diff(peak_times_1)
            inst_rate_1 = 1.0 / dt_1
            mid_times_1 = (peak_times_1[:-1] + peak_times_1[1:]) / 2
            
            peak_times_2 = time_axis[peaks_2]
            dt_2 = np.diff(peak_times_2)
            inst_rate_2 = 1.0 / dt_2
            mid_times_2 = (peak_times_2[:-1] + peak_times_2[1:]) / 2
            
            self.axes[2].plot(mid_times_1, inst_rate_1, 'o')
            self.axes[2].set_title("Instantaneous Pulse Rate Ch 1")
            self.axes[2].set_ylabel("Rate (Hz)")
            self.axes[2].set_xlabel("Time (s)")
            
            self.axes[3].plot(mid_times_2, inst_rate_2, 'o')
            self.axes[3].set_title("Instantaneous Pulse Rate Ch 2")
            self.axes[3].set_ylabel("Rate (Hz)")
            self.axes[3].set_xlabel("Time (s)")

        else:
            self.axes[2].text(0.5, 0.5, "Not enough peaks detected", ha="center")
            self.axes[3].text(0.5, 0.5, "Not enough peaks detected", ha="center")
        
        self.fig.tight_layout()
        
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()
    
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
