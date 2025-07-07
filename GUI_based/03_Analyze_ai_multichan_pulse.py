# -*- coding: utf-8 -*-
"""
Updated Analysis Tool with Peak-Trough Pulse Detection and CSV Export
Multi-channel support based on 02_Analyze_ai_multichan_freq.py structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, DoubleVar, Entry, Label, Frame, IntVar
import os
from scipy.signal import detrend, find_peaks

def find_pulses(signal, sample_rate, threshold, max_gap_us=500):
    peaks, _ = find_peaks(detrend(signal), height=threshold, distance=50)
    troughs, _ = find_peaks(detrend(-signal), height=threshold, distance=50)
    gap_samples = int((max_gap_us / 1e6) * sample_rate)

    used_troughs = set()
    pairs = []
    for peak in peaks:
        valid_troughs = [t for t in troughs if abs(t - peak) <= gap_samples and t not in used_troughs]
        if valid_troughs:
            trough = min(valid_troughs, key=lambda x: abs(x - peak))
            used_troughs.add(trough)
            if peak > trough:
                peak, trough = trough, peak
            pairs.append((peak, trough))
    return np.array(pairs)


class AnalysisGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pulse Detection and Analysis Tool")
        
        # GUI variables
        self.threshold = DoubleVar(value=0.10)
        self.time_bin_size = DoubleVar(value=1.0)
        self.start_time_s = DoubleVar(value=0.0)
        self.end_time_s = DoubleVar(value=60.0)
        self.y_offset = DoubleVar(value=2.0)  # Default y-offset for raw plot
        self.log_data = None
        self.data = None
        self.pulse_pairs = {}  # Dictionary to store pulse pairs for each channel

        # Create control area
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="top", fill="x")  # Move to top

        # Add controls
        Label(self.control_frame, text="Start (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.start_time_s, width=8).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="End (s, 0=EOF):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.end_time_s, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Pulse threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Time Bin (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.time_bin_size, width=8).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Y Offset:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.y_offset, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Export Pulse Rate CSV", command=self.export_pulse_rate_summary).pack(side="left", padx=5, pady=5)

        # Create plot area
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        # Initialize plot (will be updated for channel count)
        self.fig = None
        self.axes = None
        self.canvas = None
        self.toolbar = None

    def load_file(self):
        """Load the log file and data."""
        filepath = filedialog.askopenfilename(title="Select Log File", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.log_data, self.data = self.load_data(filepath)
            self.refresh_plot()
    
    def load_data(self, log_filepath):
        with open(log_filepath, 'r') as file:
            log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}

        base_filepath = log_filepath.split('log_')[0] + log_filepath.split('log_')[-1].split('.')[0]
        feather_filepath = base_filepath + '.feather'
        parquet_filepath = base_filepath + '.parquet'
        bin_filepath = base_filepath + '.bin'

        sample_rate = int(log_data["Sample_Rate"])
        start_time = float(self.start_time_s.get())
        end_time = float(self.end_time_s.get())
        start_sample = int(start_time * sample_rate)
        end_sample = None if end_time == 0 else int(end_time * sample_rate)

        # Determine number of channels from log or file
        if "N_Input_Channels" in log_data:
            n_channels = int(log_data["N_Input_Channels"])
        elif "Input_Channels" in log_data:
            n_channels = len(log_data["Input_Channels"].split(","))
        else:
            n_channels = 1  # fallback

        n_cols = n_channels + 1  # time_ms + channels

        if os.path.exists(bin_filepath):
            if end_sample is not None:
                n_samples = end_sample - start_sample
                offset = start_sample * n_cols * 8  # float64 = 8 bytes
                with open(bin_filepath, 'rb') as f:
                    f.seek(offset)
                    raw_data = np.fromfile(f, dtype='f8', count=n_samples * n_cols)
            else:
                offset = start_sample * n_cols * 8
                with open(bin_filepath, 'rb') as f:
                    f.seek(offset)
                    raw_data = np.fromfile(f, dtype='f8')
            reshaped = raw_data.reshape(-1, n_cols)
            columns = ['time_ms'] + [f'ch{i+1}' for i in range(n_channels)]
            data = pd.DataFrame(reshaped, columns=columns)

        elif os.path.exists(feather_filepath):
            import pyarrow.feather as feather
            import pyarrow as pa
            table = feather.read_table(feather_filepath)
            if end_sample is None:
                table = table.slice(start_sample)
            else:
                table = table.slice(start_sample, end_sample - start_sample)
            data = table.to_pandas()

        elif os.path.exists(parquet_filepath):
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_filepath)
            if end_sample is None:
                table = table.slice(start_sample)
            else:
                table = table.slice(start_sample, end_sample - start_sample)
            data = table.to_pandas()

        else:
            raise FileNotFoundError("Expected data file not found.")

        return log_data, data


    def refresh_plot(self):
        """Refresh the plots based on current settings."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return

        # Extract key parameters
        sample_rate = int(self.log_data["Sample_Rate"])
        rec_id = self.log_data.get("Recording_ID", "")

        # Determine number of channels
        channel_cols = [col for col in self.data.columns if col.startswith("ch")]
        n_channels = len(channel_cols)
        total_samples = len(self.data)
        start_offset = float(self.start_time_s.get())
        time_axis = start_offset + np.arange(total_samples) / sample_rate

        # Prepare figure and axes: n_channels raw plots + 1 instantaneous rates + 1 combined rate
        n_rows = n_channels + 2
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows), squeeze=False)
        self.axes = self.axes.flatten()

        # Clear pulse pairs dictionary
        self.pulse_pairs = {}

        # Plot each channel with detected pulses
        y_offset_val = self.y_offset.get()
        for i, ch in enumerate(channel_cols):
            channel_data = detrend(self.data[ch])
            pairs = find_pulses(channel_data, sample_rate, self.threshold.get())
            self.pulse_pairs[ch] = pairs
            
            peaks = pairs[:, 0] if len(pairs) else []
            
            # Plot with optional offset for better visualization
            if n_channels > 1:
                offset = i * y_offset_val
                self.axes[i].plot(time_axis, channel_data + offset, label=f"{ch}")
                if len(peaks):
                    self.axes[i].plot(time_axis[peaks], channel_data[peaks] + offset, "rx", label="Peaks")
            else:
                self.axes[i].plot(time_axis, channel_data, label=f"{ch}")
                if len(peaks):
                    self.axes[i].plot(time_axis[peaks], channel_data[peaks], "rx", label="Peaks")
            
            self.axes[i].set_title(f"{ch} with Detected Pulses - ID: {rec_id}")
            self.axes[i].set_ylabel("Amplitude")
            self.axes[i].legend()

        # Plot instantaneous pulse rates for all channels
        inst_rates_ax = self.axes[n_channels]
        inst_rates_ax.clear()
        inst_rates_ax.set_title("Instantaneous Pulse Rates")
        inst_rates_ax.set_ylabel("Rate (Hz)")
        
        for i, ch in enumerate(channel_cols):
            pairs = self.pulse_pairs[ch]
            if len(pairs) > 1:
                t_centers = (pairs[:, 0] + pairs[:, 1]) / 2 / sample_rate + start_offset
                dt = np.diff(t_centers)
                rates = 1 / dt
                mid_times = (t_centers[:-1] + t_centers[1:]) / 2
                inst_rates_ax.plot(mid_times, rates, 'o', label=f"{ch}")
        
        inst_rates_ax.legend()
        inst_rates_ax.set_xlabel("Time (s)")

        # Plot combined unique pulse rate
        combined_ax = self.axes[n_channels + 1]
        combined_ax.clear()
        combined_ax.set_title("Combined Unique Pulse Rate")
        combined_ax.set_ylabel("Rate (Hz)")
        combined_ax.set_xlabel("Time (s)")
        
        # Combine and deduplicate pulses from all channels
        all_pairs = []
        for ch in channel_cols:
            if len(self.pulse_pairs[ch]):
                all_pairs.extend(self.pulse_pairs[ch])
        
        if all_pairs:
            all_pairs = np.array(all_pairs)
            centers = np.mean(all_pairs, axis=1).astype(int)
            centers.sort()
            
            # Deduplicate by ±10 sample tolerance
            unique_centers = []
            for c in centers:
                if not unique_centers or abs(c - unique_centers[-1]) > 10:
                    unique_centers.append(c)
            
            unique_times = np.array(unique_centers) / sample_rate + start_offset
            
            # Bin size and rate per bin
            bin_size = float(self.time_bin_size.get())
            if bin_size > 0 and len(unique_times):
                max_time = unique_times[-1]
                bins = np.arange(start_offset, max_time + bin_size, bin_size)
                counts, _ = np.histogram(unique_times, bins)
                rates = counts / bin_size
                bin_centers = bins[:-1] + bin_size / 2
                combined_ax.plot(bin_centers, rates, 'o', color='black')

        self.fig.tight_layout()

        # Update canvas and toolbar
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        # Place toolbar at the top
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def export_pulse_rate_summary(self):
        if self.data is None or self.log_data is None:
            print("No data loaded.")
            return

        sample_rate = int(self.log_data["Sample_Rate"])
        bin_size = float(self.time_bin_size.get())
        start_time = float(self.start_time_s.get())
        end_time = float(self.end_time_s.get())
        total_time = self.data["time_ms"].iloc[-1] / 1000.0 if end_time == 0 else end_time

        n_bins = int(np.ceil((total_time - start_time) / bin_size))
        
        if not self.pulse_pairs:
            print("Run 'Refresh Plot' before exporting.")
            return

        # Determine number of channels
        channel_cols = [col for col in self.data.columns if col.startswith("ch")]
        
        def count_per_bin(times):
            return np.histogram(times, bins=np.arange(start_time, start_time + n_bins * bin_size + bin_size, bin_size))[0]

        # Create DataFrame with time column
        df_data = {"TimeStart_s": start_time + np.arange(n_bins) * bin_size}
        
        # Add pulse counts for each channel
        for ch in channel_cols:
            pairs = self.pulse_pairs[ch]
            if len(pairs):
                t_centers = (pairs[:, 0] + pairs[:, 1]) / 2 / sample_rate + start_time
                counts = count_per_bin(t_centers)
                df_data[f"{ch}_Pulses"] = counts
            else:
                df_data[f"{ch}_Pulses"] = np.zeros(n_bins)

        # Calculate combined unique pulse rate
        all_pairs = []
        for ch in channel_cols:
            if len(self.pulse_pairs[ch]):
                all_pairs.extend(self.pulse_pairs[ch])
        
        if all_pairs:
            all_pairs = np.array(all_pairs)
            centers = np.mean(all_pairs, axis=1).astype(int)
            centers.sort()
            
            # Deduplicate by ±10 sample tolerance
            unique_centers = []
            for c in centers:
                if not unique_centers or abs(c - unique_centers[-1]) > 10:
                    unique_centers.append(c)
            
            t_combined = np.array(unique_centers) / sample_rate + start_time
            combined_counts = count_per_bin(t_combined)
            combined_rate = combined_counts / bin_size
            df_data["Combined_Unique_Rate_Hz"] = combined_rate
        else:
            df_data["Combined_Unique_Rate_Hz"] = np.zeros(n_bins)

        df = pd.DataFrame(df_data)

        out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"Exported to {out_path}")


if __name__ == "__main__":
    root = Tk()
    # Set window size to fit screen (with some margin)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.95)
    window_height = int(screen_height * 0.85)
    root.geometry(f"{window_width}x{window_height}")
    app = AnalysisGUI(root)

    def on_closing():
        import matplotlib.pyplot as plt
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
