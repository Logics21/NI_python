# -*- coding: utf-8 -*-
"""
Updated Analysis Tool with Peak-Trough Pulse Detection and CSV Export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, DoubleVar, Entry, Label, Frame
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
        self.root.title("Data Analysis Tool")
        
        self.threshold = DoubleVar(value=0.10)
        self.time_bin_size = DoubleVar(value=1.0)
        self.start_time_s = DoubleVar(value=0.0)
        self.end_time_s = DoubleVar(value=60.0)
        self.log_data = None
        self.data = None
        self.pulse_pairs_ch1 = None
        self.pulse_pairs_ch2 = None

        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="bottom", fill="x")
        
        
        Label(self.control_frame, text="Start (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.start_time_s, width=8).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="End (s, 0=EOF):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.end_time_s, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Pulse threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Time Bin (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.time_bin_size, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Export Pulse Rate CSV", command=self.export_pulse_rate_summary).pack(side="left", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 8))
        self.canvas = None
        self.toolbar = None

    def load_file(self):
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
    
        sample_rate = int(log_data["Sample Rate"])
        start_time = float(self.start_time_s.get())
        end_time = float(self.end_time_s.get())
    
        start_sample = int(start_time * sample_rate)
        end_sample = None if end_time == 0 else int(end_time * sample_rate)
        n_cols = 3  # time_ms, ch1, ch2
    
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
            data = pd.DataFrame(reshaped, columns=['time_ms', 'ch1', 'ch2'])
    
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
        if self.log_data is None or self.data is None:
            print("No data loaded.")
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

        ch1 = detrend(self.data["ch1"])
        ch2 = detrend(self.data["ch2"])

        pairs1 = find_pulses(ch1, sample_rate, self.threshold.get())
        pairs2 = find_pulses(ch2, sample_rate, self.threshold.get())
        
        self.pulse_pairs_ch1 = pairs1
        self.pulse_pairs_ch2 = pairs2
        
        peaks_1 = pairs1[:, 0] if len(pairs1) else []
        peaks_2 = pairs2[:, 0] if len(pairs2) else []
        
        self.axes[0].plot(time_axis, ch1, label="Ch 1")
        self.axes[0].plot(time_axis[peaks_1], ch1[peaks_1], "rx", label="Peaks")
        self.axes[0].set_title(f"Ch 1 with Detected Pulses - ID: {rec_id}")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].legend()

        self.axes[1].plot(time_axis, ch2, label="Ch 2")
        self.axes[1].plot(time_axis[peaks_2], ch2[peaks_2], "rx", label="Peaks")
        self.axes[1].set_title("Ch 2 with Detected Pulses")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].legend()
        
        # Instantaneous pulse rates (shared plot)
        self.axes[2].clear()
        self.axes[2].set_title("Instantaneous Pulse Rates")
        self.axes[2].set_ylabel("Rate (Hz)")
        self.axes[2].set_xlabel("Time (s)")
        
        if len(pairs1) > 1:
            t1 = (pairs1[:, 0] + pairs1[:, 1]) / 2 / sample_rate
            dt1 = np.diff(t1)
            r1 = 1 / dt1
            mid1 = (t1[:-1] + t1[1:]) / 2
            self.axes[2].plot(mid1, r1, 'o', label="Ch 1")
        
        if len(pairs2) > 1:
            t2 = (pairs2[:, 0] + pairs2[:, 1]) / 2 / sample_rate
            dt2 = np.diff(t2)
            r2 = 1 / dt2
            mid2 = (t2[:-1] + t2[1:]) / 2
            self.axes[2].plot(mid2, r2, 'x', label="Ch 2")
        
        self.axes[2].legend()
        
        # Plot deduplicated combined pulse rate as a dotplot
        self.axes[3].clear()
        self.axes[3].set_title("Combined Unique Pulse Rate")
        self.axes[3].set_ylabel("Rate (Hz)")
        self.axes[3].set_xlabel("Time (s)")
        
        if len(pairs1) or len(pairs2):
            # Combine and deduplicate by pulse center, ±10 sample tolerance
            all_pairs = np.concatenate([pairs1, pairs2]) if len(pairs1) and len(pairs2) else pairs1 if len(pairs1) else pairs2
            centers = np.mean(all_pairs, axis=1).astype(int)
            centers.sort()
            unique_centers = []
            for c in centers:
                if not unique_centers or abs(c - unique_centers[-1]) > 10:
                    unique_centers.append(c)
            unique_times = np.array(unique_centers) / sample_rate
        
            # Bin size and rate per bin
            bin_size = float(self.time_bin_size.get())
            if bin_size > 0 and len(unique_times):
                max_time = unique_times[-1]
                bins = np.arange(0, max_time + bin_size, bin_size)
                counts, _ = np.histogram(unique_times, bins)
                rates = counts / bin_size
                bin_centers = bins[:-1] + bin_size / 2
                self.axes[3].plot(bin_centers, rates, 'o', color='black')

        self.fig.tight_layout()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

    def export_pulse_rate_summary(self):
        if self.data is None or self.log_data is None:
            print("No data loaded.")
            return

        sample_rate = int(self.log_data["Sample Rate"])
        bin_size = float(self.time_bin_size.get())
        total_time = self.data["time_ms"].iloc[-1] / 1000.0
        n_bins = int(np.ceil(total_time / bin_size))
        
        pairs1 = self.pulse_pairs_ch1
        pairs2 = self.pulse_pairs_ch2
        
        if pairs1 is None or pairs2 is None:
            print("Run 'Refresh Plot' before exporting.")
            return

        t1 = (pairs1[:, 0] + pairs1[:, 1]) / 2 / sample_rate if len(pairs1) else np.array([])
        t2 = (pairs2[:, 0] + pairs2[:, 1]) / 2 / sample_rate if len(pairs2) else np.array([])

        def count_per_bin(times):
            return np.histogram(times, bins=np.arange(0, n_bins * bin_size + bin_size, bin_size))[0]

        c1 = count_per_bin(t1)
        c2 = count_per_bin(t2)

        all_pairs = np.concatenate([pairs1, pairs2]) if len(pairs1) and len(pairs2) else pairs1 if len(pairs1) else pairs2
        centers = np.mean(all_pairs, axis=1).astype(int)
        centers.sort()
        unique_centers = []
        for c in centers:
            if not unique_centers or abs(c - unique_centers[-1]) > 10:
                unique_centers.append(c)

        t_combined = np.array(unique_centers) / sample_rate
        combined_rate = count_per_bin(t_combined) / bin_size


        df = pd.DataFrame({
            "TimeStart_s": np.arange(n_bins) * bin_size,
            "Ch1_Pulses": c1,
            "Ch2_Pulses": c2,
            "Combined_Unique_Rate_Hz": combined_rate
        })

        out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"Exported to {out_path}")


if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
