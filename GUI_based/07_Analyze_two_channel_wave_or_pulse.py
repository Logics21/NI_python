

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, IntVar, DoubleVar, Entry, Label, Frame, StringVar, OptionMenu
import os
from scipy.signal import detrend, find_peaks

class AnalysisGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")
        
        # GUI variables
        self.threshold = DoubleVar(value=0.00)
        self.min_y = DoubleVar(value=200)
        self.max_y = DoubleVar(value=2000)
        self.nfft = IntVar(value=13)  # Default NFFT exponent
        self.noverlap = IntVar(value=10)  # Default noverlap exponent
        self.pulse_window = DoubleVar(value=0.01)  # Pulse window in seconds (for pulse-type)
        self.analysis_type = StringVar(value="wave")  # "wave" or "pulse"
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
        Label(self.control_frame, text="Analysis Type:").pack(side="left", padx=5, pady=5)
        OptionMenu(self.control_frame, self.analysis_type, "wave", "pulse").pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec min freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.min_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec max freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.max_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="NFFT exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.nfft, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="noverlap exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.noverlap, width=10).pack(side="left", padx=5, pady=5)
        # Additional controls for pulse analysis:
        Label(self.control_frame, text="Threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Pulse Window (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.pulse_window, width=10).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Export Mean Pulse Form", command=self.export_mean_pulse).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)

        # Initialize plot (5 subplots)
        self.fig, self.axes = plt.subplots(5, 1, figsize=(15, 10))
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

    def inst_freq(self, y, fs, zerocross=0):
        """
        Computes instantaneous frequency of input signal `y` based on sampling rate `fs`
        using the threshold value `zerocross`.
        """
        y1 = y[:-1]
        y2 = y[1:]
        zerocross_idx = np.where((y1 <= zerocross) & (y2 > zerocross))[0]
        amp_step = y[zerocross_idx + 1] - y[zerocross_idx]
        amp_frac = (zerocross - y[zerocross_idx]) / amp_step
        y_frac = zerocross_idx + amp_frac
        inst_f = 1.0 / (np.diff(y_frac) / fs)
        tinst_f = np.cumsum(np.diff(y_frac) / fs) + y_frac[0] / fs
        return inst_f, tinst_f

    def compute_dominant_frequency(self, data, sample_rate, min_freq, max_freq):
        """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
        fft_result = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
        valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
        filtered_frequencies = frequencies[valid_indices]
        filtered_fft_result = np.abs(fft_result[valid_indices])
        dominant_freq = filtered_frequencies[np.argmax(filtered_fft_result)]
        return dominant_freq
    
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
        min_freq = self.min_y.get()
        max_freq = self.max_y.get()
        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate
        
        # Detrend channels
        channel_1 = detrend(self.data["ch1"])
        channel_2 = detrend(self.data["ch2"])
        
        analysis = self.analysis_type.get()
        
        if analysis == "wave":
            # Wave-type analysis (existing behavior)
            max_y_val = max(np.max(channel_1), np.max(channel_2))
            self.axes[0].plot(time_axis, channel_1, label="Ch 1")
            self.axes[0].plot(time_axis, channel_2 + 2*max_y_val, label="Ch 2")
            self.axes[0].set_title(f"Raw Data - Recording ID: {rec_id}")
            self.axes[0].set_ylabel("Amplitude")
            self.axes[0].legend(loc="lower left")
            
            # Instantaneous frequency plots
            inst_freq1, inst_time1 = self.inst_freq(channel_1, sample_rate, self.threshold.get())
            self.axes[1].plot(inst_time1, inst_freq1, '.')
            self.axes[1].set_title("Channel 1 Instantaneous Frequency")
            self.axes[1].set_ylabel("Frequency (Hz)")
            self.axes[1].set_xlabel("Time (s)")
            self.axes[1].set_ylim(min_freq, max_freq)
            
            inst_freq2, inst_time2 = self.inst_freq(channel_2, sample_rate, self.threshold.get())
            self.axes[2].plot(inst_time2, inst_freq2, '.')
            self.axes[2].set_title("Channel 2 Instantaneous Frequency")
            self.axes[2].set_ylabel("Frequency (Hz)")
            self.axes[2].set_xlabel("Time (s)")
            self.axes[2].set_ylim(min_freq, max_freq)
            
            # Spectrograms
            nfft_value = 2**self.nfft.get()
            noverlap_value = 2**self.noverlap.get()
            hanning_window = np.hanning(nfft_value)
            self.axes[3].specgram(channel_1, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window)
            self.axes[3].set_title("Channel 1 Spectrogram")
            self.axes[3].set_ylabel("Frequency (Hz)")
            self.axes[3].set_ylim(min_freq, max_freq)
            
            self.axes[4].specgram(channel_2, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window)
            self.axes[4].set_title("Channel 2 Spectrogram")
            self.axes[4].set_ylabel("Frequency (Hz)")
            self.axes[4].set_ylim(min_freq, max_freq)
            
        elif analysis == "pulse":
            # Pulse-type analysis
            # Use channel_1 for pulse detection
            peaks, properties = find_peaks(channel_1, height=self.threshold.get())
            
            # Plot raw data with detected peaks
            self.axes[0].plot(time_axis, channel_1, label="Ch 1")
            self.axes[0].plot(time_axis[peaks], channel_1[peaks], "rx", label="Peaks")
            self.axes[0].set_title(f"Raw Data with Detected Peaks - Recording ID: {rec_id}")
            self.axes[0].set_ylabel("Amplitude")
            self.axes[0].legend(loc="lower left")
            
            # Compute inter-pulse intervals and instantaneous rate
            if len(peaks) > 1:
                peak_times = time_axis[peaks]
                dt = np.diff(peak_times)
                inst_rate = 1.0 / dt
                mid_times = (peak_times[:-1] + peak_times[1:]) / 2
                
                self.axes[1].plot(mid_times, inst_rate, 'o')
                self.axes[1].set_title("Instantaneous Pulse Rate")
                self.axes[1].set_ylabel("Rate (Hz)")
                self.axes[1].set_xlabel("Time (s)")
                
                self.axes[2].plot(mid_times, dt, 'o')
                self.axes[2].set_title("Inter-Pulse Intervals")
                self.axes[2].set_ylabel("Interval (s)")
                self.axes[2].set_xlabel("Time (s)")
            else:
                self.axes[1].text(0.5, 0.5, "Not enough peaks detected", ha="center")
                self.axes[2].text(0.5, 0.5, "Not enough peaks detected", ha="center")
            
            # Extract pulse segments for mean pulse form
            window_samples = int(self.pulse_window.get() * sample_rate)
            half_window = window_samples // 2
            segments = []
            for peak in peaks:
                if peak - half_window >= 0 and peak + half_window < len(channel_1):
                    segment = channel_1[peak - half_window: peak + half_window]
                    norm_factor = np.max(np.abs(segment))
                    if norm_factor != 0:
                        segment = segment / norm_factor
                    segments.append(segment)
            if segments:
                segments = np.array(segments)
                self.mean_pulse_form = np.mean(segments, axis=0)
                t_segment = np.linspace(-self.pulse_window.get()/2, self.pulse_window.get()/2, window_samples)
                # Overlay all pulse segments
                for seg in segments:
                    self.axes[3].plot(t_segment, seg, color="gray", alpha=0.5)
                # Plot mean pulse form in red
                self.axes[3].plot(t_segment, self.mean_pulse_form, color="red", linewidth=2, label="Mean Pulse Form")
                self.axes[3].set_title("Pulse Overlays & Mean Pulse Form")
                self.axes[3].set_xlabel("Time (s)")
                self.axes[3].legend(loc="upper right")
            else:
                self.axes[3].text(0.5, 0.5, "No valid pulse segments found", ha="center")
            
            # Turn off the 5th subplot if not used
            self.axes[4].axis('off')
        
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
        
    def export_mean_pulse(self):
        """Export the computed mean pulse form as a CSV file."""
        if self.mean_pulse_form is None:
            print("No mean pulse form computed.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if save_path:
            df = pd.DataFrame(self.mean_pulse_form, columns=["Amplitude"])
            # Optionally add a time column based on pulse window
            window_samples = len(self.mean_pulse_form)
            sample_rate = int(self.log_data["Sample Rate"])
            t = np.linspace(-self.pulse_window.get()/2, self.pulse_window.get()/2, window_samples)
            df.insert(0, "Time (s)", t)
            df.to_csv(save_path, index=False)
            print(f"Mean pulse form exported to {save_path}")

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
