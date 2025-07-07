import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, IntVar, DoubleVar, Entry, Label, Frame
import os
from scipy.signal import detrend
from datetime import datetime

class AnalysisGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")

        # GUI variables
        self.start_time_s = DoubleVar(value=0.0)
        self.end_time_s = DoubleVar(value=60.0)
        self.threshold = DoubleVar(value=0.00)
        self.min_y = DoubleVar(value=200)
        self.max_y = DoubleVar(value=2000)
        self.nfft = IntVar(value=13)  # Default NFFT
        self.noverlap = IntVar(value=10)  # Default noverlap
        self.y_offset = DoubleVar(value=2.0)  # Default y-offset for raw plot
        self.log_data = None
        self.data = None

        # Create control area
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="top", fill="x")  # Move to top

        # Add controls
        Label(self.control_frame, text="Start (s):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.start_time_s, width=8).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="End (s, 0=EOF):").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.end_time_s, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Threshold:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.threshold, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec min freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.min_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec max freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.max_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="NFFT exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.nfft, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="noverlap exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.noverlap, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Y Offset:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.y_offset, width=8).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)

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

    def inst_freq(self, y, fs, zerocross=0):
        y1 = y[:-1]
        y2 = y[1:]
        zerocross_idx = np.where((y1 <= zerocross) & (y2 > zerocross))[0]
        amp_step = y[zerocross_idx + 1] - y[zerocross_idx]
        amp_frac = (zerocross - y[zerocross_idx]) / amp_step
        y_frac = zerocross_idx + amp_frac
        inst_f = 1.0 / (np.diff(y_frac) / fs)
        tinst_f = np.cumsum(np.diff(y_frac) / fs) + y_frac[0] / fs
        return inst_f, tinst_f

    def refresh_plot(self):
        """Refresh the plots based on current settings."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return

        # Extract key parameters
        sample_rate = int(self.log_data["Sample_Rate"])
        rec_id = self.log_data.get("Recording_ID", "")
        min_freq = self.min_y.get()
        max_freq = self.max_y.get()

        # Determine number of channels
        channel_cols = [col for col in self.data.columns if col.startswith("ch")]
        n_channels = len(channel_cols)
        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate

        # Prepare figure and axes: 1 raw, 1 inst freq, n_channels spectrograms, n_channels PSDs
        n_rows = 2 + 2 * n_channels  # raw + inst_freq + spectrograms + PSDs
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axes = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows), squeeze=False)
        self.axes = self.axes.flatten()

        # Plot raw data with user-defined y-offset
        y_offset_val = self.y_offset.get()
        for i, ch in enumerate(channel_cols):
            channel_data = detrend(self.data[ch])
            offset = i * y_offset_val
            self.axes[0].plot(time_axis, channel_data + offset, label=f"{ch}")
        # self.axes[0].set_title(f"Raw Data (offset) - Recording ID: {rec_id}")
        self.axes[0].set_ylabel("Amplitude (V)")
        self.axes[0].legend(loc="lower left")

        # Plot instantaneous frequency for all channels and annotate median
        threshold = self.threshold.get()
        median_freqs = []
        for i, ch in enumerate(channel_cols):
            channel_data = detrend(self.data[ch])
            instant_freq, instant_time = self.inst_freq(channel_data, sample_rate, threshold)
            self.axes[1].plot(instant_time, instant_freq, '.', label=f"{ch}")
            if len(instant_freq) > 0:
                median_val = np.median(instant_freq)
                median_freqs.append((ch, median_val))
                # Annotate median on the plot near the right edge
                self.axes[1].annotate(f"Median: {median_val:.2f} Hz", 
                    xy=(instant_time[-1], median_val),
                    xytext=(instant_time[-1], median_val),
                    textcoords='data',
                    fontsize=9, color=self.axes[1].lines[-1].get_color(),
                    va='center', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
        # self.axes[1].set_title("Instantaneous Frequency (all channels)")
        self.axes[1].set_ylabel("Inst. Freq. (Hz)")
        # self.axes[1].set_xlabel("Time (s)")
        self.axes[1].set_ylim(min_freq, max_freq)
        self.axes[1].legend(loc="lower left")

        # Plot spectrogram for each channel
        nfft_exp = self.nfft.get()
        nfft_value = 2 ** nfft_exp
        noverlap_exp = self.noverlap.get()
        noverlap_value = 2 ** noverlap_exp
        hanning_window = np.hanning(nfft_value)
        for i, ch in enumerate(channel_cols):
            channel_data = detrend(self.data[ch])
            ax = self.axes[2 + i]
            ax.specgram(
                channel_data, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window
            )
            # ax.set_title(f"Spectrogram: {ch}")
            ax.set_ylabel(f"{ch} Freq. (Hz)")
            ax.set_ylim(min_freq, max_freq)
            # Don't set xlabel for spectrograms anymore since PSDs will be below

        # Plot PSD for each channel
        for i, ch in enumerate(channel_cols):
            channel_data = detrend(self.data[ch])
            ax = self.axes[2 + n_channels + i]
            
            # Calculate PSD using the same parameters as spectrogram
            from scipy.signal import welch
            freqs, psd = welch(channel_data, fs=sample_rate, window=hanning_window, 
                              nperseg=nfft_value, noverlap=noverlap_value)
            
            # Find peak in the frequency range of interest
            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
            if np.any(freq_mask):
                freqs_roi = freqs[freq_mask]
                psd_roi = psd[freq_mask]
                peak_idx = np.argmax(psd_roi)
                peak_freq = freqs_roi[peak_idx]
                peak_power = psd_roi[peak_idx]
                
                # Plot PSD
                ax.plot(freqs, 10 * np.log10(psd))
                ax.axvline(peak_freq, color='red', linestyle='--', alpha=0.7)
                ax.annotate(f'Peak: {peak_freq:.1f} Hz\n{10*np.log10(peak_power):.1f} dB', 
                           xy=(peak_freq, 10*np.log10(peak_power)), 
                           xytext=(peak_freq + (max_freq - min_freq) * 0.1, 10*np.log10(peak_power)),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=8, color='red')
                
                ax.set_xlim(min_freq, max_freq)
                ax.set_ylabel(f"{ch} PSD (dB/Hz)")
                ax.grid(True, alpha=0.3)
                
                # Set xlabel only for the last PSD plot
                if i == len(channel_cols) - 1:
                    ax.set_xlabel("Frequency (Hz)")

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
