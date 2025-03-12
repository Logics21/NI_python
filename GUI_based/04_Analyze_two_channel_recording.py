import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, filedialog, Button, IntVar, DoubleVar, Entry, Label, Frame #Checkbutton
import os
from scipy.signal import detrend #, hilbert, butter, filtfilt

class AnalysisGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Tool")

        # GUI variables
        self.threshold = DoubleVar(value=0.00)
        self.min_y = DoubleVar(value=200)
        self.max_y = DoubleVar(value=2000)
        self.nfft = IntVar(value=13)  # Default NFFT
        self.noverlap = IntVar(value=10)  # Default noverlap
        self.log_data = None
        self.data = None

        # Create plot area
        self.plot_frame = Frame(self.root)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        # Create control area
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="bottom", fill="x")

        # Add controls
        Button(self.control_frame, text="Load File", command=self.load_file).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec min freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.min_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="Spec max freq:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.max_y, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="NFFT exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.nfft, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="NFFT exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.nfft, width=10).pack(side="left", padx=5, pady=5)
        Label(self.control_frame, text="noverlap exponent:").pack(side="left", padx=5, pady=5)
        Entry(self.control_frame, textvariable=self.noverlap, width=10).pack(side="left", padx=5, pady=5)
        Button(self.control_frame, text="Refresh Plot", command=self.refresh_plot).pack(side="left", padx=5, pady=5)

        # Initialize plot
        self.fig, self.axes = plt.subplots(5, 1, figsize=(15, 10))
        self.canvas = None
        self.toolbar = None

    def load_file(self):
        """Load the log file and data."""
        filepath = filedialog.askopenfilename(title="Select Log File", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.log_data, self.data = self.load_data(filepath)
            self.refresh_plot()

    # def load_data(self, log_filepath):
    #     """Load log file and associated data file."""
    #     with open(log_filepath, 'r') as file:
    #         log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}
    #
    #     # data_filepath = log_data['Path to Datafile']
    #     data_filepath = log_filepath.split('log_')[0]+log_filepath.split('log_')[-1].split('.')[0]+'.feather'
    #
    #     if not os.path.exists(data_filepath):
    #         raise FileNotFoundError(f"Data file not found at: {data_filepath}")
    #
    #     data = pd.read_feather(data_filepath)
    #     return log_data, data

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
        using the threshold value `zerocross` (usually 0 V).

        Parameters:
        - y: np.ndarray
            The input signal.
        - fs: float
            The sampling rate in Hz.
        - zerocross: float, optional
            The threshold value for zero-crossing (default is 0).

        Returns:
        - inst_f: np.ndarray
            Instantaneous frequency.
        - tinst_f: np.ndarray
            Time points for plotting instantaneous frequency.
        """
        y1 = y[:-1]
        y2 = y[1:]

        # Find zero-crossing indices
        zerocross_idx = np.where((y1 <= zerocross) & (y2 > zerocross))[0]

        # Compute fractional zero-crossing positions
        amp_step = y[zerocross_idx + 1] - y[zerocross_idx]  # Amplitude step
        amp_frac = (zerocross - y[zerocross_idx]) / amp_step  # Fraction of step below zero
        y_frac = zerocross_idx + amp_frac  # Adjust zero-crossing indices with fraction

        # Compute instantaneous frequency
        inst_f = 1.0 / (np.diff(y_frac) / fs)  # Instantaneous frequency
        tinst_f = np.cumsum(np.diff(y_frac) / fs) + y_frac[0] / fs  # Time points for plotting

        return inst_f, tinst_f


    def compute_dominant_frequency(self, data, sample_rate, min_freq, max_freq):
        """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
        # Compute FFT and frequency bins
        fft_result = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
        # Apply frequency bounds
        valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
        filtered_frequencies = frequencies[valid_indices]
        filtered_fft_result = np.abs(fft_result[valid_indices])
        # Find the frequency with the maximum FFT amplitude within the bounded range
        dominant_freq = filtered_frequencies[np.argmax(filtered_fft_result)]
        return dominant_freq


    def refresh_plot(self):
        """Refresh the plots based on current settings."""
        if self.log_data is None or self.data is None:
            print("No data loaded. Please load a file first.")
            return

        # Reset zoom or pan mode before refreshing the plot
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.mode = ''  # Reset active toolbar mode (zoom/pan)
            self.toolbar.update()  # Update the toolbar to reflect the change

        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Extract key parameters
        sample_rate = int(self.log_data["Sample Rate"])
        # rec_dur = float(self.log_data["Recording Duration"])
        rec_id = self.log_data["Recording ID"]

        min_freq = self.min_y.get()
        max_freq = self.max_y.get()

        total_samples = len(self.data)
        time_axis = np.arange(total_samples) / sample_rate

        # Detrend raw data
        channel_1 = detrend(self.data["ch1"])
        channel_2 = detrend(self.data["ch2"])

        # Plot raw data
        max_y = max(np.max(channel_1), np.max(channel_2))
        self.axes[0].plot(time_axis, channel_1, label="Ch 1")
        self.axes[0].plot(time_axis, channel_2 + 2*max_y, label="Ch 2")
        self.axes[0].set_title(f"Raw Data - Recording ID: {rec_id}")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].legend(loc="lower left")

        # Plot instantaneous frequency - channel 1
        threshold = self.threshold.get()
        instant_freq, instant_time = self.inst_freq(channel_1, sample_rate, threshold)
        self.axes[1].plot(instant_time, instant_freq,'.')
        self.axes[1].set_title("Channel 1 Instantaneous Frequency")
        self.axes[1].set_ylabel("Frequency (Hz)")
        self.axes[1].set_xlabel("Time (s)")
        self.axes[1].set_ylim(min_freq, max_freq)


        # Plot instantaneous frequency - channel 2
        threshold = self.threshold.get()
        instant_freq, instant_time = self.inst_freq(channel_2, sample_rate, threshold)
        self.axes[2].plot(instant_time, instant_freq,'.')
        self.axes[2].set_title("Channel 2 Instantaneous Frequency")
        self.axes[2].set_ylabel("Frequency (Hz)")
        self.axes[2].set_xlabel("Time (s)")
        self.axes[2].set_ylim(min_freq, max_freq)

        # Plot spectrogram - Channel 1
        nfft_exp = self.nfft.get()
        nfft_value = 2**nfft_exp
        noverlap_exp = self.noverlap.get()
        noverlap_value = 2**noverlap_exp
        # Create a Hanning window
        hanning_window = np.hanning(nfft_value)
        self.axes[3].specgram(
            channel_1, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window
        )
        self.axes[3].set_title("Channel 1 Spectrogram")
        self.axes[3].set_ylabel("Frequency (Hz)")
        self.axes[3].set_ylim(min_freq, max_freq)

        # Plot spectrogram - Channel 2
        nfft_exp = self.nfft.get()
        nfft_value = 2**nfft_exp
        noverlap_exp = self.noverlap.get()
        noverlap_value = 2**noverlap_exp
        # Create a Hanning window
        hanning_window = np.hanning(nfft_value)
        self.axes[4].specgram(
            channel_2, Fs=sample_rate, NFFT=nfft_value, noverlap=noverlap_value, window=hanning_window
        )
        self.axes[4].set_title("Channel 2 Spectrogram")
        self.axes[4].set_ylabel("Frequency (Hz)")
        self.axes[4].set_ylim(min_freq, max_freq)

        # Adjust layout
        self.fig.tight_layout()


        # Update canvas and toolbar
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()

        # Create a new canvas using FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Add the navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

if __name__ == "__main__":
    root = Tk()
    app = AnalysisGUI(root)
    root.mainloop()
