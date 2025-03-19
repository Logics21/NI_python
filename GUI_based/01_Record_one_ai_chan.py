# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 13:43:40 2025

@author: ShuttleBox
"""

# -*- coding: utf-8 -*-
"""
Simple one channel recording program with spectrogram option
Created on Mon Feb 17 16:21:29 2025

@author: Stefan Mucha
"""

import nidaqmx
from nidaqmx import constants
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import collections
import time
import threading
import os


# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names
if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')


def compute_dominant_frequency(data, sample_rate, min_freq, max_freq):
    """Compute the dominant frequency of a quasi-sinusoidal signal within specified bounds."""
    fft_result = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
    valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
    filtered_frequencies = frequencies[valid_indices]
    filtered_fft_result = np.abs(fft_result[valid_indices])
    dominant_freq = filtered_frequencies[np.argmax(filtered_fft_result)]
    return dominant_freq


# Data Acquisition Module
class DataAcquisition:
    def __init__(self, input_channel, sample_rate, min_voltage, max_voltage, refresh_rate, plot_duration):
        self.input_channel = input_channel
        self.sample_rate = sample_rate
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.refresh_rate = refresh_rate
        # Calculate sample_interval based on refresh_rate
        sample_interval = int(self.sample_rate / self.refresh_rate)
        buffer_size = 100000
        if buffer_size % sample_interval != 0:
            for divisor in range(sample_interval, 0, -1):
                if buffer_size % divisor == 0:
                    sample_interval = divisor
                    break
            print(f"Adjusted sample_interval for compatibility: {sample_interval}")
        self.sample_interval = sample_interval

        # Buffers for plotting and file writing
        self.plot_buffer = collections.deque(maxlen=int(plot_duration * self.sample_rate))
        self.storage_buffer = collections.deque()

        # Create and configure DAQ Task
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            self.input_channel, min_val=self.min_voltage, max_val=self.max_voltage,
            units=constants.VoltageUnits.VOLTS)
        self.task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
        self.task.register_every_n_samples_acquired_into_buffer_event(self.sample_interval, self.callback)
        self.running = False

        # Recording attributes
        self.recording_active = False
        self.acquired_samples = 0
        self.samples_to_save = 0
        self.recording_complete_callback = None

    def start(self):
        self.running = True
        self.task.start()

    def stop(self):
        self.running = False
        if self.recording_active:
            self.recording_active = False
            if self.recording_complete_callback is not None:
                self.recording_complete_callback()
        try:
            self.task.stop()
            self.task.close()
        except Exception as e:
            print("Error stopping DAQ task:", e)

    def callback(self, task_handle, event_type, number_of_samples, callback_data):
        if not self.running:
            return 0
        temp_data = self.task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        self.plot_buffer.extend(temp_data)

        if self.recording_active:
            n_samples = len(temp_data)
            if self.acquired_samples + n_samples >= self.samples_to_save:
                diff = (self.acquired_samples + n_samples) - self.samples_to_save
                n_samples = n_samples - diff
                temp_data = temp_data[:n_samples]
                recording_complete = True
            else:
                recording_complete = False

            self.storage_buffer.extend(temp_data)
            self.acquired_samples += n_samples

            if recording_complete:
                self.recording_active = False
                if self.recording_complete_callback is not None:
                    self.recording_complete_callback()


        return 0


class FileWriter(threading.Thread):
    def __init__(self, storage_buffer, buffer_lock, filepath, sample_rate, flush_interval=1):
        super().__init__(daemon=True)
        self.storage_buffer = storage_buffer
        self.buffer_lock = buffer_lock
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self.stop_event = threading.Event()

    def run(self):
        acquired_samples = 0
        dt = 1000 / self.sample_rate
        with open(self.filepath, 'ab') as f:
            while not self.stop_event.is_set():
                time.sleep(self.flush_interval)
                with self.buffer_lock:
                    if not self.storage_buffer:
                        continue
                    data_chunk = np.array(self.storage_buffer, dtype='f8')
                    self.storage_buffer.clear()

                n_samples = len(data_chunk)
                # time_vec = np.arange(acquired_samples * dt, (acquired_samples + n_samples) * dt, dt)
                time_vec = (acquired_samples + np.arange(n_samples)) * dt

                acquired_samples += n_samples

                # Interleave time and data, or store separately depending on your preference
                interleaved = np.column_stack((time_vec, data_chunk)).flatten()

                # Write binary data
                interleaved.tofile(f)

                # Ensure robust flushing
                f.flush()
                os.fsync(f.fileno())

        print("FileWriter stopped.")

    def stop(self):
        self.stop_event.set()


# Plotting Module (Graphical Data Display)
class GraphData(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.graph_title = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(13, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax1.set_title("Channel 1")
        self.ax1.set_ylabel("Voltage")
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.ax2.set_ylabel("Frequency")
        self.ax2.set_xlabel("Sample")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()


# GUI Settings Frames
class DAQSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.x_padding = (10, 10)
        self.create_widgets()

    def create_widgets(self):
        self.chosen_daq = tk.StringVar()
        self.chosen_daq.set(daqList[0])
        self.daq_selection_label = ttk.Label(self, text="Select DAQ")
        self.daq_selection_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.daq_selection_menu = ttk.OptionMenu(self, self.chosen_daq, daqList[0], *daqList)
        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        self.daq_selection_menu.grid(row=0, column=1, sticky='w', padx=self.x_padding)
        self.input_channel_label = ttk.Label(self, text="Input Channel")
        self.input_channel_label.grid(row=1, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.input_channel1_entry = ttk.Entry(self, width=10)
        self.input_channel1_entry.insert(0, "ai5")
        self.input_channel1_entry.grid(row=1, column=1, sticky='w', padx=self.x_padding)
        self.sample_rate_label = ttk.Label(self, text="Sample Rate (Hz)")
        self.sample_rate_label.grid(row=2, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.sample_rate_entry = ttk.Entry(self, width=10)
        self.sample_rate_entry.insert(0, "20000")
        self.sample_rate_entry.grid(row=2, column=1, sticky='w', padx=self.x_padding)


class PlotSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.x_padding = (10, 10)
        self.create_widgets()

    def create_widgets(self):
        self.plot_spectrogram = tk.IntVar()
        self.plot_spectrogram.set(1)
        self.plot_duration_label = ttk.Label(self, text="Plot Duration (s)")
        self.plot_duration_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.plot_duration_entry = ttk.Entry(self, width=10)
        self.plot_duration_entry.insert(0, "1")
        self.plot_duration_entry.grid(row=0, column=1, sticky='w', padx=self.x_padding)
        self.refresh_rate_label = ttk.Label(self, text="Refresh Rate (Hz)")
        self.refresh_rate_label.grid(row=1, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.refresh_rate_entry = ttk.Entry(self, width=10)
        self.refresh_rate_entry.insert(0, "10")
        self.refresh_rate_entry.grid(row=1, column=1, sticky='w', padx=self.x_padding)
        self.spec_min_freq_label = ttk.Label(self, text="Min Freq (Hz)")
        self.spec_min_freq_label.grid(row=2, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.spec_min_freq_entry = ttk.Entry(self, width=10)
        self.spec_min_freq_entry.insert(0, "100")
        self.spec_min_freq_entry.grid(row=2, column=1, sticky='w', padx=self.x_padding)
        self.spec_max_freq_label = ttk.Label(self, text="Max Freq (Hz)")
        self.spec_max_freq_label.grid(row=3, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.spec_max_freq_entry = ttk.Entry(self, width=10)
        self.spec_max_freq_entry.insert(0, "1400")
        self.spec_max_freq_entry.grid(row=3, column=1, sticky='w', padx=self.x_padding)
        self.spec_plot_check = ttk.Checkbutton(self, text='Plot Spectrogram', variable=self.plot_spectrogram)
        self.spec_plot_check.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))


class ExperimentSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.x_padding = (10, 10)
        self.create_widgets()

    def create_widgets(self):
        self.id_label = tk.Label(self, text="Recording ID")
        self.id_label.grid(row=0, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.id_entry = ttk.Entry(self, width=10)
        self.id_entry.grid(row=0, column=1, padx=self.x_padding, pady=5, sticky="w")
        self.rec_dur_label = ttk.Label(self, text="Record Duration (s)")
        self.rec_dur_label.grid(row=1, column=0, padx=self.x_padding, pady=5, sticky="w")
        self.rec_dur_entry = ttk.Entry(self, width=10)
        self.rec_dur_entry.insert(0, "60")
        self.rec_dur_entry.grid(row=1, column=1, padx=self.x_padding, pady=5, sticky="w")
        self.connect_button = ttk.Button(self, text="Start", command=self.master.start_acquisition)
        self.connect_button.grid(row=2, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.disconnect_button = ttk.Button(self, text="Stop", command=self.master.stop_acquisition)
        self.disconnect_button.grid(row=2, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.record_button = ttk.Button(self, text="Start Recording", command=self.master.start_record)
        self.record_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.record_button['state'] = 'disabled'
        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.master.reset_device)
        self.reset_button.grid(row=4, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=4, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))


# Main GUI Module
class DataAcquisitionGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("eFish Recorder")
        self.master.geometry("1600x900")
        self.create_widgets()
        self.pack()

        self.acq = None
        self.file_writer = None
        self.buffer_lock = threading.Lock()
        self.plot_job_id = None
        self.recording_active = False
        self.recording_complete = False

    def create_widgets(self):
        self.DAQSettingsFrame = DAQSettings(self, title="DAQ Settings")
        self.DAQSettingsFrame.grid(row=0, column=0, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)
        self.plotSettingsFrame = PlotSettings(self, title="Plot Settings")
        self.plotSettingsFrame.grid(row=1, column=0, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)
        self.ExperimentSettingsFrame = ExperimentSettings(self, title="Experiment Settings")
        self.ExperimentSettingsFrame.grid(row=2, column=0, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)
        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, rowspan=3, column=1, pady=(20, 0), ipady=10, sticky="ew")

    def start_acquisition(self):
        # Disable controls
        self.ExperimentSettingsFrame.connect_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.disconnect_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.reset_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.close_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
        self.DAQSettingsFrame.daq_selection_menu.state(['disabled'])
        self.DAQSettingsFrame.input_channel1_entry.state(['disabled'])
        self.DAQSettingsFrame.sample_rate_entry.state(['disabled'])
        self.plotSettingsFrame.plot_duration_entry.state(['disabled'])
        self.plotSettingsFrame.refresh_rate_entry.state(['disabled'])
        self.plotSettingsFrame.spec_min_freq_entry.state(['disabled'])
        self.plotSettingsFrame.spec_max_freq_entry.state(['disabled'])
        self.plotSettingsFrame.spec_plot_check.state(['disabled'])

        # Get parameters
        input_channel = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel1_entry.get()}"
        sample_rate = int(self.DAQSettingsFrame.sample_rate_entry.get())
        refresh_rate = int(self.plotSettingsFrame.refresh_rate_entry.get())
        plot_duration = float(self.plotSettingsFrame.plot_duration_entry.get())
        # Instantiate Data Acquisition module
        self.acq = DataAcquisition(input_channel, sample_rate, -10, 10, refresh_rate, plot_duration)
        self.acq.plot_buffer = collections.deque(maxlen=int(plot_duration * sample_rate))
        self.acq.storage_buffer = collections.deque()
        self.acq.start()
        self.schedule_plot_update()

    def update_plot(self):
        self.graphDataFrame.ax1.cla()
        if len(self.acq.plot_buffer) > 0:
            self.graphDataFrame.ax1.plot(list(self.acq.plot_buffer))
        if self.acq.recording_active:
            # Plot a green dot at the right edge of the plot (using the buffer’s maximum length)
            self.graphDataFrame.ax1.scatter(self.acq.plot_buffer.maxlen, 0, s=200, color='green', clip_on=False)
        self.graphDataFrame.ax1.set_title("Channel 1")
        self.graphDataFrame.ax1.set_ylabel("Voltage")
        self.graphDataFrame.ax1.set_xlabel("Sample")

        if int(self.plotSettingsFrame.plot_spectrogram.get()) == 1:
            self.graphDataFrame.ax2.cla()
            if len(self.acq.plot_buffer) > 0:
                data = list(self.acq.plot_buffer)
                spec_dom = compute_dominant_frequency(data, self.acq.sample_rate,
                                                       int(self.plotSettingsFrame.spec_min_freq_entry.get()),
                                                       int(self.plotSettingsFrame.spec_max_freq_entry.get()))
                self.graphDataFrame.ax2.specgram(data, Fs=self.acq.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax2.set_ylabel("Frequency")
                self.graphDataFrame.ax2.set_xlabel("Sample")
                self.graphDataFrame.ax2.set_ylim(int(self.plotSettingsFrame.spec_min_freq_entry.get()),
                                                  int(self.plotSettingsFrame.spec_max_freq_entry.get()))
                self.graphDataFrame.ax2.set_title(f"Channel 1 - Dominant Freq: {spec_dom:.2f} Hz")
            else:
                self.graphDataFrame.ax2.set_title("No data")
        self.graphDataFrame.graph.draw()

    def schedule_plot_update(self):
        self.update_plot()
        interval = int(1000 / int(self.plotSettingsFrame.refresh_rate_entry.get()))
        self.plot_job_id = self.master.after(interval, self.schedule_plot_update)

    def save_log_file(self):
        log_filename = f"log_{self.record_filepath.split('/')[-1].split('.')[0]}.txt"
        log_filepath = "/".join(self.record_filepath.split('/')[:-1]) + "/" + log_filename
        log_data = {
            "Number of Input Channels": 1,
            "Sample Rate": self.acq.sample_rate,
            "Recording ID": self.ExperimentSettingsFrame.id_entry.get(),
            "Recording Duration": self.ExperimentSettingsFrame.rec_dur_entry.get(),
        }
        with open(log_filepath, 'w') as log_file:
            for key, value in log_data.items():
                log_file.write(f"{key}: {value}\n")

    def start_record(self):
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        filepath = filedialog.asksaveasfilename(defaultextension=".bin",
                                                filetypes=[("Binary files", "*.bin")])
        if not filepath:
            self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
            return

        self.record_filepath = filepath
        self.record_duration = float(self.ExperimentSettingsFrame.rec_dur_entry.get())
        self.acq.samples_to_save = int(self.record_duration * self.acq.sample_rate)
        self.acq.acquired_samples = 0  # Reset sample count
        self.acq.storage_buffer.clear()
        
        # Save log file
        self.save_log_file()

        # Set the recording-complete callback
        self.acq.recording_complete_callback = lambda: self.master.after(0, self.on_recording_complete)
        self.acq.recording_active = True

        # Start file writing module
        self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, filepath, self.acq.sample_rate)
        self.file_writer.start()

    def on_recording_complete(self):
        self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.close_button['state'] = 'enabled'

        messagebox.showinfo("Finished", "Recording complete")
        if self.file_writer is not None:
            self.file_writer.stop()
            self.file_writer.join()

    def stop_acquisition(self):
        if self.acq:
            self.acq.stop()
        if self.plot_job_id:
            self.master.after_cancel(self.plot_job_id)
            self.plot_job_id = None
        self.ExperimentSettingsFrame.reset_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.connect_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.disconnect_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
        self.DAQSettingsFrame.daq_selection_menu.state(['!disabled'])
        self.DAQSettingsFrame.input_channel1_entry.state(['!disabled'])
        self.DAQSettingsFrame.sample_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.plot_duration_entry.state(['!disabled'])
        self.plotSettingsFrame.refresh_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_min_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_max_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_plot_check.state(['!disabled'])

    def reset_device(self):
        daq = nidaqmx.system.Device(self.DAQSettingsFrame.chosen_daq.get())
        daq.reset_device()


root = tk.Tk()
app = DataAcquisitionGUI(root)
app.mainloop()
