

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
import pandas as pd
import time
# import gc
import threading
import pyarrow.parquet as pq
import pyarrow as pa

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
    
class VoltageContinuousInput(tk.Frame):
    
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("eFish Recorder")
        self.master.geometry("1600x900")
        self.create_widgets()
        self.pack()
        self.run = False
        self.recording_active = False
        self.recording_complete = False

        # Threading buffer & lock
        self.storage_buffer = collections.deque()
        self.buffer_lock = threading.Lock()
        self.writer_thread = None
        self.stop_writing = threading.Event()
        self.plot_job_id = None # attribute to store the plot job ID
        
    def create_widgets(self):
        self.DAQSettingsFrame = DAQSettings(self, title="DAQ Settings")
        self.DAQSettingsFrame.grid(row=0, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)
        
        self.plotSettingsFrame = PlotSettings(self, title="Plot Settings")
        self.plotSettingsFrame.grid(row=1, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)
        
        self.ExperimentSettingsFrame = ExperimentSettings(self, title="Experiment Settings")
        self.ExperimentSettingsFrame.grid(row=2, column=0, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, rowspan=3, column=1, pady=(20,0), ipady=10, sticky="ew")

    def start_acquisition(self):
        # Disable certain controls to avoid re-entry
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

        self.continue_acquisition = True

        self.input_channel1 = f"{self.DAQSettingsFrame.chosen_daq.get()}/{self.DAQSettingsFrame.input_channel1_entry.get()}"
        self.max_voltage = 10
        self.min_voltage = -10
        self.daq_string = f"{self.DAQSettingsFrame.chosen_daq.get()}"

        self.sample_rate = int(self.DAQSettingsFrame.sample_rate_entry.get())
        self.refresh_rate = int(self.plotSettingsFrame.refresh_rate_entry.get())
        self.samples_to_plot = int(float(self.plotSettingsFrame.plot_duration_entry.get()) * self.sample_rate)
        self.spec_min_y = int(self.plotSettingsFrame.spec_min_freq_entry.get())
        self.spec_max_y = int(self.plotSettingsFrame.spec_max_freq_entry.get())
        self.plot_spec = int(self.plotSettingsFrame.plot_spectrogram.get())
        
        self.dt = 1000 / self.sample_rate # time steps
        self.plot_buffer1 = collections.deque(maxlen=self.samples_to_plot)
        
        self.read_task = nidaqmx.Task()
        self.read_task.ai_channels.add_ai_voltage_chan(
            self.input_channel1, min_val=self.min_voltage, max_val=self.max_voltage, 
            units=constants.VoltageUnits.VOLTS)
        self.read_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
        
        # Register callback: call nidaq_callback every (sample_rate/refresh_rate) samples
        # Compute desired sample_interval based on sample_rate and refresh_rate
        sample_interval = int(self.sample_rate / self.refresh_rate)
        # Adjust sample_interval to be a divisor of the buffer size (100000)
        buffer_size = 100000
        desired_interval = sample_interval
        if buffer_size % sample_interval != 0:
            for divisor in range(sample_interval, 0, -1):
                if buffer_size % divisor == 0:
                    sample_interval = divisor
                    break
            print(f"Adjusted sample_interval from {desired_interval} to {sample_interval} for compatibility.")
            
        self.read_task.register_every_n_samples_acquired_into_buffer_event(sample_interval, self.nidaq_callback)
        self.read_task.start()        
        self.schedule_plot_update()

    def nidaq_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        """DAQ callback function invoked every n samples."""
        # Read available samples
        temp_data = self.read_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        self.plot_buffer1.extend(temp_data)
        
        if self.recording_active:
            with self.buffer_lock:
                n_samples = len(temp_data)
                if self.acquired_samples + n_samples == self.samples_to_save:
                    self.recording_complete = True
                elif self.acquired_samples + n_samples > self.samples_to_save:
                    diff = self.acquired_samples + n_samples - self.samples_to_save
                    n_samples -= diff 
                    temp_data = temp_data[:n_samples]
                    self.recording_complete = True
                    
                time_vec = np.arange(self.acquired_samples * self.dt, (self.acquired_samples + n_samples) * self.dt, self.dt)
                self.acquired_samples += n_samples
                formatted_data = list(zip(time_vec, temp_data))                    
                self.storage_buffer.extend(formatted_data)
            
            if self.recording_complete:
                self.read_task.stop()
                self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
                self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
                self.stop_writing.set()

                if self.writer_thread and self.writer_thread.is_alive():
                    self.writer_thread.join()

                self.save_log_file()
                # Schedule message box on the main thread
                self.master.after(0, lambda: messagebox.showinfo("Finished", "Recording complete"))
                # gc.collect()
                self.read_task.start()
                self.recording_active = False

        # # Schedule the plot update on the main thread
        return 0

    def update_plot(self):
        """Update the GUI plots; safely called from the main thread."""
        self.graphDataFrame.ax1.cla()
        self.graphDataFrame.ax1.plot(self.plot_buffer1)
        if self.recording_active:
            self.graphDataFrame.ax1.scatter(self.samples_to_plot, 0, s=200, color='green', clip_on=False)
        self.graphDataFrame.ax1.set_title("Channel 1")
        self.graphDataFrame.ax1.set_ylabel("Voltage")
        self.graphDataFrame.ax1.set_xlabel("Sample")
            
        if self.plot_spec == 1:            
            if len(self.plot_buffer1) > 0:
                self.spec1_dom_f = compute_dominant_frequency(self.plot_buffer1, self.sample_rate, self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax2.cla()
                self.graphDataFrame.ax2.specgram(self.plot_buffer1, Fs=self.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax2.set_ylabel("Frequency")
                self.graphDataFrame.ax2.set_xlabel("Sample")
                self.graphDataFrame.ax2.set_ylim(self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax2.set_title(f"Channel 1 - Dominant Freq: {self.spec1_dom_f:.2f} Hz")
            else:
                self.graphDataFrame.ax2.cla()
                self.graphDataFrame.ax2.set_title("No data")
                
        self.graphDataFrame.graph.draw()

    def schedule_plot_update(self):
        """Periodically update the plot at the desired refresh rate."""
        self.update_plot()
        interval = int(1000 / self.refresh_rate)  # interval in milliseconds
        
        # store the returned job ID from the after() call
        self.plot_job_id = self.master.after(interval, self.schedule_plot_update)
        
    def start_record(self):
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        self.filepath = filedialog.asksaveasfilename(defaultextension=".parquet", filetypes=[("Parquet files", "*.parquet")])
        if not self.filepath:
            self.ExperimentSettingsFrame.record_button['state'] = 'enabled'
            return
        else:
            self.rec_dur = float(self.ExperimentSettingsFrame.rec_dur_entry.get())
            self.samples_to_save = int(self.rec_dur * self.sample_rate)
            self.stop_writing.clear()
            self.writer_thread = threading.Thread(target=self.write_data_to_file, daemon=True)
            self.writer_thread.start()

            # self.t_start = time.time()
            self.acquired_samples = 0
            self.recording_active = True
            self.recording_complete = False

    def write_data_to_file(self):
        schema = pa.schema([("time_ms", pa.float64()), ("ch1", pa.float64())])
        with pq.ParquetWriter(self.filepath, schema) as writer:
            while not self.stop_writing.is_set():
                time.sleep(1)
                with self.buffer_lock:
                    if len(self.storage_buffer) == 0:
                        continue
                    data_chunk = list(self.storage_buffer)
                    self.storage_buffer.clear()
                df = pd.DataFrame(data_chunk, columns=["time_ms", "ch1"])
                table = pa.Table.from_pandas(df, schema=schema)
                writer.write_table(table)
        print("Data writing thread stopped.")
            
    def stop_acquisition(self):
        self.continue_acquisition = False
        
        # cancel the scheduled plot update callback so that no lingering callbacks remain
        if self.plot_job_id is not None:
            self.master.after_cancel(self.plot_job_id)
            self.plot_job_id = None
        
        if hasattr(self, 'read_task'):
            try:
                self.read_task.stop()
                self.read_task.close()
            except Exception as e:
                print("Error stopping DAQ task:", e)
                
        self.ExperimentSettingsFrame.reset_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.connect_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.close_button['state'] = 'enabled'
        self.ExperimentSettingsFrame.disconnect_button['state'] = 'disabled'
        self.ExperimentSettingsFrame.record_button['state'] = 'disabled'
        self.DAQSettingsFrame.daq_selection_menu.state(['!disabled'])
        self.DAQSettingsFrame.input_channel1_entry.state(['!disabled'])
        self.DAQSettingsFrame.sample_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.plot_duration_entry.state(['!disabled'])
        self.plotSettingsFrame.refresh_rate_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_min_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_max_freq_entry.state(['!disabled'])
        self.plotSettingsFrame.spec_plot_check.state(['!disabled'])

    def save_log_file(self):
        log_filename = f"log_{self.filepath.split('/')[-1].split('.')[0]}.txt"
        log_filepath = "/".join(self.filepath.split('/')[:-1]) + "/" + log_filename
        log_data = {
            "Number of Input Channels": 1,
            "Sample Rate": self.sample_rate,
            "Recording ID": self.ExperimentSettingsFrame.id_entry.get(),
            "Recording Duration": self.rec_dur,
        }
        with open(log_filepath, 'w') as log_file:
            for key, value in log_data.items():
                log_file.write(f"{key}: {value}\n")

    def reset_device(self):
        daq = nidaqmx.system.Device(self.DAQSettingsFrame.chosen_daq.get())
        daq.reset_device()


class DAQSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
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
        self.sample_rate_label.grid(row=3, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.sample_rate_entry = ttk.Entry(self, width=10)
        self.sample_rate_entry.insert(0, "20000")
        self.sample_rate_entry.grid(row=3, column=1, sticky='w', padx=self.x_padding)


class PlotSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()
    def create_widgets(self):
        self.plot_spectrogram = tk.IntVar()
        self.plot_spectrogram.set(1)
        self.plot_duration_label = ttk.Label(self, text="Plot duration")
        self.plot_duration_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
        self.plot_duration_entry = ttk.Entry(self, width=10)
        self.plot_duration_entry.insert(0, "1")
        self.plot_duration_entry.grid(row=0, column=1, sticky='w', padx=self.x_padding)
        self.refresh_rate_label = ttk.Label(self, text="Refresh Rate")
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
        self.spec_plot_check.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10,0))


class ExperimentSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
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
        self.connect_button = ttk.Button(self, text="Start", command=self.parent.start_acquisition)
        self.connect_button.grid(row=2, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.disconnect_button = ttk.Button(self, text="Stop", command=self.parent.stop_acquisition)
        self.disconnect_button.grid(row=2, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.record_button = ttk.Button(self, text="Start Recording", command=self.parent.start_record)
        self.record_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.record_button['state'] = 'disabled'
        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.parent.reset_device)
        self.reset_button.grid(row=4, column=0, sticky='ew', padx=self.x_padding, pady=(10, 0))
        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=4, column=1, sticky='ew', padx=self.x_padding, pady=(10, 0))


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


root = tk.Tk()
app = VoltageContinuousInput(root)
app.mainloop()
