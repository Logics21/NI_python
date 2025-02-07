# -*- coding: utf-8 -*-
"""
Simple two channel recording program with spectrogram option
Created on Thu Jul 13 16:43:45 2023

@author: Stefan Mucha
"""


import nidaqmx
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import collections
import pandas as pd

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names

if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')

class VoltageContinuousInput(tk.Frame):
    
    def __init__(self, master):
        super().__init__(master)

        # Configure root tk class
        self.master = master
        self.master.title("Analog Voltage Data Acquisition")
        self.master.geometry("1400x800")

        self.create_widgets()
        self.pack()
        self.run = False

    def create_widgets(self):
        # The main frame is made up of four subframes
        self.channelSettingsFrame = ChannelSettings(self, title="Channels Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)
        
        self.plotSettingsFrame = PlotSettings(self, title="Plot Settings")
        self.plotSettingsFrame.grid(row=1, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.inputSettingsFrame = InputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=2, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, rowspan=3, column=2, pady=(20,0), ipady=10)

    def start_task(self):
        # Prevent the user from starting the task a second time
        self.inputSettingsFrame.start_button['state'] = 'disabled'
        self.inputSettingsFrame.save_button['state'] = 'disabled'
        self.inputSettingsFrame.reset_button['state'] = 'disabled'
        self.inputSettingsFrame.close_button['state'] = 'disabled'

        # Shared flag to alert task if it should stop
        self.continue_running = True

        # Get task settings from the user
        physical_channel1 = f"{self.channelSettingsFrame.chosen_daq.get()}/{self.channelSettingsFrame.physical_channel1_entry.get()}"
        physical_channel2 = f"{self.channelSettingsFrame.chosen_daq.get()}/{self.channelSettingsFrame.physical_channel2_entry.get()}"
        max_voltage = int(self.channelSettingsFrame.max_voltage_entry.get())
        min_voltage = int(self.channelSettingsFrame.min_voltage_entry.get())
        self.sample_rate = int(self.inputSettingsFrame.sample_rate_entry.get())
        self.refresh_rate = int(self.plotSettingsFrame.refresh_rate_entry.get())
        self.samples_to_plot = int(float(self.plotSettingsFrame.plot_duration_entry.get()) * self.sample_rate)
        self.spec_min_y = int(self.plotSettingsFrame.spec_min_freq_entry.get())
        self.spec_max_y = int(self.plotSettingsFrame.spec_max_freq_entry.get())
        self.plot_spec = int(self.plotSettingsFrame.plot_spectrogram.get())
        self.number_of_samples = int(self.inputSettingsFrame.recording_duration_entry.get()) * self.sample_rate
        self.val_buffer1 = collections.deque(maxlen=self.number_of_samples)
        self.val_buffer2 = collections.deque(maxlen=self.number_of_samples)
        self.plot_buffer1 = collections.deque(maxlen=self.samples_to_plot)   # New, second buffer only for plotting (uses more memory but no if/else necessary later)
        self.plot_buffer2 = collections.deque(maxlen=self.samples_to_plot)   # New, second buffer only for plotting (uses more memory but no if/else necessary later)

        # Create and start task
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(physical_channel1, min_val=min_voltage, max_val=max_voltage)
        self.task.ai_channels.add_ai_voltage_chan(physical_channel2, min_val=min_voltage, max_val=max_voltage)
        self.task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.number_of_samples * 3)
        self.task.start()

        # Spin off call to check
        self.master.after(10, self.run_task)

    def run_task(self):
        # Check if the task needs to update the data queue and plot
        samples_available = self.task._in_stream.avail_samp_per_chan

        if samples_available >= int(self.sample_rate / self.refresh_rate):
            temp_data = self.task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            self.val_buffer1.extend(temp_data[0])
            self.val_buffer2.extend(temp_data[1])
            self.plot_buffer1.extend(temp_data[0])
            self.plot_buffer2.extend(temp_data[1])
            
            self.graphDataFrame.ax1.cla()
            self.graphDataFrame.ax1.plot(self.plot_buffer1)
            self.graphDataFrame.ax1.set_title("Channel 1")
            self.graphDataFrame.ax1.set_ylabel("Voltage")
            self.graphDataFrame.ax2.cla()
            self.graphDataFrame.ax2.plot(self.plot_buffer2, 'r')
            self.graphDataFrame.ax2.set_title("Channel 2")
            self.graphDataFrame.ax2.set_ylabel("Voltage")
            self.graphDataFrame.ax2.set_xlabel("Sample")
            
            # Plot spectrogram only if checkbox is checked
            if self.plot_spec == 1:
                self.graphDataFrame.ax3.cla()
                self.graphDataFrame.ax3.specgram(self.plot_buffer1, Fs=self.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax3.set_ylabel("Frequency")
                self.graphDataFrame.ax3.set_ylim(self.spec_min_y, self.spec_max_y)
                self.graphDataFrame.ax4.cla()
                self.graphDataFrame.ax4.specgram(self.plot_buffer2, Fs=self.sample_rate, NFFT=2**12)
                self.graphDataFrame.ax4.set_ylabel("Frequency")
                self.graphDataFrame.ax4.set_xlabel("Sample")
                self.graphDataFrame.ax4.set_ylim(self.spec_min_y, self.spec_max_y)
                
            self.graphDataFrame.graph.draw()

        # Check if the task should sleep or stop
        if self.continue_running:
            self.master.after(10, self.run_task)
        else:
            self.task.stop()
            self.task.close()
            self.inputSettingsFrame.start_button['state'] = 'enabled'
            self.inputSettingsFrame.save_button['state'] = 'enabled'
            self.inputSettingsFrame.reset_button['state'] = 'enabled'
            self.inputSettingsFrame.close_button['state'] = 'enabled'

    def stop_task(self):
        # Callback for the "stop task" button
        self.continue_running = False

    def save_data(self):
        # Save data to a CSV file
        filepath = filedialog.asksaveasfilename(filetypes=(("csv file", "*.csv"), ("all files", "*.*")))
        time_vec = np.arange(0, len(self.val_buffer1) / (self.sample_rate / 1000), 1000 / self.sample_rate)
        data_output = np.vstack((time_vec, self.val_buffer1, self.val_buffer2)).T
        data_output = pd.DataFrame(data_output, columns=["Time [ms]", "ch 0", "ch 1"])
        data_output.to_csv(filepath, index=None, sep=';', decimal=",", mode='w')

    def reset_device(self):
        daq = nidaqmx.system.Device(self.channelSettingsFrame.chosen_daq.get())
        daq.reset_device()


class ChannelSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        self.chosen_daq = tk.StringVar()
        self.chosen_daq.set(daqList[0])

        self.daq_selection_label = ttk.Label(self, text="Select DAQ")
        self.daq_selection_label.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.daq_selection_menu = ttk.OptionMenu(self, self.chosen_daq, daqList[0], *daqList)
        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        self.daq_selection_menu.grid(row=1, column=0, columnspan=2, sticky="ew", padx=self.x_padding)

        self.physical_channel_label = ttk.Label(self, text="Physical Channel")
        self.physical_channel_label.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.physical_channel1_entry = ttk.Entry(self)
        self.physical_channel1_entry.insert(0, "ai0")
        self.physical_channel1_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=self.x_padding)
        
        self.physical_channel2_entry = ttk.Entry(self)
        self.physical_channel2_entry.insert(0, "ai1")
        self.physical_channel2_entry.grid(row=4, column=0, columnspan=2, sticky="ew", padx=self.x_padding)
        
        self.min_voltage_label = ttk.Label(self, text="Min Voltage")
        self.min_voltage_label.grid(row=5, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.min_voltage_entry = ttk.Entry(self, width=10)
        self.min_voltage_entry.insert(0, "-10")
        self.min_voltage_entry.grid(row=6, column=0, sticky="ew", padx=(30,5))
        
        self.max_voltage_label = ttk.Label(self, text="Max Voltage")
        self.max_voltage_label.grid(row=5, column=1, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.max_voltage_entry = ttk.Entry(self, width=10)
        self.max_voltage_entry.insert(0, "10")
        self.max_voltage_entry.grid(row=6, column=1, sticky="ew", padx=(5,30))


class PlotSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        self.plot_spectrogram = tk.StringVar()
        self.plot_spectrogram.set(0)
        
        self.plot_duration_label = ttk.Label(self, text="Plot duration")
        self.plot_duration_label.grid(row=0, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.plot_duration_entry = ttk.Entry(self, width=15)
        self.plot_duration_entry.insert(0, "1")
        self.plot_duration_entry.grid(row=1, column=0, sticky='ew', padx=(30,5))

        self.refresh_rate_label = ttk.Label(self, text="Refresh Rate")
        self.refresh_rate_label.grid(row=0, column=1, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.refresh_rate_entry = ttk.Entry(self, width=15)
        self.refresh_rate_entry.insert(0, "10")
        self.refresh_rate_entry.grid(row=1, column=1, sticky='ew', padx=(5,30))
        
        self.spec_min_freq_label = ttk.Label(self, text="Min Freq (Hz)")
        self.spec_min_freq_label.grid(row=2, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.spec_min_freq_entry = ttk.Entry(self, width=15)
        self.spec_min_freq_entry.insert(0, "100")
        self.spec_min_freq_entry.grid(row=3, column=0, sticky='ew', padx=(30,5))

        self.spec_max_freq_label = ttk.Label(self, text="Max Freq (Hz)")
        self.spec_max_freq_label.grid(row=2, column=1, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.spec_max_freq_entry = ttk.Entry(self, width=15)
        self.spec_max_freq_entry.insert(0, "2000")
        self.spec_max_freq_entry.grid(row=3, column=1, sticky='ew', padx=(5,30))
        
        self.spec_plot_check = ttk.Checkbutton(self, text='Plot Spectrogram', variable=self.plot_spectrogram)
        self.spec_plot_check.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10,0))

class InputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        self.sample_rate_label = ttk.Label(self, text="Sample Rate (Hz)")
        self.sample_rate_label.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.sample_rate_entry = ttk.Entry(self)
        self.sample_rate_entry.insert(0, "10000")
        self.sample_rate_entry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.x_padding)

        self.recording_duration_label = ttk.Label(self, text="Duration to Acquire (s)")
        self.recording_duration_label.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.recording_duration_entry = ttk.Entry(self)
        self.recording_duration_entry.insert(0, "10")
        self.recording_duration_entry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.x_padding)

        self.start_button = ttk.Button(self, text="Run", command=self.parent.start_task)
        self.start_button.grid(row=8, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.stop_button = ttk.Button(self, text="Stop", command=self.parent.stop_task)
        self.stop_button.grid(row=8, column=1, sticky='e', padx=self.x_padding, pady=(10, 0))

        self.save_button = ttk.Button(self, text="Save Data", command=self.parent.save_data)
        self.save_button.grid(row=9, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))

        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.parent.reset_device)
        self.reset_button.grid(row=10, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))

        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=11, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))


class GraphData(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.graph_title = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title("Channel 1")
        self.ax1.set_ylabel("Voltage")
        self.ax2 = self.fig.add_subplot(2, 2, 3)
        self.ax2.set_title("Channel 2")
        self.ax2.set_ylabel("Voltage")
        self.ax2.set_xlabel("Sample")
        self.ax3 = self.fig.add_subplot(2, 2, 2)
        self.ax3.set_title("Channel 1")
        self.ax3.set_ylabel("Frequency")
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title("Channel 2")
        self.ax4.set_ylabel("Frequency")
        self.ax4.set_xlabel("Sample")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()


# Create the Tkinter class and primary application "VoltageContinuousInput"
root = tk.Tk()
app = VoltageContinuousInput(root)

# Start the application
app.mainloop()
