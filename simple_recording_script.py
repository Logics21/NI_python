# -*- coding: utf-8 -*-
"""
National Instruments Analog Input Recorder

This program records one analog input channel on National Instruments devices
and saves the data in a binary file along with metadata in CSV format.

Author: Stefan Mucha_admin
Date: Mon Sep 27 16:54:30 2021
"""

import nidaqmx
from nidaqmx import constants
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Configuration parameters
recording_id = 'test'       # Identifier for the recording
input_channels = 1          # Number of input channels
channel_name = 'ai1'        # Channel name on the NI device
rec_fs = 100000             # Sampling frequency (Hz)
rec_dur = 10                # Recording duration (seconds)
input_samples = rec_fs * rec_dur  # Total number of samples to acquire

# Prompt user to select an output directory
root = tk.Tk()
root.withdraw()  # Hide the root window
output_path = filedialog.askdirectory(title="Select folder for storage of recordings")
if not output_path:
    raise ValueError("No output directory selected. Exiting...")

# Detect available National Instruments DAQ devices
system = nidaqmx.system.System.local()
if not system.devices:
    raise RuntimeError("No NI DAQ device detected. Exiting...")

device_name = system.devices[0].name  # Select the first available device

# Setup DAQ read task
read_task = nidaqmx.Task(new_task_name="in")
read_task.ai_channels.add_ai_voltage_chan(
    physical_channel=f"{device_name}/{channel_name}",
    min_val=-10,
    max_val=10,
    units=constants.VoltageUnits.VOLTS
)
read_task.timing.cfg_samp_clk_timing(
    rate=rec_fs,
    sample_mode=constants.AcquisitionType.FINITE,
    samps_per_chan=input_samples
)

# Generate timestamp for file naming
string_timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Save metadata as a CSV file
metadata = pd.DataFrame({
    'recording_id': [recording_id],
    'timestamp': [string_timestamp],
    'input_channels': [input_channels],
    'sample_rate': [rec_fs],
    'duration': [rec_dur]
})

# Start the data acquisition task
data = read_task.read(input_samples, timeout=(rec_dur + 10))

# Stop and close the read task
read_task.stop()
read_task.close()

# Construct file names for data and metadata
data_fname = f"{recording_id}_{string_timestamp}.bin"
meta_fname = f"{recording_id}_{string_timestamp}.csv"

# Save acquired data as a binary file
data_array = np.array(data, dtype=np.float32)  # Ensure consistent data type
data_array.tofile(f"{output_path}/{data_fname}")

# Save metadata as a CSV file
metadata.to_csv(f"{output_path}/{meta_fname}", index=False, sep=";")

# Print confirmation message
print(f"Saved file: {data_fname}")
