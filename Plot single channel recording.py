# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:50:15 2023

@author: ShuttleBox
"""

import tkinter
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import signal


# basic Tkinter settings
root = tkinter.Tk()
root.withdraw()

# Pick file with EOD recording
fname = filedialog.askopenfilename(title = "Select File wir EOD Recordings", filetypes = (("Recording", "*.csv"), ("All files", "*")))

# Read EOD data
df = pd.read_csv(fname, sep=";", decimal=",")

# Recording settings
fs = 20000

f, t, Sxx = signal.spectrogram(df['ch 0'], fs, nperseg = 2**12)


# Create plot
fig = plt.figure(figsize = (7,7))
voltage = fig.add_subplot(2,1,1)
voltage.set_title("Voltage")
voltage.plot(df['Time [ms]']/1000, df['ch 0'])
voltage.set_ylabel('Voltage [V]')
voltage.set_xlabel('Time [sec]')

spec = fig.add_subplot(2,1,2)
spec.pcolormesh(t, f, Sxx, shading='gouraud')
spec.set_ylabel('Frequency [Hz]')
spec.set_xlabel('Time [sec]')

# spec = fig.add_subplot(2,1,2)
# spec.specgram(df['ch 0'], Fs=fs, NFFT=2**12)
# spec.set_ylim(100,1000)
# spec.set_xlabel('Sample')
fig.show()




