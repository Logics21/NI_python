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
import itertools

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names

if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')

class voltageContinuousInput(tk.Frame):
    
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #Configure root tk class
        self.master = master
        self.master.title("Analog Voltage Data Acquisition")
        self.master.geometry("1100x800")

        self.create_widgets()
        self.pack()
        self.run = False

    def create_widgets(self):
        #The main frame is made up of three subframes
        self.channelSettingsFrame = channelSettings(self, title ="Channels Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20,0), ipady=10)


    def startTask(self):
        # global valBuffer1
        
        #Prevent user from starting task a second time
        self.inputSettingsFrame.startButton['state'] = 'disabled'
        self.inputSettingsFrame.saveButton['state'] = 'disabled'
        self.inputSettingsFrame.resetButton['state'] = 'disabled'
        self.inputSettingsFrame.closeButton['state'] = 'disabled'

        #Shared flag to alert task if it should stop
        self.continueRunning = True

        #Get task settings from the user
        physicalChannel1 = self.channelSettingsFrame.chosenDaq.get() +'/'+self.channelSettingsFrame.physicalChannel1Entry.get()
        physicalChannel2 = self.channelSettingsFrame.chosenDaq.get() +'/'+self.channelSettingsFrame.physicalChannel2Entry.get()
        maxVoltage = int(self.channelSettingsFrame.maxVoltageEntry.get())
        minVoltage = int(self.channelSettingsFrame.minVoltageEntry.get())
        self.sampleRate = int(self.inputSettingsFrame.sampleRateEntry.get())
        self.refreshRate = int(self.inputSettingsFrame.refreshRateEntry.get()) #Number of samples to be stored and plotted
        self.samplesToPlot = int(float(self.inputSettingsFrame.plotDurationEntry.get())*self.sampleRate)
        self.numberOfSamples = int(self.inputSettingsFrame.recordingDurationEntry.get())*self.sampleRate #Number of samples to be stored and plotted
        self.valBuffer1 = collections.deque(maxlen=self.numberOfSamples) #Circular buffer for read analog input values
        self.valBuffer2 = collections.deque(maxlen=self.numberOfSamples) #Circular buffer for read analog input values
        
        
        
        #Create and start task
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(physicalChannel1, min_val=minVoltage, max_val=maxVoltage)
        self.task.ai_channels.add_ai_voltage_chan(physicalChannel2, min_val=minVoltage, max_val=maxVoltage)
        self.task.timing.cfg_samp_clk_timing(self.sampleRate,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,samps_per_chan=self.numberOfSamples*3)
        self.task.start()

        #spin off call to check 
        self.master.after(10, self.runTask)

    def runTask(self):
        
        # Check if task needs to update the data queue and plot
        self.samplesAvailable = self.task._in_stream.avail_samp_per_chan
        
        if(self.samplesAvailable >= int(self.sampleRate/self.refreshRate)):
            self.tempData = self.task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            self.valBuffer1.extend(self.tempData[0])
            self.valBuffer2.extend(self.tempData[1])
            
            if(len(self.valBuffer1) >= self.samplesToPlot):
                self.graphDataFrame.ax1.cla()
                self.graphDataFrame.ax1.set_title("Channel 1")
                self.graphDataFrame.ax1.plot(list(itertools.islice(self.valBuffer1, len(self.valBuffer1)-self.samplesToPlot, len(self.valBuffer1))))
                self.graphDataFrame.ax2.cla()
                self.graphDataFrame.ax2.set_title("Channel 2")
                self.graphDataFrame.ax2.plot(list(itertools.islice(self.valBuffer2, int(len(self.valBuffer2)-self.samplesToPlot), int(len(self.valBuffer2)))), 'r')
                self.graphDataFrame.graph.draw()
            else:
                self.graphDataFrame.ax1.cla()
                self.graphDataFrame.ax1.set_title("Channel 1")
                self.graphDataFrame.ax1.plot(self.valBuffer1)
                self.graphDataFrame.ax2.cla()
                self.graphDataFrame.ax2.set_title("Channel 2")
                self.graphDataFrame.ax2.plot(self.valBuffer2, "r")
                self.graphDataFrame.graph.draw()
                            
                
            # Alternative plot code, seems like no performance difference
            
            # self.graphDataFrame.ax1.cla()
            # self.graphDataFrame.ax1.set_title("Channel 1")
            # self.graphDataFrame.ax1.plot(self.valBuffer1)
            # self.graphDataFrame.ax2.cla()
            # self.graphDataFrame.ax2.set_title("Channel 2")
            # self.graphDataFrame.ax2.plot(self.valBuffer2, 'r')
            
            # if(len(self.valBuffer1) >= self.samplesToPlot):
            #     self.graphDataFrame.ax1.set_xlim(len(self.valBuffer1)-self.samplesToPlot, len(self.valBuffer1))
            #     self.graphDataFrame.ax2.set_xlim(len(self.valBuffer2)-self.samplesToPlot, len(self.valBuffer2))
    
            # self.graphDataFrame.graph.draw()


        #check if the task should sleep or stop
        if(self.continueRunning):
            self.master.after(10, self.runTask)
            
        else:
            self.task.stop()
            self.task.close()
            self.inputSettingsFrame.startButton['state'] = 'enabled'
            self.inputSettingsFrame.saveButton['state'] = 'enabled'
            self.inputSettingsFrame.resetButton['state'] = 'enabled'
            self.inputSettingsFrame.closeButton['state'] = 'enabled'


    def stopTask(self):
        #call back for the "stop task" button
        self.continueRunning = False
        
    def saveData(self):
        # Save data to csv file
        filepath = filedialog.asksaveasfilename(filetypes = (("csv file", "*.csv"),("all files", "*.*")))
        self.timeVec = np.arange(0, (len(self.valBuffer1))/(self.sampleRate/1000), 1000/self.sampleRate)
        self.dataOutput = np.vstack((np.array(self.timeVec), np.array(self.valBuffer1), np.array(self.valBuffer2))).T
        self.dataOutput = pd.DataFrame(self.dataOutput, columns = ["Time [ms]", "ch 0", "ch 1"])
        self.dataOutput.to_csv(filepath, index=None, sep=';', decimal=",", mode='w')
        
    def resetDevice(self):
        self.daq = nidaqmx.system.Device(self.channelSettingsFrame.chosenDaq.get())
        # Reset DAQ
        self.daq.reset_device()


class channelSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30,30)
        self.create_widgets()


    def create_widgets(self):
        
        self.chosenDaq = tk.StringVar()
        self.chosenDaq.set(daqList[0])
        
        self.daqSelectionLabel = ttk.Label(self, text="Select DAQ")
        self.daqSelectionLabel.grid(row=0,sticky='w', padx=self.xPadding, pady=(10,0))
        
        self.daqSelectionMenu = ttk.OptionMenu(self, self.chosenDaq, daqList[0], *daqList)
        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        # self.daqSelectionMenu.config()
        self.daqSelectionMenu.grid(row=1, sticky="ew", padx=self.xPadding)

        self.physicalChannelLabel = ttk.Label(self, text="Physical Channel")
        self.physicalChannelLabel.grid(row=2,sticky='w', padx=self.xPadding, pady=(10,0))

        self.physicalChannel1Entry = ttk.Entry(self)
        self.physicalChannel1Entry.insert(0, "ai0")
        self.physicalChannel1Entry.grid(row=3, sticky="ew", padx=self.xPadding)
        
        self.physicalChannel2Entry = ttk.Entry(self)
        self.physicalChannel2Entry.insert(0, "ai1")
        self.physicalChannel2Entry.grid(row=4, sticky="ew", padx=self.xPadding)

        self.maxVoltageLabel = ttk.Label(self, text="Max Voltage")
        self.maxVoltageLabel.grid(row=5,sticky='w', padx=self.xPadding, pady=(10,0))
        
        self.maxVoltageEntry = ttk.Entry(self)
        self.maxVoltageEntry.insert(0, "10")
        self.maxVoltageEntry.grid(row=6, sticky="ew", padx=self.xPadding)

        self.minVoltageLabel = ttk.Label(self, text="Min Voltage")
        self.minVoltageLabel.grid(row=7,  sticky='w', padx=self.xPadding,pady=(10,0))

        self.minVoltageEntry = ttk.Entry(self)
        self.minVoltageEntry.insert(0, "-10")
        self.minVoltageEntry.grid(row=8, sticky="ew", padx=self.xPadding,pady=(0,10))

class inputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (30,30)
        self.create_widgets()

    def create_widgets(self):
        self.sampleRateLabel = ttk.Label(self, text="Sample Rate")
        self.sampleRateLabel.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.sampleRateEntry = ttk.Entry(self)
        self.sampleRateEntry.insert(0, "10000")
        self.sampleRateEntry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.recordingDurationLabel = ttk.Label(self, text="Duration to Acquire (s)")
        self.recordingDurationLabel.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.recordingDurationEntry = ttk.Entry(self)
        self.recordingDurationEntry.insert(0, "10")
        self.recordingDurationEntry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.plotDurationLabel = ttk.Label(self, text="Duration to Plot (s)")
        self.plotDurationLabel.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        
        self.plotDurationEntry = ttk.Entry(self)
        self.plotDurationEntry.insert(0, "1")
        self.plotDurationEntry.grid(row=5, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.refreshRateLabel = ttk.Label(self, text="Plot Refresh Rate (per second)")
        self.refreshRateLabel.grid(row=6, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.refreshRateEntry = ttk.Entry(self)
        self.refreshRateEntry.insert(0, "10")
        self.refreshRateEntry.grid(row=7, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.startButton = ttk.Button(self, text="Run", command=self.parent.startTask)
        self.startButton.grid(row=8, column=0, sticky='w', padx=self.xPadding, pady=(10,0))

        self.stopButton = ttk.Button(self, text="Stop", command=self.parent.stopTask)
        self.stopButton.grid(row=8, column=1, sticky='e', padx=self.xPadding, pady=(10,0))
        
        self.saveButton = ttk.Button(self, text="Save Data", command=self.parent.saveData)
        self.saveButton.grid(row=9, column=0, columnspan=2, sticky='ew', padx=self.xPadding, pady=(10,0))
        
        self.resetButton = ttk.Button(self, text="Reset DAQ", command=self.parent.resetDevice)
        self.resetButton.grid(row=10, column=0, columnspan=2, sticky='ew', padx=self.xPadding, pady=(10,0))
        
        self.closeButton = ttk.Button(self, text="Close", command=root.destroy)
        self.closeButton.grid(row=11, column=0, columnspan=2, sticky='ew', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(7,7), dpi=100)
        self.ax1 = self.fig.add_subplot(2,1,1)
        self.ax1.set_title("Channel 1")
        self.ax2 = self.fig.add_subplot(2,1,2)
        self.ax2.set_title("Channel 2")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()
        

#Creates the tk class and primary application "voltageContinuousInput"
root = tk.Tk()
app = voltageContinuousInput(root)

#start the application
app.mainloop()