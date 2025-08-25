import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

import numpy as np
import collections
import pandas as pd
import subprocess

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

import threading
from threading import Thread


daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names

if len(daqList) == 0:
    raise ValueError('No DAQ detected, check connection.')
       
def read_defaults():
    # Read default settings and filter checked channels
    default_settings = pd.read_excel('default_settings.xlsx')
    default_settings = default_settings[default_settings['Selected Channel'] == 1]
    default_settings = default_settings.reset_index(drop=True)
    
    # Read custom channel settings and filter checked channels
    default_settings_custom = pd.read_excel('default_settings.xlsx', sheet_name='custom_channel_settings')
    default_settings_custom = default_settings_custom[default_settings_custom['Selected Channel'] == 1]
    default_settings_custom = default_settings_custom.reset_index(drop=True)

    combined_settings = pd.concat([default_settings, default_settings_custom], ignore_index=True)

    # Number of channels
    No_chans = len(default_settings)
    No_chans_cust = len(default_settings_custom)

    # Selected DAQ device
    selected_daq = default_settings['DAQ'].iloc[0]
    daq = nidaqmx.system.Device(selected_daq)

    return default_settings, default_settings_custom, combined_settings, No_chans, No_chans_cust, daq, selected_daq

default_settings, default_settings_custom, combined_settings, No_chans, No_chans_cust, daq, selected_daq = read_defaults()
daq.reset_device()
  
def refresh_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()
    frame.create_widgets()
    
class VoltageContinuousInput(tk.Frame):
        
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Analog Voltage Data Acquisition")
        self.master.geometry("1400x825")
        self.create_widgets()
        self.pack()
        self.run = False
        self.continue_running = False

    def create_widgets(self):

        # The main frame is made up of six subframes
        self.channelSettingsFrame = ChannelSettings(self, title="Channels Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(15,0), padx=(10,20), ipady=10)
        
        self.plotSettingsFrame = PlotSettings(self, title="")  # Ensure consistent naming
        self.plotSettingsFrame.grid(row=1, column=1, sticky="ew", pady=(15,0), padx=(10,20), ipady=10)

        self.outputFrame = OutputSettings(self, title="Output")
        self.outputFrame.grid(row=2, column=1, sticky="ew", pady=(15,0), padx=(10,20), ipady=10)

        self.graphDataFrame = GraphData(self)
        self.graphDataFrame.grid(row=0, column=2, rowspan=3, pady=(15,0), padx=(10,20), ipady=10)
        
        self.FuncsFrame = Functions(self, title="Functions")
        self.FuncsFrame.grid(row=3, column=1, sticky="ew", pady=(0,0), padx=(10,20), ipady=10)
                
        self.appsFrame = Apps(self, title="Apps")
        self.appsFrame.grid(row=3, column=2, sticky="ew", pady=(0,0), padx=(10,20), ipady=10)
        
    def start_task(self):
        self.disable_buttons()
        self.get_user_settings()
        self.create_graph()
        self.create_buffers()
        self.create_nidaq_task()
        self.continue_running = True  # Shared flag to alert task if it should stop
        self.master.after(10, self.run_task)       
               
    def disable_buttons(self):
        # Prevent the user from starting the task a second time
        self.outputFrame.start_button['state'] = 'disabled'
        self.outputFrame.clear_button['state'] = 'disabled'
        self.outputFrame.save_button['state'] = 'disabled'
        self.outputFrame.reset_button['state'] = 'disabled'
        self.outputFrame.close_button['state'] = 'disabled'
        
        # disable check down 
        for i in range(0, (No_chans +  No_chans_cust)):
            plot_check_box = getattr(self.plotSettingsFrame, f"plot_check_box{i}")
            plot_check_box.config(state='disabled')   

        # disable drowp down 
        getattr(self.plotSettingsFrame, f"x_axis_selection").config(state='disabled') # disable drowp down
        getattr(self.plotSettingsFrame, f"title_label_entry").config(state='disabled') 
         
    def get_user_settings(self):
            # Get task settings from the user
            self.sample_rate = int(self.plotSettingsFrame.sample_rate_entry.get())
            self.recording_duration = int(self.plotSettingsFrame.recording_duration_entry.get())
            self.number_of_samples =  self.recording_duration * self.sample_rate 
            self.refresh_rate = int(self.plotSettingsFrame.refresh_rate_entry.get())      
            self.selected_x_axis = self.plotSettingsFrame.x_axis_var.get()
            
            if self.selected_x_axis == "Short Duration":
                self.plot_duration = float(self.plotSettingsFrame.plot_duration_entry.get())
            else:
                self.plot_duration = self.recording_duration
            
            self.samples_to_plot = int(self.plot_duration * self.sample_rate)
          
    def create_nidaq_task(self):          
        self.task = nidaqmx.Task()  # Create and start task          
        for index, row in default_settings.iterrows():
            physical_channel = f"{selected_daq}/{row['Channel']}"
            min_voltage = row['Min Voltage']
            max_voltage = row['Max Voltage']
            terminal_config = getattr(TerminalConfiguration, row['Terminal Configuration'].upper())
                
            # Add the channel using the extracted values
            self.task.ai_channels.add_ai_voltage_chan(physical_channel, min_val=min_voltage, max_val=max_voltage, terminal_config=terminal_config)
                
            # Configure the timing of the task
        self.task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.number_of_samples * 3)
        self.task.start()
        
    def create_buffers(self):
        for i in range(0, (No_chans +  No_chans_cust)):
            val_buff_name = f"val_buffer{i}"  # create variables for buffers
            plot_buff_name = f"plot_buffer{i}"  # create variables for plot buffers 
            setattr(self, val_buff_name, collections.deque(maxlen=self.number_of_samples))  # create buffers for data
            setattr(self, plot_buff_name, collections.deque(maxlen=self.samples_to_plot))  # New, second buffer only for plotting  
                 
    def update_buffers(self, temp_data):
        
        buffer_list = []

        #channel buffers
        for index, row in default_settings.iterrows():
            mx = int(row['Mx'])
            b = int(row['b'])
            transformed_buff = np.array(temp_data[index]) * mx + b
            transformed_buff_name = f"transformed_buff{index}"
            setattr(self, transformed_buff_name, transformed_buff)
            buffer_list.append((index, transformed_buff))

        #custom channel buffers
        for index, row in default_settings_custom.iterrows():
            f_chan_name = row['First Channel']
            s_chan_name = row['Second Channel']
            c_operation = row['Operation']
            transformed_buff_name = f"transformed_buff{index + No_chans}"

            # Find the index of f_chan_name and s_chan_name in default_settings
            f_ax = default_settings[default_settings["Channel"] == f_chan_name].index[0]
            s_ax = default_settings[default_settings["Channel"] == s_chan_name].index[0]
            
            # Extract the transformed_buff from the tuples
            f_buff = buffer_list[f_ax][1]
            s_buff = buffer_list[s_ax][1]
            
            if c_operation == 'x':
                transformed_buff = f_buff * s_buff
            elif c_operation == '+':
                transformed_buff = f_buff + s_buff
            elif c_operation == '-':
                transformed_buff = f_buff - s_buff
            elif c_operation == '÷':
                transformed_buff = f_buff / s_buff
                
            setattr(self, transformed_buff_name, transformed_buff)   
            buffer_list.append((index + No_chans, transformed_buff))

        # Update all buffers
        for i, val in buffer_list:
            val_buff_name = f"val_buffer{i}"
            plot_buff_name = f"plot_buffer{i}"
            getattr(self, val_buff_name).extend(val)
            getattr(self, plot_buff_name).extend(val)

    def get_x_data(self, buffer_length):                      
        if self.selected_x_axis == "Samples":                
            x_data = range(buffer_length)
        elif self.selected_x_axis == "Full Duration" or self.selected_x_axis == "Short Duration":
            x_data = np.linspace(0, buffer_length/ self.sample_rate, buffer_length)
        else: 
            self.x_axis_row = default_settings[default_settings['Device Name'] == self.selected_x_axis].index[0]
            plot_buff_name = f"plot_buffer{self.x_axis_row}"
            x_data = getattr(self, plot_buff_name)
        return x_data
    
    def create_graph(self):
        self.ax_count = 0  # Counter for selected checkboxes
        self.unique_ax_count = 0  # Counter for axis
        self.plot_color = ['red', 'green', 'blue', 'black', 'magenta', 'purple', 'orange', 'brown']
        self.axes_plotting = []
        unit_axes_mapping = {}  # Dictionary to map units to their corresponding axes
 
        refresh_frame(self.graphDataFrame)
                
        def handle_plotting(i, plot_buffer_attr, device_name, units):
            plot_check_var = f"plot_check_var{i}"
            if getattr(self.plotSettingsFrame, plot_check_var).get():  # Plot only if selected
                ax_name = f'self.ax{self.unique_ax_count + 1}'  # Create axis
                
                if self.unique_ax_count == 0:
                    ax = self.graphDataFrame.ax1
                else:
                    ax = self.graphDataFrame.ax1.twinx()
                                        
                color = self.plot_color[self.ax_count % len(self.plot_color)]
                self.ax_count += 1

                if units in unit_axes_mapping:  # Check if units already plotted
                    ax = unit_axes_mapping[units]  # Use the existing axis       
                else:
                    unit_axes_mapping[units] = ax  # Map units to the current axis
                    self.unique_ax_count += 1
                    setattr(self.graphDataFrame, ax_name, ax)
   
                self.axes_plotting.append((ax, plot_buffer_attr, color, device_name, units, 'y'))
                                
        for i, row in combined_settings.iterrows():  # Iterate over rows
            plot_buffer_attr = f'plot_buffer{i}'
            device_name = row['Device Name']
            units = row['Units']
            handle_plotting(i, plot_buffer_attr, device_name, units)

        graph_title = self.plotSettingsFrame.title_label_entry.get()
        self.graphDataFrame.ax1.set_title(graph_title)
        
    def plot_data(self):
        
        updated_plots = []

        buffer_length  = 0
        for ax, plot_buffer_attr, color, device_name, units, axis in self.axes_plotting:
            buffer_data = getattr(self, plot_buffer_attr)
            updated_plots.append((ax, buffer_data, color, device_name, units, axis))
            buffer_length = len(buffer_data)

        # Get x_data and batch plot
        if buffer_length > 0:
            x_data = self.get_x_data(buffer_length)

        # Clear all axes before plotting
        self.clear_axes()

        # Collect unique axes based on their name
        unique_ax = {item[0] for item in updated_plots}

        # Iterate over updated_plots and plot on respective axes
        for ax, buffer, color, label, units, axis in updated_plots:
            if len(x_data) > len(buffer):
                x_data = x_data[:len(buffer)]
            ax.plot(x_data, buffer, label=label, color=color)
            ax.set_ylabel(units, color=color)
            ax.tick_params(axis=axis, labelcolor=color)

        # Adjust legends for each unique axis
        for i, ax in enumerate(unique_ax):
            legend_offset = i * 0.15
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.0 - legend_offset))

            if i <= 1:
                continue
            else:
                ax_position = (i - 1) * 50
                ax.spines['right'].set_position(('outward', ax_position))  # Adjust the position of the y-axis

        self.graphDataFrame.ax1.set_xlabel(self.plotSettingsFrame.x_axis_var.get())
        self.graphDataFrame.graph.draw()
                  
    def clear_axes(self):
        for i in range(0, (No_chans +  No_chans_cust)):
            ax = getattr(self.graphDataFrame, f'self.ax{i+1}', None)
            if ax:
                ax.cla()
                    
    def clear_graph(self):
        # Clear the plot axes
        self.clear_axes()
            
        # Create the figure and main axis (ax1)
        self.graphDataFrame.ax1.set_title("Title")
        self.graphDataFrame.ax1.set_ylabel("Units")
        self.graphDataFrame.ax1.set_xlabel("Domain")

        # Redraw the graph
        self.graphDataFrame.graph.draw()

    def update_value_displays(self):
        for i, row in combined_settings.head(3).iterrows(): 
            reading_value_var = f"reading_value_var{i}"    
            transformed_buff_name = f"transformed_buff{i}"
            data_display = getattr(self, transformed_buff_name)             
            channel_display = getattr(self.channelSettingsFrame, reading_value_var)
            channel_display.config(text="{:.4f}".format(data_display[0]))
                      
    def run_task(self):
        # Check if the task needs to update the data queue and plot
        samples_available = self.task._in_stream.avail_samp_per_chan
        samples_per_frame = int(self.sample_rate // self.refresh_rate)
        
        if samples_available >= samples_per_frame:
            temp_data = self.task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            self.update_buffers(temp_data)
            self.plot_data()
            self.update_value_displays()         
        
        if self.continue_running:  # Check if the task should sleep or stop
            self.master.after(10, self.run_task)
        else:
            self.task.close()
            self.reset_device()
 
    def enable_buttons(self):          
        self.outputFrame.start_button['state'] = 'normal'
        self.outputFrame.clear_button['state'] = 'normal'
        self.outputFrame.save_button['state'] = 'enabled'
        self.outputFrame.reset_button['state'] = 'enabled'
        self.outputFrame.close_button['state'] = 'enabled'

        for i, row in combined_settings.iterrows(): # enable boxes for all selected axis
            plot_check_box = getattr(self.plotSettingsFrame, f"plot_check_box{i}")
            plot_check_box.config(state='enabled')
                      
        getattr(self.plotSettingsFrame, "x_axis_selection").config(state='normal')
        getattr(self.plotSettingsFrame, "title_label_entry").config(state='normal')
        
    def stop_task(self):
        # Callback for the "stop task" button
        #self.continue_running = False
         
        for i in range(0, (No_chans +  No_chans_cust)):
            plot_buff_name = f"plot_buffer{i}"
            getattr(self, plot_buff_name).clear()
        
        self.continue_running = False

        self.enable_buttons()

    def save_data(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        time_vec = np.arange(0, len(getattr(self, "val_buffer0")) / (self.sample_rate / 1000), 1000 / self.sample_rate)    
        data_output = {"Time [ms]": time_vec}

        for index, row in default_settings.iterrows():
            val_buff_name = f"val_buffer{index}"
            Device_Units = default_settings.loc[index, 'Units']
            Device_Name = default_settings.loc[index, 'Device Name']
            Label = f"{Device_Name} [{Device_Units}]"  
            data_output[Label] = getattr(self, val_buff_name) # Append to data_output

        data_output = pd.DataFrame(data_output)      
        data_output.to_excel(filepath, index=False)
        
    def reset_device(self):
        daq = nidaqmx.system.Device(selected_daq)
        daq.reset_device()
            
class ChannelSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()
        self.title = title
        
    def create_widgets(self):
        # Labels for column headers
        
        column_headers = ['Channel', 'Readings', 'Units']
        for i, header in enumerate(column_headers):
            label = ttk.Label(self, text=header)
            label.grid(row=2, column=i, sticky='w', padx=self.x_padding, pady=(5, 5))
                        
        self.x_disp_list = combined_settings["Device Name"].tolist()

        # Create widgets dynamically based on default_settings DataFrame
        for i, row in combined_settings.head(3).iterrows():
            # Channel name
            
            channel_name = ttk.Label(self, text=row['Device Name'])
            channel_name.grid(row=i + 3, column=0, columnspan=1, sticky="w", padx=self.x_padding)
            
            # Reading value
            reading_value_var = f"reading_value_var{i}"
            setattr(self, reading_value_var, ttk.Label(self, width=7, text="0.00", borderwidth=1, relief="solid"))
            getattr(self, reading_value_var).grid(row=i + 3, column=1, sticky="w", padx=self.x_padding)

            # Units
            units_label = ttk.Label(self, width=7, text=row['Units'])
            units_label.grid(row=i + 3, column=2, sticky="w", padx=self.x_padding)

        # row 6
        self.settings_edit = ttk.Button(self, text="Edit", command=self.edit_settings, width=7)
        self.settings_edit.grid(row=6, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))
            
    def edit_settings(self):
        # Start the subprocess to run DAQ_Settings.py       
        try:
            subprocess.run(['python', 'DAQ_Settings.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running subprocess: {e}")
        
        else:
            # Subprocess ran successfully
            global default_settings, default_settings_custom, combined_settings, No_chans, No_chans_cust, daq, selected_daq 
            default_settings, default_settings_custom, combined_settings, No_chans, No_chans_cust, daq, selected_daq = read_defaults()
            refresh_frame(self)
            refresh_frame(self.parent.graphDataFrame)
            refresh_frame(self.parent.plotSettingsFrame)
            
class PlotSettings(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.create_widgets()
        
    def create_widgets(self):
                
        # Create a notebook widget
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each tab
        self.plotSettings1 = ttk.Frame(self.notebook)
        self.plotSettings2 = ttk.Frame(self.notebook)
        self.OutputSettings = ttk.Frame(self.notebook)

        # Add frames to notebook as tabs
        self.notebook.add(self.plotSettings1, text='Plot Settings 1')
        self.notebook.add(self.plotSettings2, text='Plot Settings 2')
        self.notebook.add(self.OutputSettings, text='Input Settings')

        # Populate the first tab (plotSettings1)
        self.plot_label = ttk.Label(self.plotSettings1, text="Select Y-Axis")
        self.plot_label.grid(row=0, column=0, columnspan=3, sticky='w', padx=self.x_padding, pady=(5, 0))

        row_index = 0
        col_index = 0

        # First loop to create checkboxes for default_settings
        for i, row in combined_settings.iterrows(): # create check boxes for all selected axis
            plot_check_var = f"plot_check_var{i}" # create name for box variables
            plot_check_box = f"plot_check_box{i}" # create name for box variables

            setattr(self, plot_check_var, tk.BooleanVar(value=False))  # create variables
            setattr(self, plot_check_box, ttk.Checkbutton(self.plotSettings1, variable=getattr(self, plot_check_var), text=row['Channel']))  # create variables
            row_index = 1 if i <= 4 else 2
            getattr(self, plot_check_box).grid(row=row_index, column=i % 5, sticky="w", padx=5, pady=5)
            col_index = i
                   
        #x- axis drop down menu   
        self.X_axis_label = ttk.Label(self.plotSettings1, text="Select X-Axis")
        self.X_axis_label.grid(row=3, column=0, columnspan=3, sticky='w', padx=self.x_padding, pady=(5, 0))

        self.plot_duration_label = ttk.Label(self.plotSettings1, text="Plot duration")
        self.plot_duration_label.grid(row=3, column=3, columnspan=2, sticky='w', padx=(0,0), pady=(10, 0))
        self.plot_duration_label.grid_remove()
        
        self.Non_chan_list = ["Full Duration", "Short Duration", "Samples"] 
        self.chan_list = default_settings["Device Name"].tolist()
        self.x_axis_list = self.Non_chan_list + self.chan_list
        
        self.x_axis_var = tk.StringVar(value="Samples") # default 

        self.x_axis_var.trace('w', self.on_x_axis_selection_change)
        self.x_axis_selection = ttk.OptionMenu(self.plotSettings1, self.x_axis_var, "", *self.x_axis_list)
        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        self.x_axis_selection.grid(row=4, column=0, columnspan=3, sticky="w", padx=(5,0))
        
        self.plot_duration_entry = ttk.Entry(self.plotSettings1, width=15)
        self.plot_duration_entry.insert(0, "1")
        self.plot_duration_entry.grid(row=4, column=3, columnspan=2, sticky='w', padx=(0,0))
        self.plot_duration_entry.grid_remove()
        
        self.title_label = ttk.Label(self.plotSettings1, text="Graph Title")
        self.title_label.grid(row=5, column=0, columnspan=3, sticky='ew', padx=self.x_padding, pady=(5, 0))

        self.title_label_entry = ttk.Entry(self.plotSettings1, width=15)
        self.title_label_entry.grid(row=6, column=0, columnspan=3, sticky='ew', padx=(30, 5))
       
        # Populate the second tab (plotSettings2)
        self.units_list = list(set(combined_settings["Units"].tolist()))

        self.refresh_rate_label = ttk.Label(self.plotSettings2, text="Refresh Rate (Hz)")
        self.refresh_rate_label.grid(row=0, column=0, columnspan=3, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.refresh_rate_entry = ttk.Entry(self.plotSettings2, width=15)
        self.refresh_rate_entry.insert(0, "10")
        self.refresh_rate_entry.grid(row=1, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(0, 10))
               
        column_headers = ['Axis Unit', 'Min', 'Max']
        for i, header in enumerate(column_headers):
            label = ttk.Label(self.plotSettings2, text=header)
            label.grid(row=2, column=i, sticky='w', padx=self.x_padding, pady=(5, 5))

        # Create widgets dynamically based on default_settings DataFrame
        for i, unit_name in enumerate(self.units_list):
            # Channel name
            unit_label = ttk.Label(self.plotSettings2, text=unit_name)
            unit_label.grid(row=i + 3, column=0, columnspan=1, sticky="w", padx=self.x_padding)

            units_min = ttk.Entry(self.plotSettings2, width=7)
            units_min.grid(row=i + 3, column=1, sticky="w", padx=self.x_padding)

            # Units
            units_max = ttk.Entry(self.plotSettings2, width=7)
            units_max.grid(row=i + 3, column=2, sticky="w", padx=self.x_padding)

        # Populate the third tab (Input settings)  
        self.sample_rate_label = ttk.Label(self.OutputSettings, text="Sample Rate (Hz)")
        self.sample_rate_label.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.sample_rate_entry = ttk.Entry(self.OutputSettings)
        self.sample_rate_entry.insert(0, "1000")
        self.sample_rate_entry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.x_padding)

        self.recording_duration_label = ttk.Label(self.OutputSettings, text="Duration to Acquire (s)")
        self.recording_duration_label.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.recording_duration_entry = ttk.Entry(self.OutputSettings)
        self.recording_duration_entry.insert(0, "10")
        self.recording_duration_entry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.x_padding)            
                                   
    def on_x_axis_selection_change(self, *args):
        selected_x_axis = self.x_axis_var.get()

        # Iterate over available channels and update checkbox state accordingly
        for i, row in combined_settings.iterrows():
            plot_check_var = getattr(self, f"plot_check_var{i}")
            plot_check_box = getattr(self, f"plot_check_box{i}")

            device_name = row["Device Name"] 

            if selected_x_axis == device_name:
                plot_check_var.set(False)
                plot_check_box.config(state='disabled')
                self.plot_duration_label.grid_remove()
                self.plot_duration_entry.grid_remove()
            elif selected_x_axis == "Short Duration": 
                plot_check_box.config(state='normal')
                self.plot_duration_label.grid() 
                self.plot_duration_entry.grid()
            else:
                plot_check_box.config(state='normal')
                self.plot_duration_label.grid_remove()
                self.plot_duration_entry.grid_remove()

class OutputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (30, 30)
        self.create_widgets()

    def create_widgets(self):       
        self.start_button = ttk.Button(self, text="Run", command=self.parent.start_task)
        self.start_button.grid(row=8, column=0, sticky='w', padx=self.x_padding, pady=(10, 0))

        self.stop_button = ttk.Button(self, text="Stop", command=self.parent.stop_task)
        self.stop_button.grid(row=8, column=1, sticky='e', padx=self.x_padding, pady=(10, 0))
        
        self.clear_button = ttk.Button(self, text="Clear Graph", command=self.parent.clear_graph)        
        self.clear_button.grid(row=9, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))

        self.save_button = ttk.Button(self, text="Save Data", command=self.parent.save_data)
        self.save_button.grid(row=10, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))

        self.reset_button = ttk.Button(self, text="Reset DAQ", command=self.parent.reset_device)
        self.reset_button.grid(row=11, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))

        self.close_button = ttk.Button(self, text="Close", command=root.destroy)
        self.close_button.grid(row=12, column=0, columnspan=2, sticky='ew', padx=self.x_padding, pady=(10, 0))
    
class GraphData(tk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Graph Data", labelanchor='n')
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.figure.add_subplot(111)
        ax = self.ax1
        setattr(self, "self.ax1", ax)
            
        self.ax1.set_title("Title")
        self.ax1.set_ylabel("Units")
        self.ax1.set_xlabel("Domain")
        
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Adding interactive cursor
        self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=1)
        
        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.grid(row=1, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.graph, self.toolbar_frame)
        self.toolbar.update()

class Functions(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
       
        self.start_gen_button = ttk.Button(self, text="Function Generator", command=self.open_generator)
        self.start_gen_button.grid(row=1, column=1, columnspan=1, sticky="w", padx=(10,5), pady=(3, 0))
    
        self.triggers_button = ttk.Button(self, text="Custom Triggers")
        self.triggers_button.grid(row=1, column=2, columnspan=1, sticky="w", padx=(10,5), pady=(3, 0))
        
    def open_generator(self):
        subprocess.Popen(['python', 'function_generator.py'])
            
class Apps(tk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        
        self.Sweep_button = ttk.Button(self, text="Sweep_processor")
        self.Sweep_button.grid(row=1, column=1, columnspan=1, sticky="w", padx=(10,5), pady=(3, 0))
        
        self.Response_button = ttk.Button(self, text="Response_processor")
        self.Response_button.grid(row=1, column=2, columnspan=1, sticky="w", padx=(10,5), pady=(3, 0))

if __name__ == "__main__":
    root = tk.Tk()
    app = VoltageContinuousInput(master=root)
    
    app.mainloop()