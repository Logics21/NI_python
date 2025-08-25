import tkinter as tk
from tkinter import ttk
import nidaqmx
import pandas as pd
from tkinter import messagebox

class ChannelSettings(tk.LabelFrame):      
    def __init__(self, parent, title):  
        super().__init__(parent, text=title, labelanchor='n')
        self.parent = parent
        self.x_padding = (10, 10)
        self.entry_widgets = {}  # Dictionary to store entry widgets
        self.create_widgets()
        
    def create_widgets(self):
        daqSys = nidaqmx.system.System()
        daqList = daqSys.devices.device_names
        self.chosen_daq = tk.StringVar()
        #self.chosen_daq.set(daqList[0])
        
        self.checkbutton_vars = {}
        self.Terminal_options_list = ["RSE", "NRSE", "DIFFERENTIAL"]
        self.Default_terminal = "RSE"
        
        # DAQ selection
        self.daq_selection_label = ttk.Label(self, text="Select DAQ")
        self.daq_selection_label.grid(row=0, column=0, columnspan=1, sticky='w', padx=self.x_padding, pady=(5, 0))
        
        self.daq_selection_menu = ttk.OptionMenu(self, self.chosen_daq, daqList[0], *daqList)

        s = ttk.Style()
        s.configure("TMenubutton", background="white")
        self.daq_selection_menu.grid(row=1, column=0, columnspan=1, sticky="w", padx=self.x_padding, pady=10)

        # Create a tabbed frame
        self.tab_control = ttk.Notebook(self)
        self.tab_control.grid(row=2, column=0, columnspan=2, padx=self.x_padding)

        # Add tabs for input and output channels
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text='Input Channels')
        
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text='Output Channels')
        
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab3, text='Custom Channels')

        # Add a save button
        self.save_button = ttk.Button(self, text="Save Settings", command=self.save_settings)
        self.save_button.grid(row=3, column=0, columnspan=1, pady=10, sticky="e")

        self.close_button = ttk.Button(self, text="Close", command=self.close_window)
        self.close_button.grid(row=3, column=1, columnspan=1, pady=10, sticky="w")

        self.chosen_daq.trace_add('write', self.update_channels)  # Add a trace to update channels when selection changes
        self.update_channels()  # Initial channel update

    def update_channels(self, *args):
        # Clear existing widgets
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        for widget in self.tab3.winfo_children():
            widget.destroy()

        # Get the selected device
        selected_device_name = self.chosen_daq.get()
        self.device = nidaqmx.system.Device(selected_device_name)

        self.chan_names = []

        for channel in self.device.ai_physical_chans:
            channel_name = channel.name.split('/')[1]
            self.chan_names.append(channel_name)

        # Get a list of physical channels for the device
        physical_ai_channels = self.device.ai_physical_chans
        num_ai_channels = len(self.device.ai_physical_chans)

        physical_ao_channels = self.device.ao_physical_chans
        num_ao_channels = len(self.device.ao_physical_chans)

        # Define widget labels for input channels
        labels = ["Channel", "Min V", "Max V", "Terminal Configuration", "Mx", "b", "Units", "Device Name", "Selected Channel"]
        label_length = [10, 7, 7, 13, 7, 7, 10, 15, 10]  # corrected the typo in 'label_length'

        # Create labels for input channels
        for i, label in enumerate(labels):
            ttk.Label(self.tab1, text=label, width=label_length[i], wraplength=label_length[i]*8).grid(row=0, column=i, padx=5, pady=5)

        # Create entries for each input channel
        for i, channel in enumerate(physical_ai_channels):
            self.entry_widgets[f"Channel_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[0])
            self.entry_widgets[f"Channel_entry{i+1}"].insert(0, channel.name.split('/')[1])
            self.entry_widgets[f"Channel_entry{i+1}"].grid(row=i+1, column=0, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"Channel_entry{i+1}"]['state'] = 'disabled'

            self.entry_widgets[f"min_voltage_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[1])
            self.entry_widgets[f"min_voltage_entry{i+1}"].grid(row=i+1, column=1, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"max_voltage_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[2])
            self.entry_widgets[f"max_voltage_entry{i+1}"].grid(row=i+1, column=2, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"ter_selection_var{i+1}"] = ttk.Combobox(self.tab1, values=self.Terminal_options_list, width=label_length[3])
            self.entry_widgets[f"ter_selection_var{i+1}"].set(self.Default_terminal)
            self.entry_widgets[f"ter_selection_var{i+1}"].grid(row=i+1, column=3, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"mx_plus_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[4])
            self.entry_widgets[f"mx_plus_entry{i+1}"].grid(row=i+1, column=4, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"b_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[5])
            self.entry_widgets[f"b_entry{i+1}"].grid(row=i+1, column=5, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"units_entry{i+1}"] = ttk.Entry(self.tab1, width=label_length[6])
            self.entry_widgets[f"units_entry{i+1}"].grid(row=i+1, column=6, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"Device_name{i+1}"] = ttk.Entry(self.tab1, width=label_length[7])
            self.entry_widgets[f"Device_name{i+1}"].grid(row=i+1, column=7, padx=5, pady=5, sticky='w')

            self.checkbutton_vars[f"Selected Channel{i+1}"] = tk.StringVar()
            self.checkbutton_vars[f"Selected Channel{i+1}"].set(0)
            self.entry_widgets[f"Selected Channel{i+1}"] = ttk.Checkbutton(self.tab1, width=7, variable=self.checkbutton_vars[f"Selected Channel{i+1}"])
            self.entry_widgets[f"Selected Channel{i+1}"].grid(row=i+1, column=8, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"Selected Channel{i+1}"].bind("<Button-1>", self.clear_cust_dropdown)


        
        # Define widget labels for output channels
        labels_ao = ["Channel", "Min V", "Max V", "Terminal configuration", "Device Name", "Selected Channel"]
        
        # Create labels for output channels
        for i, label in enumerate(labels_ao):
            ttk.Label(self.tab2, text=label).grid(row=0, column=i, padx=5, pady=5, sticky='w')

        # Create entries for each output channel
        for i, channel in enumerate(physical_ao_channels):
            
            self.entry_widgets[f"Channel_entry_output{i+1}"] = ttk.Entry(self.tab2, width=7)
            self.entry_widgets[f"Channel_entry_output{i+1}"].insert(0, channel.name.split('/')[1])
            self.entry_widgets[f"Channel_entry_output{i+1}"].grid(row=i+1, column=0, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"Channel_entry_output{i+1}"]['state'] = 'disabled'

            self.entry_widgets[f"min_voltage_entry_output{i+1}"] = ttk.Entry(self.tab2, width=7)
            self.entry_widgets[f"min_voltage_entry_output{i+1}"].grid(row=i+1, column=1, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"max_voltage_entry_output{i+1}"] = ttk.Entry(self.tab2, width=7)
            self.entry_widgets[f"max_voltage_entry_output{i+1}"].grid(row=i+1, column=2, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"ter_selection_var_output{i+1}"] = ttk.Combobox(self.tab2, values=self.Terminal_options_list, width=15)
            self.entry_widgets[f"ter_selection_var_output{i+1}"].set(self.Default_terminal)
            self.entry_widgets[f"ter_selection_var_output{i+1}"].grid(row=i+1, column=3, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"Device_name_output{i+1}"] = ttk.Entry(self.tab2, width=15)
            self.entry_widgets[f"Device_name_output{i+1}"].grid(row=i+1, column=4, padx=5, pady=5, sticky='w')
            
            self.checkbutton_vars[f"Selected Channel_output{i+1}"] = tk.StringVar()
            self.checkbutton_vars[f"Selected Channel_output{i+1}"].set(0)
            self.entry_widgets[f"Selected Channel_output{i+1}"] = ttk.Checkbutton(self.tab2, width=7, variable=self.checkbutton_vars[f"Selected Channel_output{i+1}"])
            self.entry_widgets[f"Selected Channel_output{i+1}"].grid(row=i+1, column=5, padx=5, pady=5, sticky='w')
            

        
        # Define widget labels for Custom channels functions
        labels_co = ['Channel',"First Channel", "Operation", "Second Channel", "Device Name", "Units", "Selected Channel"]
        
        # Create labels for output channels
        for i, label in enumerate(labels_co):
            ttk.Label(self.tab3, text=label).grid(row=0, column=i, padx=5, pady=5) 
              
        self.no_of_cus_chans = 2 # number of custom operation channels to be created can be expanded       
        Opereation_list = ["+", "-", "x", "÷"]
    
 
  
        # Create entries for each custom channel
        for i in range(0, self.no_of_cus_chans):
            
            self.entry_widgets[f"Channel_entry_cust{i+1}"] = ttk.Entry(self.tab3, width=7)
            self.entry_widgets[f"Channel_entry_cust{i+1}"].insert(0, f"cu{i}")
            self.entry_widgets[f"Channel_entry_cust{i+1}"].grid(row=i+1, column=0, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"Channel_entry_cust{i+1}"]['state'] = 'disabled'
            
            
            self.entry_widgets[f"first_chan_sel{i+1}"] = ttk.Combobox(self.tab3, values=self.get_available_channels(), width=15)
            self.entry_widgets[f"first_chan_sel{i+1}"].grid(row=i+1, column=1, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"first_chan_sel{i+1}"].bind("<Button-1>", self.update_chan_sel)
            
            self.entry_widgets[f"operation_sel{i+1}"] = ttk.Combobox(self.tab3, values=Opereation_list, width=15)
            self.entry_widgets[f"operation_sel{i+1}"].grid(row=i+1, column=2, padx=5, pady=5, sticky='w')

            self.entry_widgets[f"sec_chan_sel{i+1}"] = ttk.Combobox(self.tab3, values=self.get_available_channels(), width=15)
            self.entry_widgets[f"sec_chan_sel{i+1}"].grid(row=i+1, column=3, padx=5, pady=5, sticky='w')
            self.entry_widgets[f"sec_chan_sel{i+1}"].bind("<Button-1>", self.update_chan_sel)

            self.entry_widgets[f"custom_chan_name{i+1}"] = ttk.Entry(self.tab3, width=15)
            self.entry_widgets[f"custom_chan_name{i+1}"].grid(row=i+1, column=4, padx=5, pady=5, sticky='w')
            
            self.entry_widgets[f"cust_units_entry{i+1}"] = ttk.Entry(self.tab3, width=7)       
            self.entry_widgets[f"cust_units_entry{i+1}"].grid(row=i+1, column=5, padx=5, pady=5, sticky='w')
            
            self.checkbutton_vars[f"Selected Channel_Custom{i+1}"] = tk.StringVar()
            self.checkbutton_vars[f"Selected Channel_Custom{i+1}"].set(0)
            self.entry_widgets[f"Selected Channel_Custom{i+1}"] = ttk.Checkbutton(self.tab3, width=7, variable=self.checkbutton_vars[f"Selected Channel_Custom{i+1}"])
            self.entry_widgets[f"Selected Channel_Custom{i+1}"].grid(row=i+1, column=6, padx=5, pady=5, sticky='w')
        
        self.read_defaults_from_excel('default_settings.xlsx')
        
        
    def get_available_channels(self):
        available_channels = []
        for i, name in enumerate(self.chan_names):
            if self.checkbutton_vars.get(f"Selected Channel{i+1}", tk.StringVar()).get() == "1":
                available_channels.append(name)     
        return available_channels
    
    def update_chan_sel(self, event):
        combobox = event.widget
        available_channels = self.get_available_channels()
        combobox['values'] = available_channels
            
    def clear_cust_dropdown(self, event):
        for i in range(0, self.no_of_cus_chans):     
            first_chan_sel = self.entry_widgets[f"first_chan_sel{i+1}"]
            sec_chan_sel = self.entry_widgets[f"sec_chan_sel{i+1}"]
            available_channels = self.get_available_channels()

            if first_chan_sel.get() not in available_channels:
                first_chan_sel.set('')

            if sec_chan_sel.get() not in available_channels:
                sec_chan_sel.set('')
        
    
    
    def read_defaults_from_excel(self, filename):
        try:
            df1 = pd.read_excel(filename, sheet_name='default_settings')
            df2 = pd.read_excel(filename, sheet_name='output_settings')
            df3 = pd.read_excel(filename, sheet_name='custom_channel_settings')

            # Set values for rows from 'default_settings' sheet
            for i in range(len(df1)):
                row = df1.iloc[i]
                self.set_entry_values(i + 1, row, "input")

            # Set values for rows from 'output_settings' sheet
            for i in range(len(df2)):
                row = df2.iloc[i]
                self.set_entry_values(i + 1, row, "output")
                
            # Set values for rows from 'custom_channel_settings' sheet
            for i in range(len(df3)):
                row = df3.iloc[i]
                self.set_entry_values(i + 1, row, "custom")
                

        except FileNotFoundError:
            print(f"{filename} not found. Using default settings.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    def set_entry_values(self, index, row, ctype):
        if ctype == "input":
            self.entry_widgets[f"min_voltage_entry{index}"].insert(0, row['Min Voltage'] if not pd.isna(row['Min Voltage']) else '')
            self.entry_widgets[f"max_voltage_entry{index}"].insert(0, row['Max Voltage'] if not pd.isna(row['Max Voltage']) else '')
            self.entry_widgets[f"ter_selection_var{index}"].set(row['Terminal Configuration'] if not pd.isna(row['Terminal Configuration']) else '')
            self.entry_widgets[f"Device_name{index}"].insert(0, row['Device Name'] if not pd.isna(row['Device Name']) else '')
            self.entry_widgets[f"mx_plus_entry{index}"].insert(0, row['Mx'] if not pd.isna(row['Mx']) else '')
            self.entry_widgets[f"b_entry{index}"].insert(0, row['b'] if not pd.isna(row['b']) else '')
            self.entry_widgets[f"units_entry{index}"].insert(0, row['Units'] if not pd.isna(row['Units']) else '')
            self.checkbutton_vars[f"Selected Channel{index}"].set(int(row['Selected Channel']))
            
        elif ctype == "output":
            self.entry_widgets[f"min_voltage_entry_output{index}"].insert(0, row['Min Voltage'])
            self.entry_widgets[f"max_voltage_entry_output{index}"].insert(0, row['Max Voltage'])
            self.entry_widgets[f"ter_selection_var_output{index}"].set(row['Terminal Configuration'])
            self.entry_widgets[f"Device_name_output{index}"].insert(0, row['Device Name'])
            self.checkbutton_vars[f"Selected Channel_output{index}"].set(int(row['Selected Channel']))
            
        elif ctype == "custom":
            self.entry_widgets[f"Channel_entry_cust{index}"].insert(0, row['Channel'] if not pd.isna(row['Channel']) else '')
            self.entry_widgets[f"first_chan_sel{index}"].set(row['First Channel'] if not pd.isna(row['First Channel']) else '')
            self.entry_widgets[f"operation_sel{index}"].set(row['Operation'] if not pd.isna(row['Operation']) else '')
            self.entry_widgets[f"sec_chan_sel{index}"].set(row['Second Channel'] if not pd.isna(row['Second Channel']) else '')
            self.entry_widgets[f"custom_chan_name{index}"].insert(0, row['Device Name'] if not pd.isna(row['Device Name']) else '')
            self.entry_widgets[f"cust_units_entry{index}"].insert(0, row['Units'] if not pd.isna(row['Units']) else '')
            self.checkbutton_vars[f"Selected Channel_Custom{index}"].set(int(row['Selected Channel']))   
        
    def save_settings(self):
        fieldnames = ['Channel','DAQ', 'Min Voltage', 'Max Voltage', 'Terminal Configuration', 'Mx', 'b', 'Units', 'Device Name', 'Selected Channel']
        fieldnames_cus =  ['Channel',"First Channel", "Operation", "Second Channel", "Device Name", "Units", 'Selected Channel']

        # Get a list of physical channels for the device
        physical_ai_channels = self.device.ai_physical_chans
        num_ai_channels = len(self.device.ai_physical_chans)
        physical_ao_channels = self.device.ao_physical_chans
        num_ao_channels = len(self.device.ao_physical_chans)
        
        data1 = []
        data2 = []
        data3 = []

        # Collect data for input channels
        for i in range(1, num_ai_channels + 1):
            channel = self.entry_widgets[f"Channel_entry{i}"].get()
            min_voltage = self.entry_widgets[f"min_voltage_entry{i}"].get()
            max_voltage = self.entry_widgets[f"max_voltage_entry{i}"].get()
            term_config = self.entry_widgets[f"ter_selection_var{i}"].get()
            mx_plus = self.entry_widgets[f"mx_plus_entry{i}"].get()
            b = self.entry_widgets[f"b_entry{i}"].get()
            units = self.entry_widgets[f"units_entry{i}"].get()
            name = self.entry_widgets[f"Device_name{i}"].get()
            selected = int(self.checkbutton_vars[f"Selected Channel{i}"].get())
            
            data1.append({
                'Channel': channel,
                'DAQ': self.chosen_daq.get(),
                'Min Voltage': min_voltage,
                'Max Voltage': max_voltage, 
                'Terminal Configuration': term_config, 
                'Mx': mx_plus,
                'b': b, 
                'Units': units, 
                'Device Name': name,
                'Selected Channel': selected
                })

        # Collect data for output channels
        for i in range(1, num_ao_channels + 1):
            channel = self.entry_widgets[f"Channel_entry_output{i}"].get()
            min_voltage = self.entry_widgets[f"min_voltage_entry_output{i}"].get()
            max_voltage = self.entry_widgets[f"max_voltage_entry_output{i}"].get()
            term_config = self.entry_widgets[f"ter_selection_var_output{i}"].get()
            name = self.entry_widgets[f"Device_name_output{i}"].get()
            selected = int(self.checkbutton_vars[f"Selected Channel_output{i}"].get())
            
            data2.append({
                'Channel': channel, 
                'DAQ': self.chosen_daq.get(),
                'Min Voltage': min_voltage,
                'Max Voltage': max_voltage, 
                'Terminal Configuration': term_config, 
                'Device Name': name,
                'Selected Channel': selected
            })
            
        # Collect data for custom channels
        for i in range(1, self.no_of_cus_chans + 1):
            channel = self.entry_widgets[f"Channel_entry_cust{i}"].get()
            first_channel = self.entry_widgets[f"first_chan_sel{i}"].get()
            operation = self.entry_widgets[f"operation_sel{i}"].get()
            second_channel = self.entry_widgets[f"sec_chan_sel{i}"].get()
            units = self.entry_widgets[f"cust_units_entry{i}"].get()
            name = self.entry_widgets[f"custom_chan_name{i}"].get()
            selected = int(self.checkbutton_vars[f"Selected Channel_Custom{i}"].get())
            
            data3.append({
                'Channel': channel,
                'First Channel': first_channel, 
                'Operation': operation,
                'Second Channel': second_channel,
                'Device Name': name, 
                'Units': units,
                'Selected Channel': selected
            })
            
            # Check if any of the required fields are "n/a"
            if selected:
                if first_channel == '' or second_channel == '' or operation == '':
                    messagebox.showwarning("Missing Values", f"Warning: Selected custom channel {i} is missing values")

        # Create DataFrames
        df1 = pd.DataFrame(data1, columns=fieldnames)
        df2 = pd.DataFrame(data2, columns=fieldnames)
        df3 = pd.DataFrame(data3, columns=fieldnames_cus)

        # Write to Excel file with multiple sheets
        with pd.ExcelWriter('default_settings.xlsx') as writer:
            df1.to_excel(writer, sheet_name='default_settings', index=False)
            df2.to_excel(writer, sheet_name='output_settings', index=False)
            df3.to_excel(writer, sheet_name='custom_channel_settings', index=False)

    def close_window(self):    
        self.destroy()
        self.parent.destroy()

# Main Application
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Channel Settings")
        self.geometry("700x550")

        self.channel_settings = ChannelSettings(self, "Channel Settings")
        self.channel_settings.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        self.destroy()

if __name__ == "__main__":
    set_app = App()
    set_app.mainloop()
