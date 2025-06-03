# NI_python
Python code for data acquisition using National Instruments DAQs and the nidaqmx API

# GUI-based programs
Apps that let you visualize and record analog input from National Instruments devices. They are based on [PyQT5](https://pypi.org/project/PyQt5/) and [PyQtGraph](https://www.pyqtgraph.org/). 

# Requirements & Recommendations for This Repository
## Required Python Libraries
To use the recording and analysis apps in this repository, you must install the following Python packages:

* nidaqmx - For National Instruments DAQ device access.
* numpy
* scipy
* pyqt5
* pyqtgraph
* matplotlib
* pandas
* pyarrow
* tkinter
Quick install with pip: ``pip install numpy scipy pyqt5 pyqtgraph matplotlib pandas pyarrow nidaqmx``

Note: You must have [NI-DAQmx drivers installed on your system (from National Instruments)](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html).

## System Requirements
* Windows OS (required for NI-DAQmx hardware)
* NI-DAQmx drivers installed (from National Instruments)
* Python 3.8+ (Anaconda recommended)
* A supported National Instruments DAQ device

## Quick Start
Install all required libraries and drivers.
Make sure your DAQ device is connected and recognized by NI MAX.
If you encounter issues with NI-DAQmx, refer to [NI's official documentation](https://nidaqmx-python.readthedocs.io/en/stable/).

# 01_Record_ai_multichan.py
Acquire analog input on a various number of channels. Saves data in a .bin file and metadata in a .txt logfile.
Execute in your fav IDE or simply use your command line/Anaconda prompt like so:
```python "Path\to\gui_program.py"```
![Data Acquisition](https://github.com/muchaste/NI_python/releases/download/v0.1-alpha/data_acquisition.gif)

Record files with optional file splitting (e.g., for large files).
![Record Data](https://github.com/muchaste/NI_python/releases/download/v0.1-alpha/recording.gif)

# 02_Analyze_ai_multichan.py
Tkinter app for visualization of recordings. Reads in metadata from the logfile.
[!Data Analysis](https://github.com/muchaste/NI_python/releases/download/v0.1-alpha/data_analysis_module.png)

# Old Versions
The first generation of apps was based on [TKinter](https://docs.python.org/3/library/tkinter.html) and inspired by [example code by DavidFI](https://forums.ni.com/t5/Example-Code/Python-Voltage-Continuous-Input-py/ta-p/3938650). These apps can be found in the [Tkinter](https://github.com/muchaste/NI_python/tree/main/GUI_based/Tkinter) folder - however, they are deprecated and will not be developed anymore.
