#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic one-channel recording program with spectrogram option (PyQt5/pyqtgraph version)
Created on Thu Mar  6 13:43:40 2025

Authors: ShuttleBox / Stefan Mucha (ported by ChatGPT)
"""

import sys, os, time, threading, collections
import numpy as np
import nidaqmx
from nidaqmx import constants
import pyqtgraph as pg
from pyqtgraph import ImageItem
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from PyQt5 import QtWidgets, QtCore
from scipy import signal

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


# ---------------- Data Acquisition Module ----------------
class DataAcquisition:
    def __init__(self, input_channel, sample_rate, min_voltage, max_voltage, refresh_rate, plot_duration):
        self.input_channel = input_channel  # Expected to be like "Dev2/ai5"
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
        if self.recording_active:  # In case recording is running, signal completion
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
        # Extend the plot buffer (always store data for plotting)
        self.plot_buffer.extend(temp_data)

        # If recording is active, store data for saving
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

# ---------------- File Writing Module ----------------
class FileWriter(threading.Thread):
    def __init__(self, storage_buffer, buffer_lock, filepath, sample_rate, flush_interval=5):
        super().__init__(daemon=True)
        self.storage_buffer = storage_buffer
        self.buffer_lock = buffer_lock
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self.stop_event = threading.Event()

    def run(self):
        acquired_samples = 0
        dt = 1000 / self.sample_rate  # in ms
        with open(self.filepath, 'ab') as f:
            while not self.stop_event.is_set():
                time.sleep(self.flush_interval)
                with self.buffer_lock:
                    if not self.storage_buffer:
                        continue
                    data_chunk = np.array(self.storage_buffer, dtype='f8')
                    self.storage_buffer.clear()
                n_samples = len(data_chunk)
                time_vec = (acquired_samples + np.arange(n_samples)) * dt
                acquired_samples += n_samples
                interleaved = np.column_stack((time_vec, data_chunk)).flatten()
                interleaved.tofile(f)
                f.flush()
                os.fsync(f.fileno())
        print("FileWriter stopped.")

    def stop(self):
        self.stop_event.set()

# ---------------- PyQt5 GUI Module ----------------
class DataAcquisitionGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("eFish Recorder")
        self.resize(1600, 900)
        self.acq = None
        self.file_writer = None
        self.buffer_lock = threading.Lock()
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.init_ui()

    def init_ui(self):
        # Left controls panel
        controls_layout = QtWidgets.QVBoxLayout()

        # DAQ Settings GroupBox
        self.daqGroup = QtWidgets.QGroupBox("DAQ Settings")
        daq_layout = QtWidgets.QFormLayout()
        self.daqCombo = QtWidgets.QComboBox()
        self.daqCombo.addItems(daqList)
        daq_layout.addRow("Select DAQ:", self.daqCombo)
        self.chanEdit = QtWidgets.QLineEdit("ai5")
        daq_layout.addRow("Input Channel:", self.chanEdit)
        self.sampleRateEdit = QtWidgets.QLineEdit("20000")
        daq_layout.addRow("Sample Rate (Hz):", self.sampleRateEdit)
        self.daqGroup.setLayout(daq_layout)
        controls_layout.addWidget(self.daqGroup)

        # Plot Settings GroupBox
        self.plotGroup = QtWidgets.QGroupBox("Plot Settings")
        plot_layout = QtWidgets.QFormLayout()
        self.plotDurEdit = QtWidgets.QLineEdit("1")
        plot_layout.addRow("Plot Duration (s):", self.plotDurEdit)
        self.refreshRateEdit = QtWidgets.QLineEdit("10")
        plot_layout.addRow("Refresh Rate (Hz):", self.refreshRateEdit)
        self.specMinEdit = QtWidgets.QLineEdit("100")
        plot_layout.addRow("Min Freq (Hz):", self.specMinEdit)
        self.specMaxEdit = QtWidgets.QLineEdit("1400")
        plot_layout.addRow("Max Freq (Hz):", self.specMaxEdit)
        self.specCheck = QtWidgets.QCheckBox("Plot Spectrogram")
        self.specCheck.setChecked(True)
        plot_layout.addRow(self.specCheck)
        self.plotGroup.setLayout(plot_layout)
        controls_layout.addWidget(self.plotGroup)

        # Experiment Settings GroupBox
        self.expGroup = QtWidgets.QGroupBox("Experiment Settings")
        exp_layout = QtWidgets.QFormLayout()
        self.recIdEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("Recording ID:", self.recIdEdit)
        self.recDurEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Record Duration (s):", self.recDurEdit)
        btn_layout = QtWidgets.QHBoxLayout()
        self.startBtn = QtWidgets.QPushButton("Start")
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.recordBtn = QtWidgets.QPushButton("Start Recording")
        self.resetBtn = QtWidgets.QPushButton("Reset DAQ")
        self.closeBtn = QtWidgets.QPushButton("Close")
        btn_layout.addWidget(self.startBtn)
        btn_layout.addWidget(self.stopBtn)
        btn_layout.addWidget(self.recordBtn)
        btn_layout.addWidget(self.resetBtn)
        btn_layout.addWidget(self.closeBtn)
        exp_layout.addRow(btn_layout)
        self.expGroup.setLayout(exp_layout)
        controls_layout.addWidget(self.expGroup)

        # Right plotting panel using pyqtgraph
        self.rawPlotWidget = pg.PlotWidget(title="Raw Data")
        self.specPlotWidget = pg.ImageView(view=pg.PlotItem(title="Spectrogram"))
        # Layout for plotting
        plot_layout = QtWidgets.QVBoxLayout()
        plot_layout.addWidget(self.rawPlotWidget)
        plot_layout.addWidget(self.specPlotWidget)

        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls_layout)
        main_layout.addWidget(controls_widget, 1)
        plot_widget = QtWidgets.QWidget()
        plot_widget.setLayout(plot_layout)
        main_layout.addWidget(plot_widget, 3)
        self.setLayout(main_layout)

        # Connect button signals
        self.startBtn.clicked.connect(self.start_acquisition)
        self.stopBtn.clicked.connect(self.stop_acquisition)
        self.recordBtn.clicked.connect(self.start_record)
        self.resetBtn.clicked.connect(self.reset_device)
        self.closeBtn.clicked.connect(QtWidgets.qApp.quit)

    def start_acquisition(self):
        # Disable controls
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.recordBtn.setEnabled(True)
        self.daqCombo.setEnabled(False)
        self.chanEdit.setEnabled(False)
        self.sampleRateEdit.setEnabled(False)
        self.plotDurEdit.setEnabled(False)
        self.refreshRateEdit.setEnabled(False)
        self.specMinEdit.setEnabled(False)
        self.specMaxEdit.setEnabled(False)
        self.specCheck.setEnabled(False)

        # Get parameters
        device = self.daqCombo.currentText()
        channel = self.chanEdit.text().strip()
        input_channel = f"{device}/{channel}"
        sample_rate = int(self.sampleRateEdit.text())
        refresh_rate = int(self.refreshRateEdit.text())
        plot_duration = float(self.plotDurEdit.text())
        # Instantiate DataAcquisition
        self.acq = DataAcquisition(input_channel, sample_rate, -10, 10, refresh_rate, plot_duration)
        # Clear buffers
        self.acq.plot_buffer.clear()
        self.acq.storage_buffer.clear()
        self.acq.start()
        # Start update timer
        interval = int(1000 / refresh_rate)
        self.plot_timer.start(interval)

    def update_plot(self):
        # Update raw data plot
        data = list(self.acq.plot_buffer)
        self.rawPlotWidget.clear()
        if data:
            self.rawPlotWidget.plot(data, pen='y')
            if self.acq.recording_active:
                # Plot a green dot at the right edge
                self.rawPlotWidget.plot([len(self.acq.plot_buffer)], [0], pen=None, symbol='o', symbolBrush='g')
                
        # Update spectrogram if checked
            
        if self.specCheck.isChecked() and data:
            # Compute spectrogram using numpy (simple version)
            # Use a sliding window; here we use a fixed window length
            window = 256
            if len(data) >= window:
                # Split the signal into overlapping segments
                segments = []
                step = window // 2
                for i in range(0, len(data)-window, step):
                    segment = data[i:i+window]
                    segments.append(np.abs(np.fft.rfft(segment)))
                spec = np.array(segments).T
                self.specPlotWidget.setImage(spec, autoLevels=True)
        else:
            # self.specPlotWidget.clearImage()
            self.specPlotWidget.setImage(np.array([]))
        

    def start_record(self):
        self.recordBtn.setEnabled(False)
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "Binary files (*.bin)")
        if not filepath:
            self.recordBtn.setEnabled(True)
            return
        # Clear file by opening in write-binary mode
        with open(filepath, 'wb') as f:
            pass
        self.record_filepath = filepath
        rec_duration = float(self.recDurEdit.text())
        self.acq.samples_to_save = int(rec_duration * self.acq.sample_rate)
        self.acq.acquired_samples = 0
        self.acq.storage_buffer.clear()
        # Set recording completion callback
        self.acq.recording_complete_callback = lambda: QtCore.QTimer.singleShot(0, self.on_recording_complete)
        self.acq.recording_active = True

        # Start FileWriter thread
        self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, filepath, self.acq.sample_rate)
        self.file_writer.start()

    def on_recording_complete(self):
        self.recordBtn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Finished", "Recording complete")
        if self.file_writer is not None:
            self.file_writer.stop()
            self.file_writer.join()
        self.save_log_file()

    def save_log_file(self):
        # Save log file alongside the data file.
        log_filename = f"log_{os.path.basename(self.record_filepath).split('.')[0]}.txt"
        log_filepath = os.path.join(os.path.dirname(self.record_filepath), log_filename)
        log_data = {
            "Number of Input Channels": 1,
            "Sample Rate": self.acq.sample_rate,
            "Recording ID": self.recIdEdit.text(),
            "Recording Duration": self.recDurEdit.text(),
            "Input Channel": self.chanEdit.text()
        }
        with open(log_filepath, 'w') as f:
            for key, value in log_data.items():
                f.write(f"{key}: {value}\n")

    def stop_acquisition(self):
        if self.acq:
            self.acq.stop()
        self.plot_timer.stop()
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.recordBtn.setEnabled(False)
        self.daqCombo.setEnabled(True)
        self.chanEdit.setEnabled(True)
        self.sampleRateEdit.setEnabled(True)
        self.plotDurEdit.setEnabled(True)
        self.refreshRateEdit.setEnabled(True)
        self.specMinEdit.setEnabled(True)
        self.specMaxEdit.setEnabled(True)
        self.specCheck.setEnabled(True)

    def reset_device(self):
        dev = self.daqCombo.currentText()
        try:
            nidaqmx.system.Device(dev).reset_device()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = DataAcquisitionGUI()
    mainWin.show()
    sys.exit(app.exec_())
