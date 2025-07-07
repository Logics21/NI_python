#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic one-channel recording program with spectrogram option (PyQt5/pyqtgraph version)
Created on Thu Mar  6 13:43:40 2025

Authors: ShuttleBox / Stefan Mucha (ported by ChatGPT)
"""

import sys
import os
import time
import threading
import collections
import numpy as np
import nidaqmx
from nidaqmx import constants
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy import signal
from datetime import datetime

# Get list of DAQ device names
daqSys = nidaqmx.system.System()
daqList = daqSys.devices.device_names
if not daqList:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, "DAQ Error", "No DAQ detected, check connection.")
    sys.exit(1)

# ---------------- Data Acquisition Module ----------------
class DataAcquisition:
    def __init__(self, input_channels, sample_rate, min_voltage, max_voltage, refresh_rate, plot_duration,
                 do_channel=None, di_channel=None):
        # input_channels: list of strings
        self.input_channels = input_channels
        self.sample_rate = sample_rate
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.refresh_rate = refresh_rate

        self.num_channels = len(input_channels)
        sample_interval = int(self.sample_rate / self.refresh_rate)
        buffer_size = 100000
        if buffer_size % sample_interval != 0:
            for divisor in range(sample_interval, 0, -1):
                if buffer_size % divisor == 0:
                    sample_interval = divisor
                    break
        self.sample_interval = sample_interval

        # One buffer per channel
        self.plot_buffer = [collections.deque(maxlen=int(plot_duration * self.sample_rate)) for _ in range(self.num_channels)]
        self.storage_buffer = [collections.deque() for _ in range(self.num_channels)]
        # Buffer for DI samples and event log
        self.di_buffer = collections.deque() if di_channel else None
        self.led_event_log = []  # List of (timestamp, state)

        # --- Analog Input Task ---
        self.ai_task = nidaqmx.Task()
        for ch in input_channels:
            self.ai_task.ai_channels.add_ai_voltage_chan(
                ch, min_val=self.min_voltage, max_val=self.max_voltage,
                units=constants.VoltageUnits.VOLTS)
        self.ai_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS)
        self.ai_task.register_every_n_samples_acquired_into_buffer_event(self.sample_interval, self.callback)
        self.running = False

        # --- Digital Output Task (for LED) ---
        self.do_channel = do_channel
        if do_channel:
            self.do_task = nidaqmx.Task()
            self.do_task.do_channels.add_do_chan(do_channel)
            self.do_task.start()
        else:
            self.do_task = None

        # --- Digital Input Task (for LED sync detection) ---
        self.di_channel = di_channel
        if di_channel:
            self.di_task = nidaqmx.Task()
            self.di_task.di_channels.add_di_chan(di_channel)
            # Synchronize DI sampling with AI sampling clock
            self.di_task.timing.cfg_samp_clk_timing(
                self.sample_rate,
                source=f"/{di_channel.split('/')[0]}/ai/SampleClock",
                sample_mode=constants.AcquisitionType.CONTINUOUS
            )
        else:
            self.di_task = None

        self.recording_active = False
        self.acquired_samples = 0
        self.samples_to_save = 0
        self.recording_complete_callback = None
        self.recording_start_timestamp = None
        self.logfile_written = False
        self.logfile_callback = None

    def set_led(self, state: bool):
        """Set the LED digital output (True=on, False=off)"""
        if self.do_task:
            self.do_task.write(state)

    def start(self):
        self.running = True
        self.recording_start_timestamp = None
        self.logfile_written = False
        self.ai_task.start()
        if self.di_task:
            self.di_task.start()

    def stop(self):
        self.running = False
        if self.recording_active:
            self.recording_active = False
            if self.recording_complete_callback is not None:
                self.recording_complete_callback()
        self.recording_start_timestamp = None
        try:
            self.ai_task.stop()
            self.ai_task.close()
        except Exception as e:
            print("Error stopping AI DAQ task:", e)
        if self.do_task:
            try:
                self.do_task.stop()
                self.do_task.close()
            except Exception as e:
                print("Error stopping DO DAQ task:", e)
        if self.di_task:
            try:
                self.di_task.stop()
                self.di_task.close()
            except Exception as e:
                print("Error stopping DI DAQ task:", e)

    def callback(self, task_handle, event_type, number_of_samples, callback_data):
        if not self.running:
            return 0
        temp_data = self.ai_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        # temp_data shape: (num_channels, n_samples)
        if self.num_channels == 1:
            temp_data = np.array([temp_data])  # shape (1, n_samples)
        else:
            temp_data = np.array(temp_data)
        # Extend buffers per channel
        for ch_idx in range(self.num_channels):
            self.plot_buffer[ch_idx].extend(temp_data[ch_idx])
        # --- DI acquisition and event detection ---
        if self.di_task and self.di_buffer is not None:
            try:
                di_samples = self.di_task.read(number_of_samples_per_channel=temp_data.shape[1], timeout=0)
                # di_samples: shape (n_samples,) or (1, n_samples)
                if isinstance(di_samples, list) or isinstance(di_samples, np.ndarray):
                    if isinstance(di_samples, list):
                        di_samples = np.array(di_samples)
                    if di_samples.ndim > 1:
                        di_samples = di_samples[0]
                    self.di_buffer.extend(di_samples)
                    # Edge detection (rising/falling)
                    arr = np.array(self.di_buffer)
                    if arr.size > 1:
                        # Only check new samples
                        prev = arr[:-1]
                        curr = arr[1:]
                        edges = np.where(prev != curr)[0]
                        for idx in edges:
                            # Timestamp relative to AI sample clock
                            sample_idx = self.acquired_samples + idx - (arr.size - len(di_samples))
                            timestamp = (self.recording_start_timestamp or time.time()) + sample_idx / self.sample_rate
                            state = int(curr[idx])
                            self.led_event_log.append((timestamp, state))
                    # Keep buffer short
                    if len(self.di_buffer) > 1000:
                        for _ in range(len(self.di_buffer) - 1000):
                            self.di_buffer.popleft()
            except Exception:
                pass
        if self.recording_active:
            n_samples = temp_data.shape[1]
            if self.recording_start_timestamp is None:
                self.recording_start_timestamp = time.time() - (n_samples - 1) / self.sample_rate
            if not self.logfile_written and self.recording_start_timestamp is not None:
                if self.logfile_callback:
                    self.logfile_callback()
                self.logfile_written = True
            if self.acquired_samples + n_samples >= self.samples_to_save:
                diff = (self.acquired_samples + n_samples) - self.samples_to_save
                n_samples = n_samples - diff
                temp_data = temp_data[:, :n_samples]
                recording_complete = True
            else:
                recording_complete = False
            for ch_idx in range(self.num_channels):
                self.storage_buffer[ch_idx].extend(temp_data[ch_idx])
            self.acquired_samples += n_samples
            if recording_complete:
                self.recording_active = False
                if self.recording_complete_callback is not None:
                    self.recording_complete_callback()
        return 0

    def save_led_event_log(self, filepath):
        """Save LED on/off event log to a CSV file."""
        if not self.led_event_log:
            return
        with open(filepath, 'w') as f:
            f.write('timestamp,state\n')
            for ts, state in self.led_event_log:
                f.write(f'{ts:.6f},{state}\n')

# ---------------- File Writing Module ----------------
class FileWriter(threading.Thread):
    def __init__(self, storage_buffer, buffer_lock, filepath, sample_rate, flush_interval=5):
        super().__init__(daemon=True)
        self.storage_buffer = storage_buffer  # List of deques, one per channel
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
                    # Find minimum available samples across all channels
                    min_len = min(len(buf) for buf in self.storage_buffer)
                    if min_len == 0:
                        continue
                    # Pop min_len samples from each channel
                    data_chunk = np.array([ [self.storage_buffer[ch].popleft() for _ in range(min_len)] for ch in range(len(self.storage_buffer)) ])
                n_samples = min_len
                time_vec = (acquired_samples + np.arange(n_samples)) * dt
                acquired_samples += n_samples
                # Stack as columns: time, ch1, ch2, ...
                interleaved = np.column_stack((time_vec, data_chunk.T))  # shape (n_samples, num_channels+1)
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
        self.setWindowTitle("NI DAQ Recorder")
        self.resize(1600, 900)
        self.acq = None
        self.file_writer = None
        self.buffer_lock = threading.Lock()
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.spectrogram_lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        self.img = None
        self.init_ui()

    def init_ui(self):
        controls_layout = QtWidgets.QVBoxLayout()

        # DAQ Settings
        self.daqGroup = QtWidgets.QGroupBox("DAQ Settings")
        daq_layout = QtWidgets.QFormLayout()
        self.daqCombo = QtWidgets.QComboBox()
        self.daqCombo.addItems(daqList)
        daq_layout.addRow("Select DAQ:", self.daqCombo)
        self.chanEdit = QtWidgets.QLineEdit("ai0")
        daq_layout.addRow("Input Channel:", self.chanEdit)
        self.sampleRateEdit = QtWidgets.QLineEdit("20000")
        daq_layout.addRow("Sample Rate (Hz):", self.sampleRateEdit)

        # --- Add DO/DI channel dropdowns ---
        self.doCombo = QtWidgets.QComboBox()
        self.diCombo = QtWidgets.QComboBox()
        daq_layout.addRow("Digital Output (DO) Channel:", self.doCombo)
        daq_layout.addRow("Digital Input (DI) Channel:", self.diCombo)
        # Add Test LED button
        self.testLedBtn = QtWidgets.QPushButton("Test LED")
        self.testLedBtn.setCheckable(True)
        self.testLedBtn.setStyleSheet("color: orange; font-weight: bold;")
        daq_layout.addRow(self.testLedBtn)
        # Add random blink controls
        self.randomBlinkCheck = QtWidgets.QCheckBox("Enable Random LED Blinking")
        self.randomBlinkCheck.setToolTip("If checked, the LED will blink randomly during recording.")
        self.randomBlinkSpin = QtWidgets.QSpinBox()
        self.randomBlinkSpin.setRange(1, 1000)
        self.randomBlinkSpin.setValue(10)
        daq_layout.addRow(self.randomBlinkCheck)
        daq_layout.addRow("Number of Blinks:", self.randomBlinkSpin)
        self.daqGroup.setLayout(daq_layout)
        controls_layout.addWidget(self.daqGroup)

        # Populate digital channels for the initially selected device
        self.update_digital_channel_combos(self.daqCombo.currentText())
        self.daqCombo.currentTextChanged.connect(self.update_digital_channel_combos)

        # Plot Settings
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
        self.specWindowEdit = QtWidgets.QLineEdit("11")
        plot_layout.addRow("Spec. Window Size Exponent:", self.specWindowEdit)
        self.specCheck = QtWidgets.QCheckBox("Plot Spectrogram (Ch1 only)")
        self.specCheck.setToolTip("Enable to plot the spectrogram for the first channel.")
        self.specCheck.setChecked(True)
        plot_layout.addRow(self.specCheck)
        self.domFreqCheck = QtWidgets.QCheckBox("Show Dominant Frequency")
        self.domFreqCheck.setChecked(True)
        plot_layout.addRow(self.domFreqCheck)
        self.domFreqLabel = QtWidgets.QLabel("Dominant Frequency: --- Hz")
        plot_layout.addRow(self.domFreqLabel)
        self.yOffsetEdit = QtWidgets.QDoubleSpinBox()
        self.yOffsetEdit.setRange(0.0, 1000.0)
        self.yOffsetEdit.setSingleStep(0.1)
        self.yOffsetEdit.setValue(2.0)
        plot_layout.addRow("Y Offset (V):", self.yOffsetEdit)
        self.plotGroup.setLayout(plot_layout)
        controls_layout.addWidget(self.plotGroup)

        # Recording Settings
        self.expGroup = QtWidgets.QGroupBox("Recording Settings")
        exp_layout = QtWidgets.QFormLayout()
        self.recIdEdit = QtWidgets.QLineEdit()
        exp_layout.addRow("ID:", self.recIdEdit)
        self.recDurEdit = QtWidgets.QLineEdit("60")
        exp_layout.addRow("Duration (s):", self.recDurEdit)
        self.splitFileCheck = QtWidgets.QCheckBox("Enable File Splitting")
        self.splitFileCheck.setChecked(False)
        self.splitFileCheck.stateChanged.connect(self.toggle_split_duration)
        self.splitDurEdit = QtWidgets.QLineEdit("10")
        self.splitDurEdit.setEnabled(False)
        exp_layout.addRow(self.splitFileCheck)
        exp_layout.addRow("Split Duration (s):", self.splitDurEdit)

        btn_layout = QtWidgets.QHBoxLayout()
        self.connectBtn = QtWidgets.QPushButton("Connect")
        self.connectBtn.setStyleSheet("color: green; font-weight: bold;")
        self.disconnectBtn = QtWidgets.QPushButton("Disconnect")
        self.disconnectBtn.setStyleSheet("color: #003366; font-weight: bold;")
        self.recordBtn = QtWidgets.QPushButton("● Record")
        self.recordBtn.setStyleSheet("color: red; font-weight: bold;")
        self.resetBtn = QtWidgets.QPushButton("Reset DAQ")
        self.resetBtn.setStyleSheet("color: black; font-weight: bold;")
        self.closeBtn = QtWidgets.QPushButton("Close")
        self.closeBtn.setStyleSheet("color: black; font-weight: bold;")
        btn_layout.addWidget(self.connectBtn)
        btn_layout.addWidget(self.disconnectBtn)
        btn_layout.addWidget(self.recordBtn)
        btn_layout.addWidget(self.resetBtn)
        btn_layout.addWidget(self.closeBtn)
        exp_layout.addRow(btn_layout)
        self.expGroup.setLayout(exp_layout)
        controls_layout.addWidget(self.expGroup)

        # Plotting panel
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.rawPlotWidget = pg.PlotWidget(title="Raw Data")
        self.specPlotWidget = pg.PlotWidget(title="Spectrogram")
        self.specPlotWidget.setMouseEnabled(x=False, y=False)
        plot_vlayout = QtWidgets.QVBoxLayout()
        plot_vlayout.addWidget(self.rawPlotWidget)
        plot_vlayout.addWidget(self.specPlotWidget)

        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls_layout)
        main_layout.addWidget(controls_widget, 1)
        plot_widget = QtWidgets.QWidget()
        plot_widget.setLayout(plot_vlayout)
        main_layout.addWidget(plot_widget, 3)
        self.setLayout(main_layout)

        # Connect signals
        self.connectBtn.clicked.connect(self.start_acquisition)
        self.disconnectBtn.clicked.connect(self.stop_acquisition)
        self.recordBtn.clicked.connect(self.start_record)
        self.resetBtn.clicked.connect(self.reset_device)
        self.closeBtn.clicked.connect(self.safe_close)
        self.testLedBtn.clicked.connect(self.toggle_test_led)
        self.randomBlinkCheck.stateChanged.connect(self.toggle_random_blink_ui)
        self.randomBlinkSpin.setEnabled(self.randomBlinkCheck.isChecked())

    def toggle_split_duration(self):
        self.splitDurEdit.setEnabled(self.splitFileCheck.isChecked())

    def update_digital_channel_combos(self, device_name):
        """Populate DO/DI dropdowns with available digital lines for the selected device."""
        import nidaqmx
        try:
            device = nidaqmx.system.Device(device_name)
            do_lines = [line.name.split("/", 1)[1] for line in device.do_lines]
            di_lines = [line.name.split("/", 1)[1] for line in device.di_lines]
        except Exception:
            do_lines = []
            di_lines = []
        self.doCombo.clear()
        self.diCombo.clear()
        self.doCombo.addItem("")  # Allow blank selection
        self.diCombo.addItem("")
        self.doCombo.addItems(do_lines)
        self.diCombo.addItems(di_lines)

    def start_acquisition(self):
        device = self.daqCombo.currentText()
        # Accept comma or whitespace separated channels, strip whitespace
        channel_text = self.chanEdit.text().strip()
        channel_list = [f"{device}/{ch.strip()}" for ch in channel_text.replace(',', ' ').split()]
        # --- Get selected DO/DI channels ---
        do_channel = self.doCombo.currentText().strip()
        di_channel = self.diCombo.currentText().strip()
        do_channel_full = f"{device}/{do_channel}" if do_channel else None
        di_channel_full = f"{device}/{di_channel}" if di_channel else None
        try:
            sample_rate = int(self.sampleRateEdit.text())
            min_freq = float(self.specMinEdit.text())
            max_freq = float(self.specMaxEdit.text())
            refresh_rate = int(self.refreshRateEdit.text())
            plot_duration = float(self.plotDurEdit.text())
            spec_window_size = 2 ** int(self.specWindowEdit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
            return
        if sample_rate <= 0 or refresh_rate <= 0 or plot_duration <= 0:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Sample Rate, Refresh Rate, and Plot Duration must be positive.")
            return

        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.spec_window_size = spec_window_size

        self.connectBtn.setEnabled(False)
        self.resetBtn.setEnabled(False)
        self.disconnectBtn.setEnabled(True)
        self.recordBtn.setEnabled(True)
        self.daqCombo.setEnabled(False)
        self.chanEdit.setEnabled(False)
        self.sampleRateEdit.setEnabled(False)
        self.plotDurEdit.setEnabled(False)
        self.refreshRateEdit.setEnabled(False)
        self.specMinEdit.setEnabled(False)
        self.specMaxEdit.setEnabled(False)
        self.specWindowEdit.setEnabled(False)
        self.doCombo.setEnabled(False)
        self.diCombo.setEnabled(False)

        self.acq = DataAcquisition(channel_list, sample_rate, -10, 10, refresh_rate, plot_duration,
                                   do_channel=do_channel_full, di_channel=di_channel_full)
        for buf in self.acq.plot_buffer:
            buf.clear()
        for buf in self.acq.storage_buffer:
            buf.clear()
        self.acq.start()
        interval = int(1000 / refresh_rate)
        self.plot_timer.start(interval)

    def update_plot(self):
        # Plot all channels, color-coded, with adjustable y-offset for clarity
        self.rawPlotWidget.clear()
        if self.acq and self.acq.plot_buffer:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            y_offset = self.yOffsetEdit.value()  # Adjustable y-offset for channel separation
            for ch_idx, buf in enumerate(self.acq.plot_buffer):
                data = np.array(buf)
                if data.size > 0:
                    color = colors[ch_idx % len(colors)]
                    pen = pg.mkPen(color=color, width=2)
                    offset_data = data + ch_idx * y_offset
                    self.rawPlotWidget.plot(offset_data, pen=pen, name=f"Ch{ch_idx+1}")
            self.rawPlotWidget.setLabel('bottom', "Sample", units='s')
            self.rawPlotWidget.setLabel('left', "Voltage + offset", units='V')
            # Optionally, add channel labels on the left as text
            for ch_idx in range(len(self.acq.plot_buffer)):
                label = pg.TextItem(f"Ch{ch_idx+1}", color=colors[ch_idx % len(colors)])
                label.setPos(0, ch_idx * y_offset)
                self.rawPlotWidget.addItem(label)

        # Spectrogram: only for first channel (for simplicity)
        if self.acq and self.acq.plot_buffer and self.specCheck.isChecked():
            data = np.array(self.acq.plot_buffer[0])
            if data.size > 0 and data.size >= self.spec_window_size:
                f, t, Sxx = signal.spectrogram(
                    data,
                    fs=self.sample_rate,
                    window='hann',
                    nperseg=self.spec_window_size,
                    noverlap=self.spec_window_size // 2,
                    detrend=False,
                    scaling='density',
                    mode='magnitude'
                )
                freq_mask = (f >= self.min_freq) & (f <= self.max_freq)
                Sxx = Sxx[freq_mask, :]
                f = f[freq_mask]

                if self.domFreqCheck.isChecked() and Sxx.size > 0 and f.size > 0:
                    mean_spectrum = np.mean(Sxx, axis=1)
                    dom_idx = np.argmax(mean_spectrum)
                    dom_freq = f[dom_idx]
                    self.domFreqLabel.setText(f"Dominant Frequency: {dom_freq:.1f} Hz")
                    self.specPlotWidget.setTitle(f"Spectrogram (Dominant: {dom_freq:.1f} Hz)")
                else:
                    self.domFreqLabel.setText("Dominant Frequency: --- Hz")
                    self.specPlotWidget.setTitle("Spectrogram")

                self.specPlotWidget.clear()
                if self.img is None or self.img.scene() is None:
                    self.img = pg.ImageItem(axisOrder='row-major')
                self.specPlotWidget.addItem(self.img)
                self.img.setImage(Sxx, autoLevels=True)
                self.img.setLookupTable(self.spectrogram_lut)

                if t.size > 1 and f.size > 1:
                    xscale = (t[-1] - t[0]) / float(Sxx.shape[1]) if Sxx.shape[1] > 1 else 1
                    yscale = (f[-1] - f[0]) / float(Sxx.shape[0]) if Sxx.shape[0] > 1 else 1
                    transform = QtGui.QTransform()
                    transform.scale(xscale, yscale)
                    transform.translate(0, self.min_freq / yscale)
                    self.img.setTransform(transform)
                    self.specPlotWidget.setLimits(xMin=0, xMax=t[-1], yMin=self.min_freq, yMax=f[-1])
                    self.specPlotWidget.setLabel('bottom', "Time", units='s')
                    self.specPlotWidget.setLabel('left', "Frequency", units='Hz')
            # Removed the redundant clearing of the spectrogram plot
        # File splitting logic
        if getattr(self, 'split_enabled', False) and self.acq and self.acq.acquired_samples > 0:
            if self.next_split_idx < len(self.split_points):
                next_split = self.split_points[self.next_split_idx]
                if self.acq.acquired_samples >= next_split:
                    if self.file_writer is not None:
                        self.file_writer.stop()
                        self.file_writer.join()
                    self.save_log_file()
                    self.split_counter += 1
                    self.record_filepath = self._split_filename()
                    self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
                    self.file_writer.start()
                    self.acq.logfile_written = False
                    self.acq.recording_start_timestamp = None
                    self.next_split_idx += 1
                    # Generate new blink sequence for this split
                    self._generate_random_blink_sequence(self.split_duration)
                    self._random_blink_start_time = time.time()
                    self._schedule_next_blink()

    def toggle_random_blink_ui(self):
        self.randomBlinkSpin.setEnabled(self.randomBlinkCheck.isChecked())

    def _generate_random_blink_sequence(self, duration):
        if not (self.randomBlinkCheck.isChecked() and self.acq and self.acq.do_task):
            self._random_blink_events = []
            return
        n_blinks = self.randomBlinkSpin.value()
        blink_duration = 0.1  # seconds
        import random
        seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed)
        blink_times = sorted(random.uniform(0, duration - blink_duration) for _ in range(n_blinks))
        self._random_blink_events = [(t, True) for t in blink_times] + [(t + blink_duration, False) for t in blink_times]
        self._random_blink_events.sort()
        self._random_blink_idx = 0

    def _schedule_next_blink(self):
        if not (self.randomBlinkCheck.isChecked() and self.acq and self.acq.do_task):
            return
        if not hasattr(self, '_random_blink_events') or self._random_blink_events is None or self._random_blink_idx >= len(self._random_blink_events):
            return
        event_time, state = self._random_blink_events[self._random_blink_idx]
        now = time.time()
        elapsed = now - self._random_blink_start_time
        delay = max(0, event_time - elapsed)
        QtCore.QTimer.singleShot(int(delay * 1000), lambda: self._do_random_blink(state))

    def _do_random_blink(self, state):
        if self.acq and self.acq.do_task:
            self.acq.set_led(state)
        self._random_blink_idx += 1
        self._schedule_next_blink()

    def start_record(self):
        self.recordBtn.setEnabled(False)
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "Binary files (*.bin)")
        if not filepath:
            self.recordBtn.setEnabled(True)
            return

        rec_duration = float(self.recDurEdit.text())
        self.acq.samples_to_save = int(rec_duration * self.acq.sample_rate)
        self.acq.acquired_samples = 0
        for buf in self.acq.storage_buffer:
            buf.clear()
        self.acq.recording_complete_callback = lambda: QtCore.QTimer.singleShot(0, self.on_recording_complete)
        self.acq.recording_active = True
        self.acq.logfile_written = False
        self.acq.logfile_callback = self.save_log_file

        self.split_enabled = self.splitFileCheck.isChecked()
        self._random_blink_start_time = None
        self._random_blink_events = None
        self._random_blink_idx = 0
        if self.split_enabled:
            try:
                self.split_duration = float(self.splitDurEdit.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid split duration.")
                self.recordBtn.setEnabled(True)
                return
            if self.split_duration >= rec_duration:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Split duration must be smaller than total duration.")
                self.recordBtn.setEnabled(True)
                return
            self.split_samples = int(self.split_duration * self.acq.sample_rate)
            self.split_counter = 1
            self.base_filepath = os.path.splitext(filepath)[0]
            self.record_filepath = self._split_filename()
            with open(self.record_filepath, 'wb'):
                pass
            self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
            total_samples = self.acq.samples_to_save
            self.split_points = [self.split_samples * i for i in range(1, int(np.ceil(total_samples / self.split_samples)))]
            self.next_split_idx = 0
            # Generate blink sequence for first split
            self._generate_random_blink_sequence(self.split_duration)
            self._random_blink_start_time = time.time()
            self._schedule_next_blink()
        else:
            self.record_filepath = filepath
            with open(self.record_filepath, 'wb'):
                pass
            self.file_writer = FileWriter(self.acq.storage_buffer, self.buffer_lock, self.record_filepath, self.acq.sample_rate)
            # Generate blink sequence for full duration
            self._generate_random_blink_sequence(rec_duration)
            self._random_blink_start_time = time.time()
            self._schedule_next_blink()
        self.file_writer.start()

    def _split_filename(self):
        return f"{self.base_filepath}_{self.split_counter:03d}.bin"

    def on_recording_complete(self):
        if self.file_writer is not None:
            self.file_writer.stop()
            self.file_writer.join()
        self.acq.recording_start_timestamp = None
        # Save LED event log if DI was used
        if self.acq and self.acq.di_channel:
            led_log_path = os.path.splitext(self.record_filepath)[0] + '_led_events.csv'
            self.acq.save_led_event_log(led_log_path)
            self.acq.led_event_log = []  # reset LED event list after saving
        # Clean up random blink state
        if hasattr(self, '_random_blink_events'):
            del self._random_blink_events
        self.recordBtn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Finished", "Recording complete")

    def save_log_file(self):
        log_filename = f"log_{os.path.basename(self.record_filepath).split('.')[0]}.txt"
        log_filepath = os.path.join(os.path.dirname(self.record_filepath), log_filename)
        if self.acq and self.acq.recording_start_timestamp is not None:
            dt = datetime.fromtimestamp(self.acq.recording_start_timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        else:
            timestamp_str = "N/A"
        log_data = {
            "N_Input_Channels": self.acq.num_channels,
            "Sample_Rate": self.acq.sample_rate,
            "Recording_ID": self.recIdEdit.text(),
            "Total_Recording_Duration": self.recDurEdit.text(),
            "Split_Recording_Duration": self.splitDurEdit.text() if self.splitFileCheck.isChecked() else "N/A",
            "Input_Channels": ', '.join(self.acq.input_channels),
            "Recording_Start_Timestamp": timestamp_str
        }
        with open(log_filepath, 'w') as f:
            for key, value in log_data.items():
                f.write(f"{key}: {value}\n")

    def stop_acquisition(self):
        if self.acq:
            self.acq.stop()
        self.plot_timer.stop()
        self.connectBtn.setEnabled(True)
        self.resetBtn.setEnabled(True)
        self.disconnectBtn.setEnabled(False)
        self.recordBtn.setEnabled(False)
        self.daqCombo.setEnabled(True)
        self.chanEdit.setEnabled(True)
        self.sampleRateEdit.setEnabled(True)
        self.plotDurEdit.setEnabled(True)
        self.refreshRateEdit.setEnabled(True)
        self.specMinEdit.setEnabled(True)
        self.specMaxEdit.setEnabled(True)
        self.specWindowEdit.setEnabled(True)
        self.specCheck.setEnabled(True)
        self.doCombo.setEnabled(True)
        self.diCombo.setEnabled(True)

    def reset_device(self):
        dev = self.daqCombo.currentText()
        try:
            nidaqmx.system.Device(dev).reset_device()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def safe_close(self):
        # Stop and close the DAQ task if still running
        if hasattr(self, "acq") and self.acq is not None:
            try:
                if getattr(self.acq, "ai_task", None) is not None:
                    if self.acq.running:
                        self.acq.stop()
            except Exception as e:
                print(f"Error during safe close (DAQ): {e}")
        # Stop file writer thread if running
        if hasattr(self, "file_writer") and self.file_writer is not None:
            try:
                self.file_writer.stop()
                self.file_writer.join(timeout=2)
            except Exception as e:
                print(f"Error stopping file writer: {e}")
        QtWidgets.qApp.quit()

    def toggle_test_led(self):
        """Toggle the digital output (LED) on/off for testing using set_led."""
        if self.acq and hasattr(self.acq, 'set_led') and self.acq.do_task:
            state = self.testLedBtn.isChecked()
            self.acq.set_led(state)
            if state:
                self.testLedBtn.setText("Test LED (ON)")
            else:
                self.testLedBtn.setText("Test LED (OFF)")
        else:
            QtWidgets.QMessageBox.warning(self, "No DO Task", "Digital Output task is not initialized. Connect first and select a DO channel.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = DataAcquisitionGUI()
    mainWin.show()
    sys.exit(app.exec_())
