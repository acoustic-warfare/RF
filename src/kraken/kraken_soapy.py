import numpy as np
import numpy.linalg as lin
import sys
import SoapySDR as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import pyargus.directionEstimation as pa
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from SoapySDR import *
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor

class KrakenReceiver():

    """
    Represents a Kraken receiver system for signal processing.

    Attributes:
    num_devices : int
        Number of receiver devices.
    center_freq : float
        Center frequency of the signal.
    num_samples : int
        Number of samples per device per capture.
    sample_rate : float
        Sampling rate in samples per second.
    bandwidth : float
        Bandwidth of the signal.
    gain : float
        Gain of the receiver devices.
    buffer : numpy.ndarray
        Buffer to store captured signal data, shape (num_devices, num_samples).
    devices : numpy.ndarray
        List of device objects representing the receiver devices.
    streams : numpy.ndarray
        List of streams associated with each device for data capture.
    x : float
        X-coordinate of the receiver location (scaled by antenna distance).
    y : float
        Y-coordinate of the receiver location.
    num_devices : int, optional
        Number of receiver devices (default is 5).
    filter : scipy filter object
        A signal processing filter
    """
    def __init__(self, center_freq, num_samples, sample_rate, bandwidth, gain, 
                 antenna_distance, x, y, num_devices=5, circular = 0,
                 simulation = 0, simulation_angles = [0], simulation_frequencies = [434.4e6], f_type = 'LTI', detection_range = 360):
        
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        #self.bandwidth = bandwidth
        self.gain = gain
        self.f_type = f_type
        self.devices, self.streams = (0,0) #self._setup_devices()

        self.simulation = simulation
        self.simulation_angles = simulation_angles
        self.simulation_frequencies = simulation_frequencies
        self.circular = circular
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.detection_range = detection_range
        self.real_offs = 00.0

        if simulation:
            if self.circular:
                self.buffer = signals_linear(self.simulation_frequencies, self.simulation_angles ,self.num_devices, self.num_samples, self.x, antenna_distance)
            else:
                self.buffer = signals_circular(self.simulation_frequencies, self.simulation_angles ,self.num_devices, self.num_samples, self.x, self.y, antenna_distance)
            self.offs = 180.0
        else:
            self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)
            self.offs = self.real_offs
        
        
        if f_type == 'butter':
            #Build digital filter
            fc = self.center_freq
            fs = 4*fc
            fn = 0.5*fs
            f_bandwidth = 0.6*fc
            wn = [(f_bandwidth/2) / fn]
            wn = [np.finfo(float).eps, (f_bandwidth/2) / fn] 
            sos = signal.butter(0, wn, btype='lowpass', output='sos')
            self.filter = sos

        elif f_type == 'FIR':
            #Design a FIR filter using the firwin function
            numtaps = 21  # Number of filter taps (filter length)
            fc = self.center_freq
            fs = 4*fc
            bandwidth = 0.3*fc
            highcut = bandwidth/2  # Upper cutoff frequency (Hz)
            taps = signal.firwin(numtaps, [highcut], fs=fs, pass_zero=True)
            self.filter = taps

        elif f_type == 'LTI':
        
            num = [0.0, 1.0]
            den = [4e-7, 1.0]

            #system = signal.lti(num, den)
            #(b, a) = signal.TransferFunction(num, den)
            
            # Convert to discrete-time system
            dt = 1e-6
            discrete_system = signal.cont2discrete((num, den), dt)
            #self.b, self.a = discrete_system[0], discrete_system[1]
            self.b = np.array(discrete_system[0].flatten(), dtype=np.float64)
            self.a = np.array(discrete_system[1].flatten(), dtype=np.float64)    

    def _setup_device(self, device_args, hw_time):
        """
        Set up a receiver device with specified parameters.

        Parameters:
        device_args : dict
            Dictionary containing arguments needed to initialize the device.

        Returns:
        sp.Device
            Initialized receiver device object configured with specified parameters.

        """
        device = sp.Device(device_args) 
        device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        device.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        device.setGain(SOAPY_SDR_RX, 0, self.gain)
        device.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        device.setHardwareTime(hw_time)
        return device
    
    
    def _setup_devices(self):
        """
        Set up multiple receiver devices for data capture.

        Returns:
        tuple
            A tuple containing:
            - numpy.ndarray: Array of initialized receiver device objects (`sp.Device`).
            - numpy.ndarray: Array of activated data streams associated with each device.

        Raises:
        ValueError:
            If setting up or activating the data stream for any device fails.
        """
        devices = np.zeros(self.num_devices, dtype=object)
        streams = np.zeros(self.num_devices, dtype=object)
        available_devices = sp.Device.enumerate()
        
        available_devices = (available_devices[1:] + available_devices[:1])[:self.num_devices]

        hw_time = int((time.time() * 1e9) + 5e9)

        for i, device_args in enumerate(available_devices):
            device = self._setup_device(device_args, hw_time)
            devices[i] = device
            rx_stream = device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
            msg = device.activateStream(rx_stream)
            if msg != 0:
                raise ValueError(f"Stream setup of device {i} failed with error message {msg}")
            streams[i] = rx_stream

        return devices, streams

    def close_streams(self):
        """
        Deactivates and closes all active data streams for the receiver devices.
        """
        for i, device in enumerate(self.devices):
            device.deactivateStream(self.streams[i])
            device.closeStream(self.streams[i])
            del device

    def _read_stream(self, device, timestamp):
        """
        Reads data from a specified device's stream and handles any errors.

        Parameters:
        device : int
            Index of the device from which to read data.
        timestamp : int
            Timestamp indicating the start time of the read operation.

        Raises:
        ValueError:
            If an error occurs while reading the stream (e.g., timeout, overflow).
        """
        #self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)
        
        sr = self.devices[device].readStream(self.streams[device], [self.buffer[device]], 
                                            self.num_samples, 0, timestamp)
        print(f"stream {device} read")
        
        ret = sr.ret

        if ret < 0:
        #Handle errors based on the error codes
            if ret == SOAPY_SDR_TIMEOUT:
                raise ValueError("Timeout when reading stream")
            elif ret == SOAPY_SDR_STREAM_ERROR:
                raise ValueError("Stream error when reading stream")
            elif ret == SOAPY_SDR_CORRUPTION:
                raise ValueError("Data corruption when reading stream")
            elif ret == SOAPY_SDR_OVERFLOW:
                raise ValueError("Overflow when reading stream")
            elif ret == SOAPY_SDR_NOT_SUPPORTED:
                raise ValueError("Requested operation or flag setting is not supported")
            elif ret == SOAPY_SDR_TIME_ERROR:
                raise ValueError("Encountered a stream time which was expired or too early to process")
            elif ret == SOAPY_SDR_UNDERFLOW:
                raise ValueError("write caused an underflow condition")

    def apply_filter(self):
        if self.f_type == 'none': 
            pass
        elif self.f_type == 'LTI':
            self.buffer = signal.lfilter(self.b, self.a, self.buffer)
        elif self.f_type == 'butter':
            self.buffer = signal.sosfilt(self.filter, self.buffer)
        elif self.f_type == 'FIR':
            self.buffer = signal.lfilter(self.filter, 1.0, self.buffer)

    def read_streams(self):
        """
        Reads data from all receiver devices and filters the captured signals.

        Uses a thread pool to read data from multiple devices concurrently,
        then applies a filter to the captured signals stored in `self.buffer`.
        """
        current_time_ns = int(time.time() * 1e9)
        start_time_ns = int(current_time_ns + 5e9)
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(self._read_stream, i, start_time_ns) for i in range(self.num_devices)]
            for future in futures:
                future.result()
    
    #Rewrite of pyArgus spatial smoothing to match our buffer dimensions
    def spatial_smoothing_rewrite(self, P, direction): 
        
        M = self.num_devices
        N = self.num_samples  
        L = M - P + 1  # Number of subarrays    

        Rss = np.zeros((P, P), dtype=complex)  # Spatially smoothed correlation matrix np.arange(-self.detection_range/2 + self.real_offs, self.detection_range/2 + self.real_offs))

        if direction == "backward" or direction == "forward-backward":         
            for l in range(L): 
                Rxx = np.zeros((P, P), dtype=complex)  # Correlation matrix allocation 
                for n in np.arange(0,N,1): 
                    d = np.conj(self.buffer[M-l-P:M-l, n][::-1]) 
                    Rxx += np.outer(d, np.conj(d)) 
                np.divide(Rxx, N)  # normalization 
                Rss += Rxx 

        if not (direction == "forward" or direction == "backward" or direction == "forward-backward"):     
            print("ERROR: Smoothing direction not recognized!") 
            return -1 

        # normalization            
        if direction == "forward-backward": 
            np.divide(Rss, 2*L)  
        else: 
            np.divide(Rss,L)  

        return Rss

    def music(self, signal_dimension):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        #smoothed_buffer = self.spatial_smoothing_rewrite(2, 'forward-backward')
        #spatial_corr_matrix = np.dot(smoothed_buffer, smoothed_buffer.conj().T)
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = np.divide(spatial_corr_matrix, self.num_samples)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2 + self.offs, self.detection_range/2 + self.offs))
        doa = pa.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, signal_dimension=signal_dimension)

        return doa

    def mem(self):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = np.divide(spatial_corr_matrix, self.num_samples)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2, self.detection_range/2))
        doa = pa.DOA_MEM(spatial_corr_matrix,scanning_vectors)

        return doa

    def capon(self):
        """
        Performs Direction of Arrival (DOA) estimation using the Capon algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = np.divide(spatial_corr_matrix, self.num_samples)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2, self.detection_range/2))
        doa = pa.DOA_Capon(spatial_corr_matrix,scanning_vectors)

        return doa

    def bartlett(self):
        """
        Performs Direction of Arrival (DOA) estimation using the Bartlett algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = np.divide(spatial_corr_matrix, self.num_samples)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2, self.detection_range/2))
        doa = pa.DOA_Bartlett(spatial_corr_matrix,scanning_vectors)

        return doa

    def lpm(self, element_select):
        """
        Performs Direction of Arrival (DOA) estimation using the LPM algorithm with specified element selection.

        Parameters:
        element_select : int
            Index of the element to select for DOA estimation.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = np.divide(spatial_corr_matrix, self.num_samples)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2, self.detection_range/2))
        doa = pa.DOA_LPM(spatial_corr_matrix,scanning_vectors, element_select)

        return doa

def get_device_info():
        """
        Retrieves and prints information about available SDR devices.

        Returns:
        list of dict:
            A list where each element is a dictionary containing information about
            one SDR device, including its sample rate range, frequency range,
            gain range, and bandwidth range.

        """
        available_devices = sp.Device.enumerate()
        device_info_list = []

        for i, device_args in enumerate(available_devices):
            device_info = {}
            print(device_args)
            device = sp.Device(device_args)
            device_info['sample_rate_range'] = device.getSampleRateRange(SOAPY_SDR_RX, 0)
            device_info['sample_rate'] = device.getSampleRate(SOAPY_SDR_RX, 0)
            device_info['list_sample_rates'] = device.listSampleRates(SOAPY_SDR_RX, 0)
            device_info['frequency_range'] = device.getFrequencyRange(SOAPY_SDR_RX, 0)
            device_info['has_frequency_correction'] = device.hasFrequencyCorrection(SOAPY_SDR_RX, 0)
            device_info['get_frequency_correction'] = device.getFrequencyCorrection(SOAPY_SDR_RX, 0)
            device_info['bandwidth_range'] = device.getBandwidthRange(SOAPY_SDR_RX, 0)
            device_info['list_bandwidths'] = device.listBandwidths(SOAPY_SDR_RX, 0)
            device_info['get_bandwidths'] = device.getBandwidth(SOAPY_SDR_RX, 0)
            device_info['gain_range'] = device.getGainRange(SOAPY_SDR_RX, 0)
            device_info['has_gain_mode'] = device.hasGainMode(SOAPY_SDR_RX, 0)
            device_info['gain_mode'] = device.getGainMode(SOAPY_SDR_RX, 0)
            device_info['list_clock_sources'] = device.listClockSources()
            device_info['get_master_clock_rate'] = device.getMasterClockRate()
            device_info['get_master_clock_rates'] = device.getMasterClockRates()
            device_info['get_reference_clock_rate'] = device.getReferenceClockRate()
            device_info['get_reference_clock_rates'] = device.getReferenceClockRates()
            device_info['get_clock_source'] = device.getClockSource()
            device_info['has_hardware_time'] = device.hasHardwareTime()
            device_info['list_time_sources'] = device.listTimeSources()
            device_info['get_time_source'] = device.getTimeSource()
            device_info['list_antennas'] = device.listAntennas(SOAPY_SDR_RX, 0)
            device_info['get_antenna'] = device.getAntenna(SOAPY_SDR_RX, 0)
            device_info['has_dc_offset_mode'] = device.hasDCOffsetMode(SOAPY_SDR_RX, 0)
            device_info['get_dc_offset_mode'] = device.getDCOffsetMode(SOAPY_SDR_RX, 0)
            device_info['has_dc_offset'] = device.hasDCOffset(SOAPY_SDR_RX, 0)
            device_info['get_dc_offset'] = device.getDCOffset(SOAPY_SDR_RX, 0)
            device_info['has_iq_balance_mode'] = device.hasIQBalanceMode(SOAPY_SDR_RX, 0)
            device_info['get_iq_balance_mode'] = device.getIQBalanceMode(SOAPY_SDR_RX, 0)
            device_info['has_iq_balance'] = device.hasIQBalance(SOAPY_SDR_RX, 0)
            device_info['get_iq_balance'] = device.getIQBalance(SOAPY_SDR_RX, 0)
            device_info['list_gpio_banks'] = device.listGPIOBanks()
            device_info['list_uart'] = device.listUARTs()
            device_info['get_native_stream_format'] = device.getNativeStreamFormat(SOAPY_SDR_RX, 0)


            print(f"Device {i}:")
            for key, value in device_info.items():
                print(f"  {key}: {value}")
                
            device_info_list.append(device_info)

        return device_info_list

def signals_linear(frequencies, angles, num_sensors, num_snapshots, antenna_positions, antenna_distance, wavelength=1.0, noise_power=1e-3):
    """
    Generates signals received by sensor array.

    Parameters:
    frequencies : list
        List of frequencies (in Hz) of the transmitted signals.
    angles : list
        List of angles (in degrees) of arrival corresponding to each frequency.
    num_sensors : int
        Number of sensors in the array.
    num_snapshots : int
        Number of signal snapshots to generate.
    wavelength : float, optional
        Wavelength of the transmitted signals (default is 1.0).
    noise_power : float, optional
        Power of additive Gaussian noise (default is 1e-3).

    Returns:
    numpy.ndarray
        2D array of complex numbers representing received signals at each sensor
        over time (shape: (num_sensors, num_snapshots)).

    """
    sensor_positions = antenna_positions * antenna_distance
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    frequency_offset = frequencies[0]

    for f, angle in zip(frequencies, angles):
        f_cal = f - frequency_offset
        signal = np.exp(1j * 2 * np.pi * f_cal * np.arange(num_snapshots) / num_snapshots)
        steering_vector = np.exp(1j * 2 * np.pi * sensor_positions[:, np.newaxis] * np.sin(np.radians(angle)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + 500* noise


def signals_circular(frequencies, angles, num_sensors, num_snapshots, antenna_positions_x, antenna_positions_y , antenna_distance, wavelength=1.0, noise_power=1e-3):
    """
    Generates signals received by a circular sensor array.

    Parameters:
    frequencies : list
        List of frequencies (in Hz) of the transmitted signals.
    angles : list
        List of angles (in degrees) of arrival corresponding to each frequency.
    num_sensors : int
        Number of sensors in the circular array.
    num_snapshots : int
        Number of signal snapshots to generate.
    radius : float
        Radius of the circular array.
    wavelength : float, optional
        Wavelength of the transmitted signals (default is 1.0).
    noise_power : float, optional
        Power of additive Gaussian noise (default is 1e-3).

    Returns:
    numpy.ndarray
        2D array of complex numbers representing received signals at each sensor
        over time (shape: (num_sensors, num_snapshots)).
    """
    sensor_positions_x = antenna_positions_x * antenna_distance
    sensor_positions_y = antenna_positions_y * antenna_distance
    
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    frequency_offset = frequencies[0]

    for f, angle in zip(frequencies, angles):
        f_cal = f - frequency_offset
        signal = np.exp(1j * 2 * np.pi * f_cal * np.arange(num_snapshots) / num_snapshots)
        angle_rad = np.radians(angle)
        steering_vector = np.exp(1j * 2 * np.pi * (sensor_positions_x[:, np.newaxis] * np.cos(angle_rad) +
                                                   sensor_positions_y[:, np.newaxis] * np.sin(angle_rad)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + 200 * noise
    
class RealTimePlotter(QtWidgets.QMainWindow):
    """
    A PyQt-based GUI window for real-time data visualization of direction of arrival (DOA) and FFT plots.

    Attributes:
    timer : QtCore.QTimer
        QTimer object responsible for triggering the update of plots at regular intervals.
    """
    def __init__(self):
        """
        Initializes the RealTimePlotter instance.
        """
        super().__init__()
        
        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(0)

    def initUI(self):
        """
        Sets up the user interface (UI) layout.
        """
        self.setWindowTitle('Real-Time Data Visualization')
        
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QGridLayout(self.centralWidget)
        
        self.doa_plot = pg.PlotWidget(title="Direction of Arrival")
        self.doa_plot.setAspectLocked(True) 
        self.doa_plot.showAxis('left', False) 
        self.doa_plot.showAxis('bottom', False)
        self.layout.addWidget(self.doa_plot, 0, 0, 1, 1)
        
        self.fft_plot_0 = pg.PlotWidget(title="FFT Antenna 0")
        self.fft_curve_0 = self.fft_plot_0.plot(pen='r')
        self.layout.addWidget(self.fft_plot_0, 0, 1, 1, 1)
        
        self.fft_plot_1 = pg.PlotWidget(title="FFT Antenna 1")
        self.fft_curve_1 = self.fft_plot_1.plot(pen='g')
        self.layout.addWidget(self.fft_plot_1, 1, 0, 1, 1)
        
        self.fft_plot_2 = pg.PlotWidget(title="FFT Antenna 2")
        self.fft_curve_2 = self.fft_plot_2.plot(pen='b')
        self.layout.addWidget(self.fft_plot_2, 1, 1, 1, 1)

        self.fft_plot_3 = pg.PlotWidget(title="FFT Antenna 3")
        self.fft_curve_3 = self.fft_plot_3.plot(pen='y')  # Changed to yellow
        self.layout.addWidget(self.fft_plot_3, 2, 0, 1, 1)

        self.fft_plot_4 = pg.PlotWidget(title="FFT Antenna 4")
        self.fft_curve_4 = self.fft_plot_4.plot(pen='c')  # Changed to cyan
        self.layout.addWidget(self.fft_plot_4, 2, 1, 1, 1)

        self.create_polar_grid()
        self.doa_curve = None  # Initialize doa_curve to None

    def create_polar_grid(self):
        """
        Creates a polar grid on the Direction of Arrival (DOA) plot.
        The grid consists of a circle representing the outer boundary and direction lines
        spaced every 20 degrees, along with labeled text items indicating the angle in degrees.
        """
        angle_ticks = np.linspace(0, 2 * np.pi, 360)
        radius = 1

        #Plot the circle
        x = radius * np.cos(angle_ticks)
        y = radius * np.sin(angle_ticks)
        self.doa_plot.plot(x, y, pen=pg.mkPen('b', width=2))

        #Add direction lines (every 20 degrees)
        for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            x_line = [0, radius * np.cos(angle)]
            y_line = [0, radius * np.sin(angle)]
            self.doa_plot.plot(x_line, y_line, pen=pg.mkPen('r', width=1))

        #Add labels (every 20 degrees)
        for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            text = f'{int(np.ceil(np.degrees(angle)))}°'
            text_item = pg.TextItem(text, anchor=(0.5, 0.5))
            text_item.setPos(1.1 * np.cos(angle), 1.1 * np.sin(angle))
            self.doa_plot.addItem(text_item)

    def plot_doa_circle(self, doa_data):
        """
        Plots the direction of arrival (DOA) circle based on provided DOA data.
        
        Args:
        - doa_data (numpy.ndarray): Array of DOA data values, typically normalized between 0 and 1.
        If len(doa_data) == 180, the data is mirrored to cover 360 degrees.
        """
        angles = np.linspace(0, 2 * np.pi, len(doa_data))
        x_values = doa_data * np.cos(angles)
        y_values = doa_data * np.sin(angles)

        #Close the polar plot loop
        x_values = np.append(x_values, [0])
        y_values = np.append(y_values, [0])

        if self.doa_curve is not None:
            self.doa_plot.removeItem(self.doa_curve)

        self.doa_curve = self.doa_plot.plot(x_values, y_values, pen=pg.mkPen('y', width=2), 
                                            fillLevel=0, brush=(255, 255, 0, 50))

    def update_plots(self):
        """
        Updates the direction of arrival (DOA) and FFT plots with real-time data.

        Reads data from the `kraken` instance using `kraken.read_streams()`.
        Performs DOA estimation using the MUSIC algorithm, computes FFTs of received signals,
        and updates the corresponding PlotWidget curves (`doa_curve`, `fft_curve_0`, `fft_curve_1`, `fft_curve_2`).
        """

        if kraken.simulation:
            if kraken.circular:
                kraken.buffer = signals_linear(kraken.simulation_frequencies, kraken.simulation_angles ,kraken.num_devices, kraken.num_samples, x, antenna_distance)
            else:
                kraken.buffer = signals_circular(kraken.simulation_frequencies, kraken.simulation_angles ,kraken.num_devices, kraken.num_samples, x, y, antenna_distance)
        else:
            kraken.read_streams()

        kraken.apply_filter()

        #doa_data = kraken.capon()
        #doa_data = kraken.lpm(2)
        #doa_data = kraken.bartlett()
        #doa_data = kraken.mem()
        doa_data = kraken.music(1)
        doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        
        freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.sample_rate)
        
        ant0 = np.abs(fft(kraken.buffer[0]))
        ant1 = np.abs(fft(kraken.buffer[1]))
        ant2 = np.abs(fft(kraken.buffer[2]))
        ant3 = np.abs(fft(kraken.buffer[3]))
        ant4 = np.abs(fft(kraken.buffer[4]))  
        
        self.plot_doa_circle(doa_data)
        self.fft_curve_0.setData(freqs, ant0)
        self.fft_curve_1.setData(freqs, ant1)
        self.fft_curve_2.setData(freqs, ant2)
        self.fft_curve_3.setData(freqs, ant3)
        self.fft_curve_4.setData(freqs, ant4)

        print(np.argmax(doa_data))
        # print(doa_data)
        kraken.buffer = np.zeros((kraken.num_devices, kraken.num_samples), dtype=np.complex64)

if __name__ == '__main__':
    num_samples = 1024*128
    sample_rate = 1.024e6
    center_freq = 434.4e6
    bandwidth =  2e5 
    gain = 40
    circular = 0
    
    if circular:
        # Circular setup
        ant0 = [1,    0]
        ant1 = [0.3090,    0.9511]
        ant2 = [-0.8090,    0.5878]
        ant3 = [-0.8090,   -0.5878]
        ant4 = [0.3090,   -0.9511]
        y = np.array([ant0[1], ant1[1], ant2[1], ant3[1], ant4[1]])
        x = np.array([ant0[0], ant1[0], ant2[0], ant3[0], ant4[0]])
        antenna_distance = 0.148857 # (radius) actual antenna distance: 0.175
    
    else:
        # Linear Setup
        y = np.array([0, 0, 0, 0, 0])
        x = np.array([0, 1, 2, 3, 4])
        antenna_distance = 0.175

    kraken = KrakenReceiver(center_freq, num_samples, sample_rate, bandwidth, gain, 
                            antenna_distance, x, y, num_devices=5, circular = circular,
                            simulation = 1, simulation_angles = [0], simulation_frequencies = [center_freq],
                            f_type = 'FIR', detection_range=360)
    
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    sys.exit(app.exec_())