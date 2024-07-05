import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import sys
import pyargus.directionEstimation as pa
import pyqtgraph as pg
import asyncio
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from rtlsdr import *
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor
from pylab import *

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
    def __init__(self, center_freq, num_samples, sample_rate, gain, antenna_distance, x, y, 
                 num_devices=5, mode = 0, f_type = 'LTI', detection_range = 360):
        
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.gain = gain
        self.f_type = f_type

        modes = ['normal', 'async', 'simulation']
        if mode in modes:
            self.mode = mode
        else: self.mode = modes[mode]

        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.detection_range = detection_range



        if self.mode == 'simulation':
            #self.buffer = signals_linear([self.center_freq], [30] ,self.num_devices, self.num_samples, self.x, antenna_distance)
            self.buffer = signals_circular([self.center_freq], [300] ,self.num_devices, self.num_samples, self.x, self.y, antenna_distance)
        else: 
            if self.mode == 'normal':
                self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)
        
        #self.devices, self.streams = self._setup_devices()
        self.devices = self._setup_devices()

        if f_type == 'butter':
            #Build digital filter
            fc = self.center_freq
            fs = 4*fc
            fn = 0.5*fs
            f_bandwidth = 0.2*fc
            wn = [(f_bandwidth/2) / fn]
            #wn = [np.finfo(float).eps, (f_bandwidth/2) / fn] 
            sos = signal.butter(4, wn, btype='lowpass', output='sos')
            self.filter = sos

        elif f_type == 'FIR':
            #Design a FIR filter using the firwin function
            numtaps = 51  # Number of filter taps (filter length)
            fc = self.center_freq
            fs = 4*fc
            bandwidth = 0.1*fc
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

    def _setup_device(self, serial):
        device = RtlSdr(serial_number=serial)
        device.sample_rate = self.sample_rate
        device.center_freq = self.center_freq
        device.gain = self.gain
        return device
    
    def _setup_devices(self):
        devices = np.zeros(self.num_devices, dtype=object)
        #streams = np.zeros(self.num_devices, dtype=object)
        #loop = asyncio.get_event_loop()

        available_devices = RtlSdr.get_device_serial_addresses()
        selected_devices = (available_devices[1:] + available_devices[:1])[:self.num_devices]

        for i, serial in enumerate(selected_devices):
            device = self._setup_device(serial)
            devices[i] = device
            
            # if self.mode == 'async':
            #     stream = loop.create_task(self._setup_stream(device, i))
            #     streams[i] = stream
            #     return devices, streams
        return devices
    
    # async def _setup_stream(self, device, device_index):
        
    #     async for samples in device.stream():
    #         print(f"Writing to buffer for device {device_index}")
    #         self.buffer[device_index] = samples
            

    # def start_streams(self):
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(asyncio.gather(*self.streams))

    
    def _read_stream(self, device):
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
        
        self.buffer[device] = self.devices[device].read_samples(self.num_samples)
        


    def read_streams(self):
        """
        Reads data from all receiver devices and filters the captured signals.

        Uses a thread pool to read data from multiple devices concurrently,
        then applies a filter to the captured signals stored in `self.buffer`.
        """
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(self._read_stream, i) for i in range(self.num_devices)]
            for future in futures:
                future.result()

    def apply_filter(self):
        if self.f_type == 'none': 
            pass
        elif self.f_type == 'LTI':
            self.buffer = signal.lfilter(self.b, self.a, self.buffer)
        elif self.f_type == 'butter':
            self.buffer = signal.sosfilt(self.filter, self.buffer)
        elif self.f_type == 'FIR':
            self.buffer = signal.lfilter(self.filter, 1.0, self.buffer)
    
    def music(self):
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
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2, self.detection_range/2))
        doa = pa.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, signal_dimension=1)

        return doa
    
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
    return signals + 600 * noise


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
        steering_vector = np.exp(1j * 2 * np.pi * (sensor_positions_x[:, np.newaxis] * np.sin(angle_rad) +
                                                   sensor_positions_y[:, np.newaxis] * np.cos(angle_rad)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + 300 * noise
    
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

        #kraken.start_streams()

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

        #print(kraken.mode)

        if kraken.mode == 'simulation':
            #kraken.buffer = signals_linear([kraken.center_freq], [300] ,kraken.num_devices, kraken.num_samples, x, antenna_distance)
            kraken.buffer = signals_circular([kraken.center_freq], [10] ,kraken.num_devices, kraken.num_samples, x, y, antenna_distance)
        elif kraken.mode == 'normal':
            kraken.read_streams()
            

        kraken.apply_filter()

        doa_data = kraken.music()
        doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        #print(np.sum(kraken.filter)) 
        
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

        #print(doa_data)
        print(np.argmax(doa_data))

if __name__ == '__main__':
    num_samples = 1024*128
    sample_rate = 2.048e6
    center_freq = 434.4e6
    gain = 40

    #Circular setup
    ant0 = [1,    0]
    ant1 = [0.3090,    0.9511]
    ant2 = [-0.8090,    0.5878]
    ant3 = [-0.8090,   -0.5878]
    ant4 = [0.3090,   -0.9511]
    
    y = np.array([ant0[1], ant1[1], ant2[1], ant3[1], ant4[1]])
    x = np.array([ant0[0], ant1[0], ant2[0], ant3[0], ant4[0]])
    antenna_distance = 0.148857 # actual antenna distance: 0.175

    #Linear Setup
    # y = np.array([0, 0, 0, 0, 0])
    # x = np.array([0, 1, 2, 3, 4])
    # antenna_distance = 0.175


    kraken = KrakenReceiver(center_freq, num_samples, 
                           sample_rate, gain, antenna_distance, x, y, num_devices=5, mode = 2, f_type = 'FIR', detection_range=360)
    

    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    sys.exit(app.exec_())


