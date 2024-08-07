import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import numpy as np
import scipy.signal as signal
from scipy.fft import fft
import direction_estimation as pa
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore



class KrakenSim():

    def __init__(self, center_freq, num_samples, sample_rate, gain, 
                 antenna_distance, x, y, array_type, num_devices=5, circular = 0,
                simulation_angles = [0], simulation_frequencies = [434.4e6], simulation_distances = [100], simulation_noise = 1e2,
                 f_type = 'FIR', detection_range = 360, music_dim = 2):
        
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.gain = gain
        self.f_type = f_type

        self.simulation_angles = simulation_angles
        self.simulation_frequencies = simulation_frequencies
        self.simulation_distances = simulation_distances
        self.simulation_noise = simulation_noise
        self.circular = circular
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.antenna_distance = antenna_distance
        self.array_type = array_type
        self.detection_range = detection_range
        self.music_dim = music_dim
           
        if self.circular:
            self.buffer = signals_arbitrary(self.simulation_frequencies, self.simulation_angles ,self.num_devices, self.num_samples, self.x, self.y, noise_power = self.simulation_noise)
            self.offs = 180.0
        else:
            self.buffer = signals_linear2(self.simulation_frequencies, self.simulation_angles, self.simulation_distances, self.num_devices, self.num_samples, self.x, noise_power = self.simulation_noise)
            self.offs = -90.0
        
        
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
            # Convert to discrete-time system
            dt = 1e-6
            discrete_system = signal.cont2discrete((num, den), dt)
            #self.b, self.a = discrete_system[0], discrete_system[1]
            self.b = np.array(discrete_system[0].flatten(), dtype=np.float64)
            self.a = np.array(discrete_system[1].flatten(), dtype=np.float64)    

    def apply_filter(self):
        if self.f_type == 'none': 
            pass
        elif self.f_type == 'LTI':
            self.buffer = signal.lfilter(self.b, self.a, self.buffer)
        elif self.f_type == 'butter':
            self.buffer = signal.sosfilt(self.filter, self.buffer)
        elif self.f_type == 'FIR':
            self.buffer = signal.lfilter(self.filter, 1.0, self.buffer)

    def music(self, signal_dimension, index = [0,-1]):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        antenna_distance_ula = 0.175

        if self.array_type == "ULA":
            spatial_corr_matrix = pa.spatial_correlation_matrix(self.buffer, self.num_samples)
            scanning_vectors = pa.gen_scanning_vectors_linear(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2 + self.offs, self.detection_range/2 + self.offs, 0.25))
            spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
            doa = pa.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, signal_dimension=signal_dimension)
        else:
            uca = pa.gen_scanning_vectors_circular(self.num_devices, antenna_distance, 
                                               self.center_freq, np.arange(-self.detection_range/2 , self.detection_range/2))
            spatial_corr_matrix = pa.spatial_correlation_matrix(self.buffer, self.num_samples)
            #spatial_corr_matrix = phase_mode_transform(uca, ula, spatial_corr_matrix)
            doa = pa.DOA_MUSIC(spatial_corr_matrix, uca, signal_dimension=signal_dimension)
        
        return doa
    

def whiten_transform(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    w = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return w @ A @ w.T

def phase_mode_transform(ula, uca, corr):
    Tr = ula @ uca.conj().T @ np.linalg.inv(uca @ uca.conj().T)

    # n = data.shape[1]
    # new_n = (n // 5) * 5
    
    # # Trim the matrix to be (5, new_n) if necessary
    # trimmed_matrix = data[:, :new_n] if new_n < n else data

    # reshaped_matrix = trimmed_matrix.reshape((5, n // 5, 5)).transpose(1, 0, 2)
    # (print(reshaped_matrix.shape))
    
    # # Apply the transformation to each (5, 5) block
    # transformed_blocks = Tr @ reshaped_matrix @ Tr.conj().T
    
    # # Reshape back to (5, n)
    # transformed_matrix = transformed_blocks.transpose(1, 0, 2).reshape(5, new_n)

    transformed_matrix = Tr @ corr @ Tr.conj().T

    return transformed_matrix


def signals_linear(frequencies, angles, num_sensors, num_snapshots, antenna_positions, wavelength=1.0, noise_power=1e1):
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
    
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    frequency_offset = frequencies[0]

    for f, angle in zip(frequencies, angles):
        f_cal = f - frequency_offset
        signal = np.exp(1j * 2 * np.pi * f_cal * np.arange(num_snapshots) / num_snapshots)
        steering_vector = np.exp(1j * 2 * np.pi * antenna_positions[:, np.newaxis] * np.sin(np.radians(angle)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + noise


def signals_circular(frequencies, angles, num_sensors, num_snapshots, x, y, wavelength=1.0, noise_power=1e1):
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
    
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    frequency_offset = frequencies[0]

    for f, angle in zip(frequencies, angles):
        f_cal = f - frequency_offset
        signal = np.exp(1j * 2 * np.pi * f_cal * np.arange(num_snapshots) / num_snapshots)
        angle_rad = np.radians(angle)
        steering_vector = np.exp(1j * 2 * np.pi * (x[:, np.newaxis] * np.cos(angle_rad) +
                                                   y[:, np.newaxis] * np.sin(angle_rad)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + noise

def signals_linear2(frequencies, angles, distances, num_sensors, num_snapshots, antenna_positions, wavelength=1.0, noise_power=1e-3):
    """
    Generates signals received by sensor array.

    Parameters:
    frequencies : list
        List of frequencies (in Hz) of the transmitted signals.
    angles : list
        List of angles (in degrees) of arrival corresponding to each frequency.
    distances : list
        List of distances to the signal sources corresponding to each frequency.
    num_sensors : int
        Number of sensors in the array.
    num_snapshots : int
        Number of signal snapshots to generate.
    antenna_positions : numpy.ndarray
        Positions of the antennas in the array.
    wavelength : float, optional
        Wavelength of the transmitted signals (default is 1.0).
    noise_power : float, optional
        Power of additive Gaussian noise (default is 1e-1).

    Returns:
    numpy.ndarray
        2D array of complex numbers representing received signals at each sensor
        over time (shape: (num_sensors, num_snapshots)).
    """
    
    frequency_offset = frequencies[0] 
    
    c = 3e8  # Speed of light in m/s
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)

    for f, angle, distance in zip(frequencies, angles, distances):
        
        
        # Offset calibration
        f = f - frequency_offset
        
        # Time vector
        t = np.arange(num_snapshots)
        
        # Generate the baseband signal at the desired frequency
        signal = np.exp(1j * 2 * np.pi * f * t / num_snapshots)
        
        # Create the steering vector
        steering_vector = np.exp(1j * 2 * np.pi * antenna_positions[:, np.newaxis] * np.sin(np.radians(angle)) / wavelength)
        
        # Calculate time delay based on distance
        time_delay = distance / c
        phase_delay = np.exp(-1j * 2 * np.pi * f * time_delay)
        
        # Adjust signal for distance (attenuation and phase delay)
        attenuation = 1 # 1 / (distance ** 2)
        
        # Add the signal from this source to the total signals
        signals += attenuation * steering_vector @ (phase_delay * signal)[np.newaxis, :]

    # Generate and add noise
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    
    return signals + 5 * noise
    
def signals_arbitrary(frequencies, angles, num_sensors, num_snapshots, x, y, wavelength=1.0, noise_power=1e-3):
    """
    Generates signals received by an arbitrary sensor array.

    Parameters:
    frequencies : list
        List of frequencies (in Hz) of the transmitted signals.
    angles : list
        List of angles (in degrees) of arrival corresponding to each frequency.
    num_sensors : int
        Number of sensors in the array.
    num_snapshots : int
        Number of signal snapshots to generate.
    x : numpy.ndarray
        1D array of x coordinates of the sensors.
    y : numpy.ndarray
        1D array of y coordinates of the sensors.
    wavelength : float, optional
        Wavelength of the transmitted signals (default is 1.0).
    noise_power : float, optional
        Power of additive Gaussian noise (default is 1e-3).

    Returns:
    numpy.ndarray
        2D array of complex numbers representing received signals at each sensor
        over time (shape: (num_sensors, num_snapshots)).
    """
    
    # Combine x and y coordinates into a single array of positions
    antenna_positions = np.vstack((x, y)).T
    
    # Initialize the array to store the received signals
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    
    # Reference frequency for baseband conversion
    f_cal = frequencies[0]
    
    # Generate the received signal for each frequency and angle pair
    for f, angle in zip(frequencies, angles):
        # Calculate the baseband signal
        f_baseband = f - f_cal
        time = np.arange(num_snapshots)
        signal = np.exp(1j * 2 * np.pi * f_baseband * time / num_snapshots)
        
        # Convert the angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate the steering vector for the given angle
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        steering_vector = np.exp(1j * 2 * np.pi * (antenna_positions @ direction_vector) / wavelength)
        
        # Add the contribution of this signal to the overall received signal
        signals += steering_vector[:, np.newaxis] * signal[np.newaxis, :]
    
    # Additive white Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    
    return signals + 1 * noise

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
        
        # self.fft_plot_1 = pg.PlotWidget(title="FFT Antenna 1")
        # self.fft_curve_1 = self.fft_plot_1.plot(pen='g')
        # self.layout.addWidget(self.fft_plot_1, 1, 0, 1, 1)
        
        # self.fft_plot_2 = pg.PlotWidget(title="FFT Antenna 2")
        # self.fft_curve_2 = self.fft_plot_2.plot(pen='b')
        # self.layout.addWidget(self.fft_plot_2, 1, 1, 1, 1)

        # self.fft_plot_3 = pg.PlotWidget(title="FFT Antenna 3")
        # self.fft_curve_3 = self.fft_plot_3.plot(pen='y')  # Changed to yellow
        # self.layout.addWidget(self.fft_plot_3, 2, 0, 1, 1)

        # self.fft_plot_4 = pg.PlotWidget(title="FFT Antenna 4")
        # self.fft_curve_4 = self.fft_plot_4.plot(pen='c')  # Changed to cyan
        # self.layout.addWidget(self.fft_plot_4, 2, 1, 1, 1)

        self.doa_cartesian_plot = pg.PlotWidget(title="Direction of Arrival (Cartesian)")
        self.doa_cartesian_curve = self.doa_cartesian_plot.plot(pen=pg.mkPen(pg.mkColor(70,220,0), width=2))
        self.doa_cartesian_plot.showAxis('top', True)
        
        ax = self.doa_cartesian_plot.getAxis('bottom')
        if kraken.detection_range == 180:
            num_ticks = 19
        else:
            num_ticks = 25
        ang_ticks = np.linspace(-kraken.detection_range / 2, kraken.detection_range / 2, num_ticks)
        ax.setTicks([[(round(v), str(round(v))) for v in ang_ticks]])
        ax2 = self.doa_cartesian_plot.getAxis('top')
        ax2.setTicks([[(round(v), str(round(v))) for v in ang_ticks]])
        self.layout.addWidget(self.doa_cartesian_plot, 0, 4, 1, 1) 

        self.create_polar_grid()
        self.doa_curve = None  # Initialize doa_curve to None

    def create_polar_grid(self):
        """
        Creates a polar grid on the Direction of Arrival (DOA) plot.
        The grid consists of a circle representing the outer boundary and direction lines
        spaced every 20 degrees, along with labeled text items indicating the angle in degrees.
        """
        rad_limit = np.radians(kraken.detection_range)
        if kraken.detection_range > 180:
            endpoint = False
        else:
            endpoint = True
        
        angle_ticks = np.linspace(0, rad_limit, kraken.detection_range)
        radius = 1

        #Plot the circle
        x = radius * np.cos(angle_ticks)
        y = radius * np.sin(angle_ticks)
        self.doa_plot.plot(x, y, pen=pg.mkPen('dark green', width=2))

        #Add direction lines (every 20 degrees)
        for angle in np.linspace(0, rad_limit, 19, endpoint=endpoint):
            x_line = [0, radius * np.cos(angle)]
            y_line = [0, radius * np.sin(angle)]
            self.doa_plot.plot(x_line, y_line, pen=pg.mkPen('dark green', width=1))

        #Add labels (every 20 degrees)
        for angle in np.linspace(0, rad_limit, 19, endpoint=endpoint):
            text = f'{int(round(np.degrees(angle-rad_limit/2), -1))}°'
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
        rad_limit = np.radians(kraken.detection_range)
        #print(f'rad_limit = {rad_limit}')
        
        angles = np.linspace(0, rad_limit, len(doa_data))
        x_values = doa_data * np.cos(angles)
        y_values = doa_data * np.sin(angles)

        #Close the polar plot loop
        x_values = np.append(x_values, [0])
        y_values = np.append(y_values, [0])

        if self.doa_curve is not None:
            self.doa_plot.removeItem(self.doa_curve)

        self.doa_curve = self.doa_plot.plot(x_values, y_values, pen=pg.mkPen(pg.mkColor(70,220,0), width=2), 
                                            fillLevel=0, brush=(255, 255, 0, 50))

    def update_plots(self):
        """
        Updates the direction of arrival (DOA) and FFT plots with real-time data.

        Reads data from the `kraken` instance using `kraken.read_streams()`.
        Performs DOA estimation using the MUSIC algorithm, computes FFTs of received signals,
        and updates the corresponding PlotWidget curves (`doa_curve`, `fft_curve_0`, `fft_curve_1`, `fft_curve_2`).
        """
        if kraken.circular:
            kraken.buffer = signals_arbitrary(kraken.simulation_frequencies, kraken.simulation_angles ,kraken.num_devices, kraken.num_samples, kraken.x, kraken.y, noise_power = kraken.simulation_noise)
        else:
            kraken.buffer = signals_linear2(kraken.simulation_frequencies, kraken.simulation_angles, kraken.simulation_distances, kraken.num_devices, kraken.num_samples, kraken.x, noise_power = kraken.simulation_noise)
            #kraken.buffer = signals_linear(kraken.simulation_frequencies, kraken.simulation_angles, kraken.num_devices, kraken.num_samples, kraken.x, noise_power = kraken.simulation_noise)

        kraken.apply_filter()

        doa_data = kraken.music(kraken.music_dim)
        doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        
        
        freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.sample_rate)
        ant0 = np.abs(fft(kraken.buffer[0]))
        # ant1 = np.abs(fft(kraken.buffer[1]))
        # ant2 = np.abs(fft(kraken.buffer[2]))
        # ant3 = np.abs(fft(kraken.buffer[3]))
        # ant4 = np.abs(fft(kraken.buffer[4]))  
        
        self.plot_doa_circle(doa_data)
        self.fft_curve_0.setData(freqs, ant0)
        # self.fft_curve_1.setData(freqs, ant1)
        # self.fft_curve_2.setData(freqs, ant2)
        # self.fft_curve_3.setData(freqs, ant3)
        # self.fft_curve_4.setData(freqs, ant4)

        self.doa_cartesian_curve.setData(np.linspace(-kraken.detection_range / 2, kraken.detection_range / 2, len(doa_data)), doa_data)

        print(np.argmax(doa_data))

if __name__ == '__main__':
    num_samples = 1024*64
    sample_rate = 2.048e6
    center_freq = 434.4e6
    bandwidth =  2e5 
    gain = 40
    circular = 1
    
    if circular:
        # Circular setup
        ant0 = [1,    0]
        ant1 = [0.3090,    0.9511]
        ant2 = [-0.8090,    0.5878]
        ant3 = [-0.8090,   -0.5878]
        ant4 = [0.3090,   -0.9511]
        y = np.array([ant0[1], ant1[1], ant2[1], ant3[1], ant4[1]])
        x = np.array([ant0[0], ant1[0], ant2[0], ant3[0], ant4[0]])
        antenna_distance =  0.35
        antenna_distance = antenna_distance / 2.0 / np.sin(36.0*np.pi/180.0) # distance = 0.175 -> radius = 0.148857 
    
    else:
        # Linear Setup
        y = np.array([0, 0, 0, 0, 0])
        x = np.array([0, 1, 2, 3, 4])
        antenna_distance = 0.175

    kraken = KrakenSim(center_freq, num_samples, sample_rate, gain,    
                            antenna_distance, x, y, "UCA", num_devices = 5, circular = circular,
                            simulation_angles = [30], simulation_frequencies = [center_freq], simulation_noise = 1e-2,
                            f_type = 'FIR', detection_range = 360, music_dim = 1)
    
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    sys.exit(app.exec_())
    