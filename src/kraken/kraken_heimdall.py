import os
import numpy as np
from shmemIface import inShmemIface
from iq_header import IQHeader
import sys
import scipy.signal as signal
import direction_estimation as de
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from scipy.fft import fft
from config import read_kraken_config

class KrakenReceiver():
    def __init__(self):

        center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type = read_kraken_config()

        self.daq_center_freq = center_freq  # MHz
        self.num_samples = num_samples
        self.daq_sample_rate = sample_rate
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.f_type = f_type

        #Shared memory setup
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        daq_path = os.path.join(os.path.dirname(root_path), "RF/src/kraken/heimdall_daq_fw")
        self.daq_shmem_control_path = os.path.join(os.path.join(daq_path, "Firmware"), "_data_control/")
        self.init_data_iface()

        self.iq_samples = np.empty(0)
        self.iq_header = IQHeader()
        self.num_antennas = 0  

        if f_type == 'butter':
            #Build digital filter
            fc = self.daq_center_freq
            fs = 4*fc
            fn = 0.5*fs
            f_bandwidth = 0.6*fc
            wn = [(f_bandwidth/2) / fn]
            wn = [np.finfo(float).eps, (f_bandwidth/2) / fn] 
            sos = signal.butter(0, wn, btype='lowpass', output='sos')
            self.filter = sos

        elif f_type == 'FIR':
            #Design a FIR filter using the firwin function
            numtaps = 51  # Number of filter taps (filter length)
            fc = self.daq_center_freq
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
            self.b = np.array(discrete_system[0].flatten(), dtype=np.float64)
            self.a = np.array(discrete_system[1].flatten(), dtype=np.float64)

    def init_data_iface(self):
        # Open shared memory interface to capture the DAQ firmware output
        self.in_shmem_iface = inShmemIface(
            "delay_sync_iq", self.daq_shmem_control_path, read_timeout=5.0
        )
        if not self.in_shmem_iface.init_ok:
            self.in_shmem_iface.destory_sm_iq_samples()
            raise RuntimeError("Shared Memory Init Failed")
        print("Successfully Initilized Shared Memory")

    def get_iq_online(self):
        """
        This function obtains a new IQ data frame through the Ethernet IQ data or the shared memory interface
        """

        active_buff_index = self.in_shmem_iface.wait_buff_free()
        if active_buff_index < 0 or active_buff_index > 1:
            # If we cannot get the new IQ frame then we zero the stored IQ header
            self.iq_header = IQHeader()
            self.iq_samples = np.empty(0)
            raise RuntimeError(f"Terminating.., signal: {active_buff_index}")

        buffer = self.in_shmem_iface.buffers[active_buff_index]

        iq_header_bytes = buffer[:1024].tobytes()
        self.iq_header.decode_header(iq_header_bytes)

        # Initialization from header - Set channel numbers
        if self.num_antennas == 0:
            self.num_antennas = self.iq_header.active_ant_chs

        incoming_payload_size = (
            self.iq_header.cpi_length * self.iq_header.active_ant_chs * 2 * (self.iq_header.sample_bit_depth // 8)
        )

        shape = (self.iq_header.active_ant_chs, self.iq_header.cpi_length)
        iq_samples_in = buffer[1024 : 1024 + incoming_payload_size].view(dtype=np.complex64).reshape(shape)

        # Reuse the memory allocated for self.iq_samples if it has the
        # correct shape
        if self.iq_samples.shape != shape:
            self.iq_samples = np.empty(shape, dtype=np.complex64)

        np.copyto(self.iq_samples, iq_samples_in)

        self.in_shmem_iface.send_ctr_buff_ready(active_buff_index)

        return self.iq_header.frame_type

    def apply_filter(self):
        if self.f_type == 'none': 
            pass
        elif self.f_type == 'LTI':
            self.iq_samples = signal.lfilter(self.b, self.a, self.iq_samples)
        elif self.f_type == 'butter':
            self.iq_samples = signal.sosfilt(self.filter, self.iq_samples)
        elif self.f_type == 'FIR':
            self.iq_samples = signal.lfilter(self.filter, 1.0, self.iq_samples)

    def music(self):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        #thetas = tuple(np.arange(-180, 180))
        thetas = np.arange(-180, 180)
        spatial_corr_matrix = de.spatial_correlation_matrix(self.iq_samples, self.num_samples)
        spatial_corr_matrix = de.forward_backward_avg(spatial_corr_matrix)
        sig_dim = infer_signal_dimension(spatial_corr_matrix)
        print(sig_dim)
        scanning_vectors = de.gen_scanning_vectors(self.num_antennas, self.x, self.y, thetas)
        doa = de.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, sig_dim)

        return doa
    

def infer_signal_dimension(correlation_matrix, threshold_ratio=0.1):
    # Perform eigenvalue decomposition
    eigenvalues, _ = np.linalg.eig(correlation_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Determine the threshold based on the largest eigenvalue
    threshold = threshold_ratio * eigenvalues[0]
    
    # Count the number of eigenvalues greater than the threshold
    signal_dimension = np.sum(eigenvalues > threshold)
    
    return min(signal_dimension,4)
        
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

        self.doa_cartesian_plot = pg.PlotWidget(title="Direction of Arrival (Cartesian)")
        self.doa_cartesian_curve = self.doa_cartesian_plot.plot(pen=pg.mkPen(pg.mkColor(70,220,0), width=2))
        self.layout.addWidget(self.doa_cartesian_plot, 3, 0, 1, 2)  # Adding Cartesian plot

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
        self.doa_plot.plot(x, y, pen=pg.mkPen('dark green', width=2))

        #Add direction lines (every 20 degrees)
        for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            x_line = [0, radius * np.cos(angle)]
            y_line = [0, radius * np.sin(angle)]
            self.doa_plot.plot(x_line, y_line, pen=pg.mkPen('dark green', width=1))

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
        - doa_data (numpy.ndarray): Array of DOA data values, normalized between 0 and 1.
        """
        angles = np.linspace(0, 2 * np.pi, len(doa_data))
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
        frame_type = kraken.get_iq_online()
        if frame_type == 0:   

            kraken.apply_filter()

            doa_data = kraken.music()
            doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
                
            freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.daq_sample_rate)  
            ant0 = np.abs(fft(kraken.iq_samples[0]))
            ant1 = np.abs(fft(kraken.iq_samples[1]))
            ant2 = np.abs(fft(kraken.iq_samples[2]))
            ant3 = np.abs(fft(kraken.iq_samples[3]))
            ant4 = np.abs(fft(kraken.iq_samples[4]))  
                
            self.plot_doa_circle(doa_data)
            self.fft_curve_0.setData(freqs, ant0)
            self.fft_curve_1.setData(freqs, ant1)
            self.fft_curve_2.setData(freqs, ant2)
            self.fft_curve_3.setData(freqs, ant3)
            self.fft_curve_4.setData(freqs, ant4)
            self.doa_cartesian_curve.setData(np.linspace(0, len(doa_data), len(doa_data)), doa_data)

            #print(np.argmax(doa_data))

        elif frame_type == 1:
            print("Received Dummy frame")
        elif frame_type == 2:
            print("Received Ramp frame")
        elif frame_type == 3:
            print("Received Calibration frame")
        elif frame_type == 4:
            print("Receiver Trigger Word frame")
        else:
            print("Received Empty frame")


kraken = KrakenReceiver()
    
app = QtWidgets.QApplication(sys.argv)
plotter = RealTimePlotter()
plotter.show()
sys.exit(app.exec_())