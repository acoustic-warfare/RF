import os
import numpy as np
import h5py
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import socket
import scipy.signal as signal
import direction_estimation as de
import pyqtgraph as pg  
import _thread
from threading import Lock
from struct import pack
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from scipy.fft import fft
from config_tri import read_kraken_config
from datetime import datetime
from shmemIface import inShmemIface
from iq_header import IQHeader

class KrakenReceiver():
    def __init__(self):

        center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type, detection_range = read_kraken_config()
        self.daq_center_freq = center_freq  # MHz
        self.num_samples = num_samples
        self.daq_sample_rate = sample_rate
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.num_antennas = x.size
        self.f_type = f_type
        self.detection_range = detection_range
        self.thetas = np.arange(-self.detection_range/2 - 90, self.detection_range/2 - 90)
        self.scanning_vectors = de.gen_scanning_vectors_linear(self.num_antennas, self.x, self.y, self.thetas)

        #Shared memory setup
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        daq_path = os.path.join(os.path.dirname(root_path), "src/kraken/heimdall_daq_fw")
        self.daq_shmem_control_path = os.path.join(os.path.join(daq_path, "Firmware"), "_data_control/")
        self.init_data_iface()

        self.iq_samples = np.empty(0)
        self.iq_header = IQHeader()
        self.file = None

        #Control interface setup
        self.ctr_iface_socket = socket.socket()
        self.ctr_iface_port = 5001
        self.ctr_iface_thread_lock = Lock()
        self.ctr_iface_socket.connect(('127.0.0.1', self.ctr_iface_port))
        self.ctr_iface_init()

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
            bandwidth = 0.1*fc
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
    
    def ctr_iface_init(self):
        """
        Initialize connection with the DAQ FW through the control interface
        """
        # Assembling message
        cmd = "INIT"
        msg_bytes = cmd.encode() + bytearray(124)
        try:
            _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
        except:
            RuntimeError()
                
    def set_center_freq(self, center_freq):

        self.daq_center_freq = int(center_freq)
        #Set center frequency
        cmd = "FREQ"
        freq_bytes = pack("Q", int(center_freq))
        msg_bytes = cmd.encode() + freq_bytes + bytearray(116)
        try:
            print("sending message")
            _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            print("message_sent")
        except:
            RuntimeError("Failed sending message to HWC")


    def set_if_gain(self, gain):
        """
        Configures the IF gain of the receiver through the control interface

        Paramters:
        ----------
            :param: gain: IF gain value [dB]
            :type:  gain: int
        """

        # Check connection
        self.daq_rx_gain = gain
            

        # Set center frequency
        cmd = "GAIN"
        gain_list = [int(gain * 10)] * self.num_antennas
        gain_bytes = pack("I" * self.num_antennas, *gain_list)
        msg_bytes = cmd.encode() + gain_bytes + bytearray(128 - (self.num_antennas + 1) * 4)
        try:
            _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
        except:
            RuntimeError("Failed sending message to HWC")
            
    def ctr_iface_communication(self, msg_bytes):
        """
        Handles communication on the control interface with the DAQ FW

        Parameters:
        -----------

            :param: msg: Message bytes, that will be sent ont the control interface
            :type:  msg: Byte array
        """
        self.ctr_iface_thread_lock.acquire()
        print("Sending control message")
        self.ctr_iface_socket.send(msg_bytes)

        # Waiting for the command to take effect
        reply_msg_bytes = self.ctr_iface_socket.recv(128)

        print("Control interface communication finished")
        self.ctr_iface_thread_lock.release()

        status = reply_msg_bytes[0:4].decode()
        if status == "FNSD":
            print("Reconfiguration succesfully finished")
            
        else:
            raise RuntimeError("Failed to set the requested parameter, reply: {0}".format(status))
            

    def init_data_iface(self):
        # Open shared memory interface to capture the DAQ firmware output
        self.in_shmem_iface = inShmemIface(
            "delay_sync_iq", self.daq_shmem_control_path, read_timeout=5.0
        )
        if not self.in_shmem_iface.init_ok:
            self.in_shmem_iface.destory_sm_buffer()
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


    def music(self, index = [0, None]):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        
        buffer = self.iq_samples[index[0]:index[1]]
        x = self.x[index[0]:index[1]]
        y = self.y[index[0]:index[1]]
        buffer_dim = len(x)

        # print(f'buffer_dim = {buffer_dim}')
        #print(f' x = {x}')
        #smoothed_buffer = self.spatial_smoothing_rewrite(2, 'forward-backward')
        #spatial_corr_matrix = np.dot(smoothed_buffer, smoothed_buffer.conj().T)
        spatial_corr_matrix = de.spatial_correlation_matrix(buffer, self.num_samples)
        spatial_corr_matrix = de.forward_backward_avg(spatial_corr_matrix)
        # scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(-self.detection_range/2 + self.offs, self.detection_range/2 + self.offs))
        scanning_vectors = de.gen_scanning_vectors_linear(buffer_dim, x, y, np.arange(-self.detection_range/2 -90, self.detection_range/2 -90))
        sig_dim = 1 #de.infer_signal_dimension(spatial_corr_matrix, self.num_devices)
        doa = de.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, sig_dim)
        #print(f'doa_max = {np.argmax(doa)}')
        
        return doa

    def music_old(self):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """
        spatial_corr_matrix = de.spatial_correlation_matrix(self.iq_samples, self.num_samples)
        spatial_corr_matrix = de.forward_backward_avg(spatial_corr_matrix)
        sig_dim = de.infer_signal_dimension(spatial_corr_matrix)
        doa = de.DOA_MUSIC(spatial_corr_matrix, self.scanning_vectors, sig_dim)

        return doa

    def record_samples(self):
        if self.file:
            with h5py.File(self.file, 'a') as hf:
                # Check if the dataset exists
                if 'iq_samples' in hf:
                    # Get the existing dataset
                    existing_data = hf['iq_samples']
                    existing_shape = existing_data.shape

                    # Calculate the new shape
                    new_shape = (existing_shape[0], existing_shape[1] + self.num_samples)

                    # Resize the dataset to accommodate new data
                    existing_data.resize(new_shape)

                    # Write new data to the dataset
                    existing_data[:, existing_shape[1]:] = self.iq_samples
        else:
            timestamp = datetime.now()
            self.file = 'recordings/' + timestamp.strftime("%Y-%m-%d_%H:%M:%S") + '.h5'
            with h5py.File(self.file, 'w') as hf:
                hf.create_dataset('iq_samples', data=self.iq_samples, maxshape=(self.iq_samples.shape[0], None))
    
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
        self.doa_curve_2 = None  # Initialize doa_curve to None

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

    def plot_doa_circle(self, doa_data, doa_data_2):
        """
        Plots the direction of arrival (DOA) circle based on provided DOA data.
        
        Args:
        - doa_data (numpy.ndarray): Array of DOA data values, typically normalized between 0 and 1.
        If len(doa_data) == 180, the data is mirrored to cover 360 degrees.
        """
        rad_limit = np.radians(kraken.detection_range)
        
        cal = 0 #np.radians(10)
        cal_2 = 0 # np.radians(-7)
        
        angles = np.linspace(0 + cal, rad_limit + cal, len(doa_data))
        angles_2 = np.linspace(0 + cal_2, rad_limit + cal_2, len(doa_data))
        
        x_values = doa_data * np.cos(angles)
        y_values = doa_data * np.sin(angles)
        x_values_2 = doa_data_2 * np.cos(angles_2)
        y_values_2 = doa_data_2 * np.sin(angles_2)

        #Close the polar plot loop
        x_values = np.append(x_values, [0])
        y_values = np.append(y_values, [0])
        x_values_2 = np.append(x_values_2, [0])
        y_values_2 = np.append(y_values_2, [0])

        if self.doa_curve is not None:
            self.doa_plot.removeItem(self.doa_curve)
        if self.doa_curve_2 is not None:
            self.doa_plot.removeItem(self.doa_curve_2)

        self.doa_curve = self.doa_plot.plot(x_values, y_values, pen=pg.mkPen(pg.mkColor(70,220,0), width=2), 
                                            fillLevel=0, brush=(255, 255, 0, 50))

        self.doa_curve_2 = self.doa_plot.plot(x_values_2, y_values_2, pen=pg.mkPen(pg.mkColor(255,0,0), width=2), 
                                            fillLevel=0, brush=(255, 255, 0, 50))

    import numpy as np

    def find_intersection(self, p1_start, angle1, p2_start, angle2):
        
        p1_start = np.asarray(p1_start)
        p2_start = np.asarray(p2_start) 
        
        # Convert angles (in degrees) to direction vectors
        r = np.array([np.sin(np.radians(angle1)), np.cos(np.radians(angle1))])
        s = np.array([np.sin(np.radians(angle2)), np.cos(np.radians(angle2))])
        
        p = p1_start
        q = p2_start
        
        # Calculate the cross products
        cross_r_s = np.cross(r, s)
        if cross_r_s == 0:
            raise ValueError("Lines are parallel and do not intersect.")
        
        t = np.cross(q - p, s) / cross_r_s
        
        # Calculate the intersection point
        intersection = p + t * r
        return intersection


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
            #kraken.record_samples()

            doa_data = kraken.music()
            doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
            doa_data_2 = kraken.music()
            doa_data_2 = np.divide(np.abs(doa_data_2), np.max(np.abs(doa_data_2)))
                
            freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.daq_sample_rate)  
            ant0 = np.abs(fft(kraken.iq_samples[0]))
            ant1 = np.abs(fft(kraken.iq_samples[1]))
            ant2 = np.abs(fft(kraken.iq_samples[2]))
            ant3 = np.abs(fft(kraken.iq_samples[3]))
            ant4 = np.abs(fft(kraken.iq_samples[4]))  
                
            self.plot_doa_circle(doa_data, doa_data_2)
            self.fft_curve_0.setData(freqs, ant0)
            self.fft_curve_1.setData(freqs, ant1)
            self.fft_curve_2.setData(freqs, ant2)
            self.fft_curve_3.setData(freqs, ant3)
            self.fft_curve_4.setData(freqs, ant4)
            self.doa_cartesian_curve.setData(np.linspace(0, len(doa_data), len(doa_data)), doa_data)


            ang_1 = np.argmax(doa_data) -87 #+7
            ang_2 = np.argmax(doa_data_2) -87 #-10
            #point = self.find_intersection([-0.175, 0], ang_1, [0.175, 0], ang_2)
            #distance = np.sqrt(point[0]**2 + point[1]**2)
            print(f'ang_1 = {ang_1} degrees') 
            print(f'ang_2 = {ang_2} degrees')
            # print(f'doa_1 = {np.argmax(doa_data) - 90} degrees') 
            # print(f'doa_2 = {np.argmax(doa_data_2) - 90} degrees')
            # print(f'point of intersection = {point}') 
            # print(f'distance = {distance} meters')

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