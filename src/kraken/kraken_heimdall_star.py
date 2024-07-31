import os
import numpy as np
import h5py
import sys
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
from config_star import read_kraken_config
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
        self.thetas = np.arange(0, self.detection_range)
        self.scanning_vectors = de.gen_scanning_vectors(self.num_antennas, self.x, self.y, self.thetas)

        #Shared memory setup
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        daq_path = os.path.join(os.path.dirname(root_path), "RF/src/kraken/heimdall_daq_fw")
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

        print(f'antenna_distance = {antenna_distance}')

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


    def music(self, index = [0, 1, 2, 3, 4]):
        """
        Performs Direction of Arrival (DOA) estimation using the MEM algorithm.

        Returns:
        numpy.ndarray
            Array of estimated DOA angles in degrees.
        """

        x = np.array([self.x[i] for i in index])
        y = np.array([self.y[i] for i in index])

        buffer = np.array([self.iq_samples[i] for i in index])
        buffer_dim = len(x)

        spatial_corr_matrix = de.spatial_correlation_matrix(buffer, self.num_samples)
       
        scanning_vectors = de.gen_scanning_vectors(buffer_dim, x, y, np.arange(0, self.detection_range))
        sig_dim = 1
        doa = de.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, sig_dim)
        
        return doa
    

    def placeholder(self, doa_data):
            ang_0 = np.argmax(doa_data)
            print(f'first angle = {ang_0} degrees') 

            # Makes reference list for the optimal recieving angles for each antenna pair. 
            # The last and second-to-last index belongs to the same antenna pair.
            ang_centers = [[72, 252], [144, 324], [36, 216], [108, 288], [0, 180], [360, 180]] 
           
            # Finds the optimal antenna pair for the current angle.
            ang_center_diffs = [min([abs(c[0] - ang_0), abs(c[1] - ang_0)]) for c in ang_centers]
            ang_min_diff = min(ang_center_diffs)
            index_best = ang_center_diffs.index(ang_min_diff)
            ang_best = ang_centers[index_best]
            print(f'best angles = {ang_best} degrees') 

            #Selects which antennas to use based on the previosly derived optimal antenna pair.
            if index_best == 5:
                index_best = 4
                index_next = 1
            elif index_best > 2:
                index_next = index_best - 3
            else:
                index_next = index_best + 2

            # index_next = (index_best + 2) % 6 if index_best <= 2 else (index_best - 3)
            # if index_best == 5:
            #     index_best, index_next = 4, 1

            print(f'antennas used = {index_best, index_next}') 

            # Preforms a percise DOA approximation using the most optimal pair of antennas.
            doa_data_1 = kraken.music([index_best, index_next])
            doa_data_1 = np.divide(np.abs(doa_data_1), np.max(np.abs(doa_data_1)))            

            # Finds which side of the semi circle that contains the true angle.
            bottom_half =  abs(ang_best[0] - ang_0) > abs(ang_best[1] - ang_0)
            print(f'Left half-plane: {bottom_half}')

            # Removes the un-true (mirrored) DOA-angle
            if bottom_half:
                ang_worst = ang_best[0]
            else:
                ang_worst = ang_best[1]
                
            if ang_worst < 90: 
                doa_data_1[ang_worst +270 : ] = 0.0
                doa_data_1[ : ang_worst +90] = 0.0
            elif ang_worst > 270: 
                doa_data_1[ : ang_worst -270] = 0.0
                doa_data_1[ang_worst -90 : ] = 0.0
            else:
                doa_data_1[ang_worst -90 : ang_worst +90] = 0.0

            # Prints and plots the resulting DOA-angle
            ang_1 = np.argmax(doa_data_1)
            print(f'angle = {ang_1} degrees')
            return doa_data_1

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
        self.fft_curve_0 = self.fft_plot_0.plot(pen='dark green')
        self.layout.addWidget(self.fft_plot_0, 0, 1, 1, 1)

        self.create_polar_grid()
        self.doa_curve = None  # Initialize doa_curve to None
        self.doa_curve_2 = None
        self.doa_curve_3 = None
        self.doa_curves = [self.doa_curve, self.doa_curve_2]
        self.color_list = ['blue', 'red']
        

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
        for angle in np.linspace(0, rad_limit, 18, endpoint=endpoint):
            x_line = [0, radius * np.cos(angle)]
            y_line = [0, radius * np.sin(angle)]
            self.doa_plot.plot(x_line, y_line, pen=pg.mkPen('dark green', width=1))

        #Add labels (every 20 degrees)
        for angle in np.linspace(0, rad_limit, 18, endpoint=endpoint):
            text = f'{int(round(np.degrees(angle), -1))}°'
            text_item = pg.TextItem(text, anchor=(0.5, 0.5))
            text_item.setPos(1.1 * np.cos(angle), 1.1 * np.sin(angle))
            self.doa_plot.addItem(text_item)

    def plot_doa_circle(self, doa_datas):
        """
        Plots the direction of arrival (DOA) circle based on provided DOA data.
        
        Args:
        - doa_data (numpy.ndarray): Array of DOA data values, typically normalized between 0 and 1.
        If len(doa_data) == 180, the data is mirrored to cover 360 degrees.
        """
        rad_limit = np.radians(kraken.detection_range)
        
        angles = [None for datas in doa_datas]
        x_values = [None for datas in doa_datas]
        y_values = [None for datas in doa_datas]
        self.doa_curve = [None for datas in doa_datas]

        for n, data in enumerate(doa_datas):

            angles[n] = np.linspace(0, rad_limit, len(data))
        
            x_values[n] = data * np.cos(angles[n])
            y_values[n] = data * np.sin(angles[n])
            
            #Close the polar plot loop
            x_values[n] = np.append(x_values[n], [0])
            y_values[n] = np.append(y_values[n], [0])

            if self.doa_curves[n] is not None:
                self.doa_plot.removeItem(self.doa_curves[n])

            self.doa_curves[n] = self.doa_plot.plot(x_values[n], y_values[n], pen=pg.mkPen(self.color_list[n], width = 2), 
                                                fillLevel=0, brush= pg.mkBrush(None))


    def update_plots(self):

        """
        This method processes and visualizes direction-of-arrival (DOA) data using the kraken system.

        1. Retrieves and filters IQ data if the frame type is 0.
        2. Performs a broad DOA approximation with all antennas and identifies the initial angle (ang_0).
        3. Determines the optimal antenna pair for precise DOA estimation based on the initial angle.
        4. Identifies which half of the plane contains the true angle and removes mirrored DOA results.
        5. Calculates the final DOA angle and visualizes it on a circular plot.
        6. Computes and plots the FFT of the first antenna's IQ samples.
        """
        
        frame_type = kraken.get_iq_online()
        if frame_type == 0:   

            kraken.apply_filter()
            #kraken.record_samples()

            # Preforms the broad DOA approximation using all antennas.
            doa_data = kraken.music()
            doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        
            doa_data_1 = kraken.placeholder(doa_data)
            doa_datas = [doa_data_1, doa_data]
            self.plot_doa_circle(doa_datas)
            
            # Plots the FFT
            freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.daq_sample_rate)  
            ant0 = np.abs(fft(kraken.iq_samples[0]))
            self.fft_curve_0.setData(freqs, ant0)


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