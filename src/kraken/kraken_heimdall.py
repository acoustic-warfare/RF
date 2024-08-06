import os
import numpy as np
import h5py
import sys
import socket
import scipy.signal as signal
import direction_estimation as de
import pyqtgraph as pg  
import _thread
import time
import logging
import gi
#os.environ['GST_DEBUG'] = "3" #Uncomment to enable GST debug logs
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib
from threading import Lock
from struct import pack
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui
from scipy.fft import fft
from config import read_kraken_config
from datetime import datetime
from shmemIface import inShmemIface
from iq_header import IQHeader



class KrakenReceiver():
    """
    KrakenReceiver class for managing data acquisition and signal processing for a KrakenSDR.

    """
    def __init__(self):

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='kraken.log',
            filemode='w'  
        )
        self.logger = logging.getLogger(__name__)

        
        center_freq, num_samples, sample_rate, antenna_distance, x, y, array_type, f_type, waraps = read_kraken_config()
        self.daq_center_freq = center_freq  # MHz
        self.num_samples = num_samples
        self.daq_sample_rate = sample_rate
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.array_type = array_type
        self.num_antennas = x.size
        self.f_type = f_type
        if array_type == 'ULA':
            self.detection_range = 180
            self.scanning_vectors = de.gen_scanning_vectors_linear(self.num_antennas, self.x, self.y, 
                                                            np.arange(-self.detection_range/2 - 90, self.detection_range/2 -90)) 
            
        else:
            self.detection_range = 360
            self.scanning_vectors = de.gen_scanning_vectors_circular(self.num_antennas, antenna_distance, 
                                               self.daq_center_freq, np.arange(-self.detection_range/2 , self.detection_range/2))
        
        self.waraps = waraps

        #Shared memory setup
        self.daq_shmem_control_path = "heimdall_daq_fw/Firmware/_data_control/"
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

        self.init_filter()

    def init_filter(self):
        """
        Initialize the filter based on the specified filter type.

        This function initializes a digital filter according to the `f_type` attribute of the instance. 
        The filter can be of three types: Butterworth ('butter'), Finite Impulse Response ('FIR'), 
        or Linear Time-Invariant ('LTI'). The function sets the appropriate filter coefficients for the selected filter type.

        """
        if self.f_type == 'butter':
            #Build digital filter
            fc = self.daq_center_freq
            fs = 4*fc
            fn = 0.5*fs
            f_bandwidth = 0.6*fc
            wn = [(f_bandwidth/2) / fn]
            wn = [np.finfo(float).eps, (f_bandwidth/2) / fn] 
            sos = signal.butter(0, wn, btype='lowpass', output='sos')
            self.filter = sos

        elif self.f_type == 'FIR':
            #Design a FIR filter using the firwin function
            numtaps = 51  # Number of filter taps (filter length)
            fc = self.daq_center_freq
            fs = 4*fc
            bandwidth = 0.1*fc
            highcut = bandwidth/2  # Upper cutoff frequency (Hz)
            taps = signal.firwin(numtaps, [highcut], fs=fs, pass_zero=True)
            self.filter = taps

        elif self.f_type == 'LTI':
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
            RuntimeError("Control interface init failed")
                
    def set_center_freq(self, center_freq):
        """
        Set the center frequency of the DAQ.

        Parameters:
        -----------
        center_freq : int
            The center frequency in MHz.
        """
        self.daq_center_freq = int(center_freq)
        #Set center frequency
        cmd = "FREQ"
        freq_bytes = pack("Q", int(center_freq))
        msg_bytes = cmd.encode() + freq_bytes + bytearray(116)
        try:
            self.logger.info("Sending center frequency configuration message")
            _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
        except:
            self.logger.error("Failed sending message to HWC")


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
            self.logger.info("Sending gain configuration message")
            _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
        except:
            self.logger.error("Failed sending message to HWC")
            
    def ctr_iface_communication(self, msg_bytes):
        """
        Handles communication on the control interface with the DAQ FW

        Parameters:
        -----------
            :param: msg: Message bytes, that will be sent ont the control interface
            :type:  msg: Byte array
        """
        self.ctr_iface_thread_lock.acquire()
        self.logger.info("Sending hwc message")
        self.ctr_iface_socket.send(msg_bytes)

        # Waiting for the command to take effect
        reply_msg_bytes = self.ctr_iface_socket.recv(128)

        self.logger.info("Control interface communication finished")
        self.ctr_iface_thread_lock.release()

        status = reply_msg_bytes[0:4].decode()
        if status == "FNSD":
            self.logger.info("Reconfiguration succesfully finished")
            
        else:
            self.logger.error(f"Failed to set the requested parameter, reply: {status}")
            

    def init_data_iface(self):
        """
        Open shared memory interface to capture the DAQ firmware output.
        """
        self.in_shmem_iface = inShmemIface(
            "delay_sync_iq", self.daq_shmem_control_path, read_timeout=5.0
        )
        if not self.in_shmem_iface.init_ok:
            self.in_shmem_iface.destory_sm_buffer()
            self.logger.error("Shared Memory Init Failed")
        else:
            self.logger.info("Successfully Initilized Shared Memory")

    def get_iq_online(self):
        """
        Obtain a new IQ data frame through the Ethernet IQ data or the shared memory interface.

        Returns:
        --------
        frame_type : int
            Type of the frame received.
        """

        active_buff_index = self.in_shmem_iface.wait_buff_free()
        if active_buff_index < 0 or active_buff_index > 1:
            # If we cannot get the new IQ frame then we zero the stored IQ header
            self.iq_header = IQHeader()
            self.iq_samples = np.empty(0)
            self.logger.critical(f"Terminating.., signal: {active_buff_index}")

        buffer = self.in_shmem_iface.buffers[active_buff_index]

        iq_header_bytes = buffer[:1024].tobytes()
        self.iq_header.decode_header(iq_header_bytes)

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
        
        if self.num_antennas != 5:
            self.iq_samples = self.iq_samples[:self.num_antennas :]

        self.in_shmem_iface.send_ctr_buff_ready(active_buff_index)

        return self.iq_header.frame_type

    def apply_filter(self):
        """
        Apply the configured filter to the IQ samples.

        """
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
        spatial_corr_matrix = de.spatial_correlation_matrix(self.iq_samples, self.num_samples)
        sig_dim = de.infer_signal_dimension(spatial_corr_matrix)
        if self.array_type == 'ULA':
            spatial_corr_matrix = de.forward_backward_avg(spatial_corr_matrix)
        doa = de.DOA_MUSIC(spatial_corr_matrix, self.scanning_vectors, sig_dim)

        return doa

    def record_samples(self):
        """
        Record IQ samples to a HDF5 file.

        """
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

        if kraken.waraps:
            Gst.init(None)
            kraken.logger.info("GStreamer initialized successfully")
            # 1725, 760)
            self.pipeline = Gst.parse_launch(
                " appsrc name=doa is_live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,width=1720,format=RGB,height=760 "
                " ! videoconvert ! x264enc tune=zerolatency speed-preset=superfast bitrate=4000"
                " ! queue ! flvmux streamable=true ! rtmp2sink location=rtmp://ome.waraps.org/app/KrakenSDR"
                )
            
            self.appsrc = self.pipeline.get_by_name('doa')
            self.start_time = time.time()
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                kraken.logger.critical("Unable to set the pipeline to the playing state")
                raise RuntimeError("Unable to set the pipeline to the playing state.")
                

    def grab_frame(self):
        """
        Capture the current frame of the widget and convert it to a NumPy array.

        This function captures the current visual state of the widget using Qt's rendering capabilities. It converts the captured image into a QImage in RGB format and then transforms it into a NumPy array.

        Returns:
            np.ndarray: A 3D NumPy array representing the captured frame, with shape (height, width, 3).
        """
        pixmap = QtGui.QPixmap(self.size())
        self.render(pixmap)

        qimage = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        print(ptr.getsize())
        ptr.setsize(qimage.byteCount())
        print(qimage.bytesPerLine())
        print(ptr.getsize())
        arr = np.array(ptr).reshape(height, width, 3)
        return arr 
    
    def send_frame(self):
        """
        Capture the current frame and send it as a GStreamer buffer.

        This function captures the current frame using the `grab_frame` method, converts the frame to a byte array, 
        and sends it to a GStreamer pipeline. It timestamps the buffer and sets its duration for a 30 FPS stream.

        Returns:
            bool: Always returns True.
        """
        frame = self.grab_frame()
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        timestamp = (time.time() - self.start_time) * Gst.SECOND
        buf.pts = timestamp
        buf.dts = timestamp
        buf.duration = Gst.SECOND // 30
        buf.fill(0, data)
        self.appsrc.emit('push-buffer', buf)
        kraken.logger.info("Sent frame to waraps")
        return True

    def initUI(self):
        """
        Sets up the user interface (UI) layout.
        """
        self.setWindowTitle('Real-Time Data Visualization')
        self.setGeometry(2040, 115, 1720, 760)
        
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QGridLayout(self.centralWidget)
        
        self.doa_plot = pg.PlotWidget(title="Direction of Arrival")
        self.doa_plot.setAspectLocked(True) 
        self.doa_plot.showAxis('left', False) 
        self.doa_plot.showAxis('bottom', False)
        self.layout.addWidget(self.doa_plot, 0, 0, 1, 4)
        
        # self.fft_plot_0 = pg.PlotWidget(title="FFT Antenna 0")
        # self.fft_curve_0 = self.fft_plot_0.plot(pen='r')
        # self.layout.addWidget(self.fft_plot_0, 0, 1, 1, 1)
        
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
            num_ticks = 24
        else:
            endpoint = True
            num_ticks = 19
        
        angle_ticks = np.linspace(0, rad_limit, kraken.detection_range)
        radius = 1

        #Plot the circle
        x = radius * np.cos(angle_ticks)
        y = radius * np.sin(angle_ticks)
        self.doa_plot.plot(x, y, pen=pg.mkPen('dark green', width=2))

        #Add direction lines (every 20 degrees)
        for angle in np.linspace(0, rad_limit, num_ticks, endpoint=endpoint):
            x_line = [0, radius * np.cos(angle)]
            y_line = [0, radius * np.sin(angle)]
            self.doa_plot.plot(x_line, y_line, pen=pg.mkPen('dark green', width=1))

        #Add labels (every 20 degrees)
        for angle in np.linspace(0, rad_limit, num_ticks, endpoint=endpoint):
            text = f'{int(round(np.degrees(angle-rad_limit/2), 1))}°'
            text_item = pg.TextItem(text, anchor=(0.5, 0.5))
            text_item.setPos(1.1 * np.cos(angle), 1.1 * np.sin(angle))
            self.doa_plot.addItem(text_item)

    def plot_doa_circle(self, doa_data):
        """
        Plots the direction of arrival (DOA) circle based on provided DOA data.
        
        Args:
        - doa_data (numpy.ndarray): Array of DOA data values, normalized between 0 and 1.
        """
        rad_limit = np.radians(kraken.detection_range)
        
        angles = np.linspace(0, rad_limit, len(doa_data))
        x_values = doa_data * np.cos(angles)
        y_values = doa_data * np.sin(angles)

        #Close the polar plot loop
        x_values = np.append(x_values, [0])
        y_values = np.append(y_values, [0])

        if self.doa_curve is not None:
            self.doa_plot.removeItem(self.doa_curve)

        self.doa_curve = self.doa_plot.plot(x_values, y_values, pen=pg.mkPen(pg.mkColor(70,220,0), width=2))

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
            
                
            #freqs = np.fft.fftfreq(kraken.num_samples, d=1/kraken.daq_sample_rate)  
            #ant0 = np.abs(fft(kraken.iq_samples[0]))
            # ant1 = np.abs(fft(kraken.iq_samples[1]))
            # ant2 = np.abs(fft(kraken.iq_samples[2]))
            # ant3 = np.abs(fft(kraken.iq_samples[3]))
            # ant4 = np.abs(fft(kraken.iq_samples[4]))  
                
            
            #self.fft_curve_0.setData(freqs, ant0)
            # self.fft_curve_1.setData(freqs, ant1)
            # self.fft_curve_2.setData(freqs, ant2)
            # self.fft_curve_3.setData(freqs, ant3)
            # self.fft_curve_4.setData(freqs, ant4)
            self.plot_doa_circle(doa_data)
            self.doa_cartesian_curve.setData(np.linspace(-kraken.detection_range / 2, kraken.detection_range / 2, len(doa_data)), doa_data)

            #print(np.argmax(doa_data) - 90) 

        elif frame_type == 1:
            kraken.logger.info("Received Dummy frame")
        elif frame_type == 2:
            kraken.logger.info("Received Ramp frame")
        elif frame_type == 3:
            kraken.logger.info("Received Calibration frame")
        elif frame_type == 4:
            kraken.logger.info("Receiver Trigger Word frame")
        else:
            kraken.logger.info("Received Empty frame")

if __name__ == "__main__":
    kraken = KrakenReceiver()
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    if kraken.waraps:
        GLib.timeout_add(1000 // 30, plotter.send_frame)
    sys.exit(app.exec_())