import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import gi
import os
os.environ['GST_DEBUG'] = "3" #Uncomment to enable GST debug logs
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
import adi
import numpy as np
from pyqtgraph.Qt import QtGui
sys.path.append("/usr/local/lib/python3.10/site-packages")

class LiveSpectrogram(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(np.ndarray)

    """
    Represents the backend of a waterfall plot.

    Attributes:
    frames : int
        Number frames/segments a window contain.
    num_samples : int
        Number of samples that the PlutoSDR recieves in one go and the FFT will be performed on.
    sample_rate : float
        Sampling rate in samples per second.
    sdr : ad9361
        The device used to recieve samples.
    center_freq : int
        The center frequency of the signal.
    bandwidth : float
        Bandwidth of the signal.
    window : numpy.ndarray
        Window contains the data used in the spectrogram.

    """

    def __init__(self , frames, num_samples, sample_rate, sdr, center_freq, bandwidth, ):
        super().__init__()
        self.sdr = sdr
        self.sample_rate = sample_rate
        self.num_samples = num_samples    
        self.center_freq = center_freq
        self.frames = frames
        self.bandwidth = bandwidth
        self.window = self.create_spectrogram()
    

    def create_segment(self, sample):

        """
        Creates a segment of the spectrogram. This is done by performing FFT on the sample and converting it to dB.
        
        Parameters:
        
        sample : numpy.ndarray
            A numpy.ndarray containing the samples that the plutoSDR recieves/saves in it's buffer.
        
        Returns: 

        numpy.ndarray
            A segement containing the amplitudes of the sample in dB.
            
        """

        segment = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sample))**2))
        return segment
    
    def update_spectrogram(self):

        """
        Updates the spectrogram. The oldest line of the spectrogram(window) is updated with a new one by using create_spectrogram and data.
        
        """
        data = self.sdr.rx()
        self.window = np.vstack([self.window[1:], (self.create_segment(data))])
        self.data_ready.emit(self.window)



    def create_spectrogram(self):

        """
        Creates a spectrogram for the waterfall plot to start from. 
         
        Returns:
        numpy.ndarray
            A full spectrogram.
        
        """

        spectrogram = np.zeros((self.frames, self.num_samples))

        for i in range(self.frames):
            spectrogram[i] = self.create_segment(self.sdr.rx())
    
        return spectrogram


    def set_frequency(self, new_frequency):
        """
        Changes the center frequency of the PlutoSDR.
        
        Params:
        new_frequency : float
            The frequency that the PlutoSDR should receive at. The new frequency should have the unit MHz.

        """

        self.center_freq = int(new_frequency*1e6)      
        self.sdr.rx_lo = self.center_freq

    def start(self):
        while True:
            self.update_spectrogram()


class RealTimePlotter(QtWidgets.QMainWindow):
    
    def __init__(self, liveSpectrogram, waraps, streamer):

        """
        Initiliazes the RealTimePlotter
        
        """

        super().__init__()
        self.liveSpectrogram = liveSpectrogram
        self.initUI()
        
        self.waraps = waraps

        self.streamer = streamer
        self.streamer.start_local_stream() 
        
        if self.waraps:
            self.streamer.start_rtmp_stream()

    
    def initUI(self):

        """
        Sets up the user interface for the waterfall plot.
        
        """

        self.setWindowTitle('Waterfall Plot')
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.setGeometry(100, 100, 1280, 720)

        # Creates the initial waterfall plot
        spectrogram = self.liveSpectrogram.window
        self.waterfall = pg.ImageItem(spectrogram)
        self.layout = QtWidgets.QGridLayout(self.centralWidget)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)
        self.p1 = self.win.addPlot()
        self.p1.addItem(self.waterfall)
        self.waterfall.setImage(spectrogram.T)


        # Add labels to the axis
        self.p1.setLabel('left', "Time", units='s')
        self.p1.setLabel('bottom', "Frequency", units='Hz')

        
        # Rescales the x and y axis from pixels to frequency and time
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        freq_scale = self.liveSpectrogram.bandwidth / self.liveSpectrogram.num_samples  # Frequency scale in Hz per pixel
        time_scale = 1/self.liveSpectrogram.frames  # Time scale in seconds per pixel
        tr.scale(freq_scale, -time_scale)
        self.waterfall.setTransform(tr)
        self.waterfall.setPos(self.liveSpectrogram.center_freq - (self.liveSpectrogram.bandwidth / 2),1)  # Position the image correctly
        self.p1.invertY()


        # Create a color map
        colors = [
            (68, 1, 84),            #Purple
            (0, 0, 139),            #Dark Blue
            (59, 82, 139),          #Blue
            (33, 145, 140),         #Turquoise
            (253, 231, 37),         #Yellow
            (238, 75, 43)           #Red

        ]

        # Create a ColorMap object
        color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)

        # Apply the color map to the ImageItem
        self.waterfall.setLookupTable(color_map.getLookupTable())

        bar = pg.ColorBarItem(colorMap=color_map)
        bar.setImageItem(self.waterfall)
        self.win.addItem(bar)

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
        ptr.setsize(qimage.byteCount())
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
        return True

        
    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, spectrogram):
        """
        Updates the waterfall plot. 

        Parameters:
        spectrogram : numpy.ndarray
            spectrogram contains segments. The oldest segment in the spectrogram has been updated with a new one. 
        
        """
        self.waterfall.setImage(spectrogram.T)
        self.waterfall.setPos(self.liveSpectrogram.center_freq - (self.liveSpectrogram.bandwidth / 2), 1)
        if self.waraps: 
            frame = self.grab_frame()
            self.streamer.send_frame(frame.tobytes())

    def stop_waraps_stream(self):
        self.streamer.stop_rtmp_stream()

    def stop_waraps_stream(self):
        self.streamer.start_rtmp_stream()
