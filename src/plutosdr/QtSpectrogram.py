import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import adi
import numpy as np
from pyqtgraph.Qt import QtGui

class LiveSpectrogram(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(np.ndarray)

    '''
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
    rx_lo : int
        The center frequency of the signal.
    bandwidth : float
        Bandwidth of the signal.
    window : numpy.ndarray
        Window contains the data used in the spectrogram.

    '''

    def __init__(self , frames, num_samples, sample_rate, sdr, rx_lo, bandwidth):
        super().__init__()
        self.sdr = sdr
        self.sample_rate = sample_rate
        self.num_samples = num_samples    
        self.rx_lo = rx_lo
        self.frames = frames
        self.bandwidth = bandwidth
        self.window = self.create_spectrogram()

    def create_segment(self, sample):

        '''
        Creates a segment of the spectrogram. This is done by performing FFT on the sample and converting it to dB.
        
        Parameters:
        
        sample : numpy.ndarray
            A numpy.ndarray containing the samples that the plutoSDR recieves/saves in it's buffer.
        
        Returns: 

        numpy.ndarray
            A segement containing the amplitudes of the sample in dB.
            
        '''

        segment = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sample))**2))
        return segment
    
    def update_spectrogram(self):

        '''
        Updates the spectrogram. The oldest line of the spectrogram(window) is updated with a new one by using create_spectrogram and data.
        
        '''

        data = self.sdr.rx()
        self.window = np.vstack([self.window[1:], (self.create_segment(data))])
        self.data_ready.emit(self.window)



    def create_spectrogram(self):

        '''
        Creates a spectrogram for the waterfall plot to start from. 
         
        Returns:
        numpy.ndarray
            A full spectrogram.
        
        '''

        spectrogram = np.zeros((self.frames, self.num_samples))

        for i in range(self.frames):
            spectrogram[i] = self.create_segment(self.sdr.rx())
    
        return spectrogram

    def start(self):
        while True:
            self.update_spectrogram()


class RealTimePlotter(QtWidgets.QMainWindow):
    
    def __init__(self):

        '''
        Initiliazes the RealTimePlotter
        
        '''

        super().__init__()
        
        self.initUI()

    def initUI(self):

        '''
        Sets up the user interface for the waterfall plot.
        
        '''

        self.setWindowTitle('Waterfall Plot')
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        # Creates the initial waterfall plot
        spectrogram = liveSpectrogram.window
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
        freq_scale = liveSpectrogram.bandwidth / liveSpectrogram.num_samples  # Frequency scale in Hz per pixel
        time_scale = liveSpectrogram.frames / liveSpectrogram.sample_rate  # Time scale in seconds per pixel
        tr.scale(freq_scale, -time_scale)
        self.waterfall.setTransform(tr)
        self.waterfall.setPos(liveSpectrogram.rx_lo - (liveSpectrogram.bandwidth / 2),0.0012)  # Position the image correctly
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

        # Optionally, you can set the levels of the ImageItem for better color mapping
        #self.waterfall.setLevels([np.min(data), np.max(data)])


        
    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, spectrogram):
        '''
        Updates the waterfall plot. 

        Parameters:
        spectrogram : numpy.ndarray
            spectrogram contains segments. The oldest segment in the spectrogram has been updated with a new one. 
        
        '''
        self.waterfall.setImage(spectrogram.T)


if __name__ == '__main__':

    # Creates the parameters for the PlutoSDR and LiveSpectrogram
    samp_rate = 30e6
    num_samples = 65536
    rx_lo = 105e6           # Center Frequency
    rx_mode = "manual"
    rx_gain = 70
    bandwidth = int(30e6)

    # Creates the PlutoSDR and sets the properties
    sdr = adi.ad9361(uri='ip:192.168.2.1')
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = bandwidth
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(num_samples)
    
    # Number of FFTs/segments that the window will contain
    frames = 200

    liveSpectrogram = LiveSpectrogram( frames, num_samples, samp_rate, sdr, rx_lo, bandwidth)
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()

    # Puts and runs the LiveSpectrogram in another thread
    liveSpectrogram.data_ready.connect(plotter.update_plot)
    thread = QtCore.QThread()
    liveSpectrogram.moveToThread(thread)
    thread.started.connect(liveSpectrogram.start)
    thread.start()

    plotter.show()
    sys.exit(app.exec_())
