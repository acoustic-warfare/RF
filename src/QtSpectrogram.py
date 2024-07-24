# import sys
# import pyqtgraph as pg
# from PyQt5 import QtWidgets
# from pyqtgraph.Qt import QtCore
# import pyqtgraph as pg
# import numpy as np
# import adi


# class LiveSpectrogram():
    
#     def __init__(self, segment_size, overlap, window_function, frames, num_samples, sample_rate, sdr, rx_lo):
#         self.window_function = window_function
#         self.sdr = sdr
#         self.segment_size = segment_size
#         self.overlap = overlap
#         self.frames = frames
#         self.sample_rate = sample_rate
#         self.increment = 0
#         self.num_samples = num_samples
#         self.step_size = self.segment_size - self.overlap
#         self.num_segments = (num_samples- self.overlap) // self.step_size
#         self.big_spectrogram = []
#         self.spectrogram = np.zeros((frames, segment_size // 2 + 1))
#         self.rx_lo = rx_lo

#     def creating_spectrogram(self, samples):
#         segments = []
#         # print("num of samples ", self.num_samples)
#         # print("Num of segments: ", self.num_segments)
#         for i in range(self.num_segments):
#             segments.append(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*self.overlap:i*self.overlap + self.segment_size]))**2))) #from pySDR
#             #segments[i] = 10*np.log10(np.abs((np.fft.rfft(samples[i*overlap:i*overlap + self.segment_size])))**2) #one liner.
#         #print("Segements: ", segments)
#         return segments
 
    
#     def update_big_spectrogram(self):
#         samples = self.sdr.rx()
#         self.big_spectrogram.extend(self.creating_spectrogram(samples))
#         if (len(self.big_spectrogram) > segment_size):
#             self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])
#         else:
#             self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])


        
#         self.increment = self.increment+1
#         return self.spectrogram





# class RealTimePlotter(QtWidgets.QMainWindow):
#     def __init__(self):
#         """
#         Initializes the RealTimePlotter instance.
#         """
#         super().__init__()
        
#         self.initUI()
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start(100)

#     def initUI(self):
#         """
#         Sets up the user interface (UI) layout.
#         """
#         self.setWindowTitle('Real-Time Data Visualization')
        
#         self.centralWidget = QtWidgets.QWidget()
#         self.setCentralWidget(self.centralWidget)
        
#         spec = np.zeros((frames, segment_size // 2 + 1))
#         self.waterfall = pg.ImageItem(spec)
#         self.layout = QtWidgets.QGridLayout(self.centralWidget)
#         self.win = pg.GraphicsLayoutWidget()
#         self.p1 = self.win.addPlot()
#         self.p1.addItem(self.waterfall)
#         hist = pg.HistogramLUTItem()
#         # Link the histogram to the image
#         hist.setImageItem(self.waterfall)
#         # If you don't add the histogram to the window, it stays invisible, but I find it useful.
#         self.win.addItem(hist)
#         hist.setLevels(30, 70)
#         self.waterfall.setImage(spec)


#     def update_plot(self):
#         spec = liveSpectrogram.update_big_spectrogram()
#         self.waterfall.setImage(spec)
        

# if __name__ == '__main__':
#     #Creating Pluto
#     samp_rate = 30e6    # must be <=30.72 MHz if both channels are enabled
#     num_samples = 32768
#     rx_lo = 105e6
#     rx_mode = "manual"  # can be "manual" or "slow_attack"
#     rx_gain0 = 40
#     fft_size = 256 
#     bandwidth = int(30e6)

#     '''Create Radio'''
#     sdr = adi.ad9361(uri='ip:192.168.2.1')

#     '''Configure properties for the Radio'''
#     sdr.rx_enabled_channels = [0]
#     sdr.sample_rate = int(samp_rate)
#     sdr.rx_rf_bandwidth = bandwidth
#     sdr.rx_lo = int(rx_lo)
#     sdr.gain_control_mode = rx_mode
#     sdr.rx_hardwaregain_chan0 = int(rx_gain0)

#     sdr.rx_buffer_size = int(num_samples)


#     segment_size = 256
#     overlap = 128
#     window_function = np.hanning(segment_size)
#     frames = 32
#     # sample_rate = 1e6
#     # t = np.arange(1024*1000)/sample_rate # time vector
#     # f = 50e3 # freq of tone
#     # samples = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))
    

#     liveSpectrogram= LiveSpectrogram(segment_size, overlap, window_function, frames, num_samples, samp_rate, sdr, rx_lo)
#     app = QtWidgets.QApplication(sys.argv)
#     plotter=RealTimePlotter()
#     plotter.show()
#     sys.exit(app.exec_())
import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import adi
import numpy as np
from pyqtgraph.Qt import QtGui

class LiveSpectrogram(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, segment_size, overlap, window_function, frames, num_samples, sample_rate, sdr, rx_lo):
        super().__init__()
        self.window_function = window_function
        self.sdr = sdr
        self.segment_size = segment_size
        self.overlap = overlap
        self.frames = frames
        self.sample_rate = sample_rate
        self.increment = 0
        self.num_samples = num_samples
        self.step_size = self.segment_size - self.overlap
        self.num_segments = (num_samples - self.overlap) // self.step_size
        self.big_spectrogram = []
        self.spectrogram = np.zeros((frames, segment_size // 2 + 1))
        self.rx_lo = rx_lo

    def creating_spectrogram(self, samples):
        segments = []
        # print("num of samples ", self.num_samples)
        # print("Num of segments: ", self.num_segments)
        for i in range(self.num_segments):
            segments.append(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*self.overlap:i*self.overlap + self.segment_size]))**2))) #from pySDR
            #segments[i] = 10*np.log10(np.abs((np.fft.rfft(samples[i*overlap:i*overlap + self.segment_size])))**2) #one liner.
        #print("Segements: ", segments)
        return segments
 
    
    def update_big_spectrogram(self):
        samples = self.sdr.rx()
        self.big_spectrogram.extend(self.creating_spectrogram(samples))
        if (len(self.big_spectrogram) > segment_size):
            self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])
        else:
            self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])


        
        self.increment = self.increment+1
        self.data_ready.emit(self.spectrogram)

    def start(self):
        while True:
            self.update_big_spectrogram()


class RealTimePlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Real-Time Data Visualization')
        
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        spec = np.zeros((frames, segment_size // 2 + 1))
        self.waterfall = pg.ImageItem(spec)
        self.layout = QtWidgets.QGridLayout(self.centralWidget)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)
        self.p1 = self.win.addPlot()

        # Add labels to the axis
        self.p1.setLabel('left', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
        self.p1.setLabel('bottom', "Frequency", units='Hz')
        self.p1.addItem(self.waterfall)
        self.waterfall.setImage(spec.T)
        self.waterfall.setColorMap(colorMap='viridis')
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        xscale = (30/260)*1e6
        tr.scale(xscale, 8)
        self.waterfall.setTransform(tr) # assign transform

    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, spec):
        self.waterfall.setImage(spec.T)


if __name__ == '__main__':
    samp_rate = 30e6
    num_samples = 32736
    rx_lo = 105e6
    rx_mode = "manual"
    rx_gain0 = 40
    bandwidth = int(30e6)

    sdr = adi.ad9361(uri='ip:192.168.2.1')
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = bandwidth
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_buffer_size = int(num_samples)

    segment_size = 256
    overlap = 128
    window_function = np.hanning(segment_size)
    frames = 60

    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    liveSpectrogram = LiveSpectrogram(segment_size, overlap, window_function, frames, num_samples, samp_rate, sdr, rx_lo)
    liveSpectrogram.data_ready.connect(plotter.update_plot)

    thread = QtCore.QThread()
    liveSpectrogram.moveToThread(thread)
    thread.started.connect(liveSpectrogram.start)
    thread.start()

    plotter.show()
    sys.exit(app.exec_())
