import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import adi
import numpy as np
from pyqtgraph.Qt import QtGui

class LiveSpectrogram(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self , frames, num_samples, sample_rate, sdr, rx_lo):
        super().__init__()
        self.sdr = sdr
        self.frames = segment_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.increment = 0
        self.num_samples = num_samples    
        self.rx_lo = rx_lo
        self.segment_array = np.zeros((int(frames), num_samples))
        #self.change_freq=0

        self.data = self.sdr.rx()

    def create_segment(self, sample):

        segment = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(sample))**2))
        return segment
    
    def updating_spectrogram(self):
        # if self.change_freq == 200:
        #     self.sdr.rx_lo = int(433e6)
        self.data = self.sdr.rx()

        self.segment_array = np.vstack([self.segment_array[1:], (self.create_segment(self.data))])
        #self.change_freq = self.change_freq +1
        self.data_ready.emit(self.segment_array)



    def creating_spectrogram(self, samples):
        segments = []

        for i in range(self.num_segments):
            segments.append(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*self.overlap:i*self.overlap + self.frames]))**2))) #from pySDR
    
        return segments

    def start(self):
        while True:
            self.updating_spectrogram()


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
        freq_scale= (30e6)/260
        time_scale= (-num_samples/samp_rate)/260
        tr.scale(freq_scale, time_scale)
        self.waterfall.setTransform(tr) # assign transform
        self.waterfall.setPos(rx_lo-(bandwidth/2), num_samples/samp_rate)
        self.p1.invertY()

    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, spec):
        self.waterfall.setImage(spec.T)


if __name__ == '__main__':
    samp_rate = 30e6
    num_samples = 65536
    rx_lo = 105e6
    rx_mode = "manual"
    rx_gain0 = 70
    bandwidth = int(30e6)

    sdr = adi.ad9361(uri='ip:192.168.2.1')
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = bandwidth
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_buffer_size = int(num_samples)

    segment_size = 512
    overlap = 256
    window_function = np.hanning(segment_size)
    frames = 60

    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    liveSpectrogram = LiveSpectrogram( frames, num_samples, samp_rate, sdr, rx_lo)
    liveSpectrogram.data_ready.connect(plotter.update_plot)

    thread = QtCore.QThread()
    liveSpectrogram.moveToThread(thread)
    thread.started.connect(liveSpectrogram.start)
    thread.start()

    plotter.show()
    sys.exit(app.exec_())
