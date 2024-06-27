import numpy as np
import sys
import SoapySDR as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import pyargus.directionEstimation as pa
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from SoapySDR import *
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor

class KrakenReceiver():
    def __init__(self, center_freq, num_samples, sample_rate, bandwidth, gain, antenna_distance, x, y, num_devices=5):
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64) #signals([self.center_freq], [90] ,self.num_devices, self.num_samples) #
        self.devices, self.streams = self._setup_devices()
        self.x = x * antenna_distance
        self.y = y

        #Build digital filter
        # fc = self.center_freq
        # fs = 4*fc
        # fn = 0.2*fs
        # bandwidth = 0.2*fc
        # wn = [np.finfo(float).eps, (bandwidth/2) / fn] 
        # sos = butter(4, wn, btype='bandpass', output='sos')
        # self.filter = sos


        numtaps = 101  # Number of filter taps (filter length)
        fc = self.center_freq
        fs = 4*fc
        bandwidth = 0.4*fc
        highcut = bandwidth/2  # Upper cutoff frequency (Hz)

        # Design a band-pass FIR filter using the firwin function
        taps = signal.firwin(numtaps, [highcut], fs=fs, pass_zero=True, window='hamming')
        self.filter = taps

        # Compute the frequency response of the filter
        #freq_response = np.abs(np.fft.fft(taps, 1000))  # Compute FFT and take magnitude

        #freqz = np.fft.fftfreq(freq_response.size, 1/fs)

    def _setup_device(self, device_args):
        device = sp.Device(device_args) 
        device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        device.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        device.setGain(SOAPY_SDR_RX, 0, self.gain)
        device.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        return device
    
    
    def _setup_devices(self):
        devices = np.zeros(self.num_devices, dtype=object)
        streams = np.zeros(self.num_devices, dtype=object)
        available_devices = sp.Device.enumerate()
        
        available_devices = (available_devices[1:] + available_devices[:1])[:self.num_devices]

        for i, device_args in enumerate(available_devices):
            device = self._setup_device(device_args)
            devices[i] = device
            rx_stream = device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
            device.activateStream(rx_stream) #,, numElems=self.num_samples flags=SOAPY_SDR_END_BURST,
            streams[i] = rx_stream

        return devices, streams

    def close_streams(self):
        for i, device in enumerate(self.devices):
            device.deactivateStream(self.streams[i])
            device.closeStream(self.streams[i])
            del device

    def _read_stream(self, device, timestamp):
        sr = self.devices[device].readStream(self.streams[device], [self.buffer[device]], 
                                            self.num_samples, 0,timestamp)
        
        ret = sr.ret

        if ret < 0:
        #Handle errors based on the error codes
            if ret == SOAPY_SDR_TIMEOUT:
                raise ValueError("Timeout when reading stream")
            elif ret == SOAPY_SDR_STREAM_ERROR:
                raise ValueError("Stream error when reading stream")
            elif ret == SOAPY_SDR_CORRUPTION:
                raise ValueError("Data corruption when reading stream")
            elif ret == SOAPY_SDR_OVERFLOW:
                raise ValueError("Overflow when reading stream")
            elif ret == SOAPY_SDR_NOT_SUPPORTED:
                raise ValueError("Requested operation or flag setting is not supported")
            elif ret == SOAPY_SDR_TIME_ERROR:
                raise ValueError("Encountered a stream time which was expired or too early to process")
            elif ret == SOAPY_SDR_UNDERFLOW:
                raise ValueError("write caused an underflow condition")
            
        #print(f"Device {device}: \n Samples = {sr.ret} \n Flag = {sr.flags}\n")

    def read_streams(self):
        current_time_ns = int(time.time() * 1e9)
        start_time_ns = int(current_time_ns + 5e9)
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(self._read_stream, i, start_time_ns) for i in range(self.num_devices)]
            for future in futures:
                future.result()
    
        self.buffer = signal.lfilter(self.filter, 1.0, self.buffer)

    def plot_fft(self):

        if self.num_devices > 1:
            fig, axes = plt.subplots(self.num_devices, 2, figsize=(18, 8 * self.num_devices))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            axes = np.array([axes])  #Ensure axes is always a 2D array

        for i in range(self.num_devices):
            #FFT of the samples
            freqs = np.fft.fftfreq(self.num_samples, d=1/self.sample_rate)
            fft_samples = fft(self.buffer[i])
            axes[i, 0].plot(freqs, np.abs(fft_samples))
            axes[i, 0].grid()
            axes[i, 0].set_title(f'FFT of Received Samples (Channel {i})')
            axes[i, 0].set_xlabel('Frequency (Hz)')
            axes[i, 0].set_ylabel('Amplitude')

            #Phase of the samples
            fft_samples[np.abs(fft_samples) < 2000] = 0
            phases = np.angle(fft_samples, deg=True)
            phases[np.angle(fft_samples) == 0] = np.NaN
            axes[i, 1].plot(freqs, phases, 'o')
            axes[i, 1].grid()
            axes[i, 1].set_xlim(-self.sample_rate / 2, self.sample_rate / 2)
            axes[i, 1].set_ylim(-180,180)
            axes[i, 1].set_title(f'Phase of Received Samples (Channel {i})')
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Phase (radians)')

        plt.tight_layout()
        plt.show()

    def music(self):
        #smoothed_buffer = pa.spatial_smoothing(self.buffer, 2, direction = 'forward-backward')
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        spatial_corr_matrix = pa.forward_backward_avg(spatial_corr_matrix)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(0,180))
        doa = pa.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, signal_dimension=1)


        return doa

    def mem(self):
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(0,180))
        doa = pa.DOA_MEM(spatial_corr_matrix,scanning_vectors)

        return doa

    def capon(self):
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(0,180))
        doa = pa.DOA_Capon(spatial_corr_matrix,scanning_vectors)

        return doa

    def bartlett(self):
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(0,180))
        doa = pa.DOA_Bartlett(spatial_corr_matrix,scanning_vectors)

        return doa

    def lpm(self, element_select):
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
        scanning_vectors = pa.gen_scanning_vectors(self.num_devices, self.x, self.y, np.arange(0,180))
        doa = pa.DOA_LPM(spatial_corr_matrix,scanning_vectors, element_select)

        return doa

def get_device_info():
        available_devices = sp.Device.enumerate()
        device_info_list = []

        for i, device_args in enumerate(available_devices):
            device_info = {}
            print(device_args)
            device = sp.Device(device_args)
            device_info['sample_rate_range'] = device.getSampleRateRange(SOAPY_SDR_RX, 0)
            device_info['frequency_range'] = device.getFrequencyRange(SOAPY_SDR_RX, 0)
            device_info['gain_range'] = device.getGainRange(SOAPY_SDR_RX, 0)
            device_info['bandwidth_range'] = device.getBandwidthRange(SOAPY_SDR_RX, 0)

            print(f"Device {i}:")
            for key, value in device_info.items():
                print(f"  {key}: {value}")
                
            device_info_list.append(device_info)

        return device_info_list

def signals(frequencies, angles, num_sensors, num_snapshots, wavelength=1.0, noise_power=1e-3):

    antenna_distance_m = 0.35
    sensor_positions = np.array([0,1,2]) * antenna_distance_m
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    frequency_offset = frequencies[0]


    for f, angle in zip(frequencies, angles):
        f_cal = f - frequency_offset
        signal = np.exp(1j * 2 * np.pi * f_cal * np.arange(num_snapshots) / num_snapshots)
        steering_vector = np.exp(1j * 2 * np.pi * sensor_positions[:, np.newaxis] * np.sin(np.radians(angle)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + 1000 * noise
    
class RealTimePlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(0)
        
    def initUI(self):
        self.setWindowTitle('Real-Time Data Visualization')
        
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QGridLayout(self.centralWidget)
        
        self.doa_plot = pg.PlotWidget(title="Direction of Arrival")
        self.doa_curve = self.doa_plot.plot(pen='y')
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
        
    def update_plots(self):
        kraken.read_streams()
        doa_data = kraken.music()
        doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        print(np.argmax(doa_data))
        
        num_samples = len(kraken.buffer[0])
        sample_rate = 1000  # Assuming a sample rate of 1000 Hz
        freqs = np.fft.fftfreq(num_samples, d=1/sample_rate)
        
        ant0 = np.abs(np.fft.fft(kraken.buffer[0]))
        ant1 = np.abs(np.fft.fft(kraken.buffer[1]))
        ant2 = np.abs(np.fft.fft(kraken.buffer[2]))
        
        self.doa_curve.setData(np.linspace(0, 179, 180), doa_data)
        self.fft_curve_0.setData(freqs, ant0)
        self.fft_curve_1.setData(freqs, ant1)
        self.fft_curve_2.setData(freqs, ant2)

if __name__ == '__main__':
    num_samples = 1024*256
    sample_rate = 2.048e6
    center_freq = 103.3e6
    bandwidth =  2e5 
    gain = 40
    y = np.array([0,0,0])
    x = np.array([0,1,2])
    antenna_distance = 0.725

    kraken = KrakenReceiver(center_freq, num_samples, 
                           sample_rate, bandwidth, gain, antenna_distance, x, y, num_devices=3)
    kraken.read_streams()
    kraken.plot_fft()
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    sys.exit(app.exec_())

# # Apply the filter to the signal
# filtered_x = signal.lfilter(taps, 1.0, x)

# # Plot the frequency response
# plt.figure(figsize=(10, 6))
# plt.plot(freqz, 20 * np.log10(freq_response))
# plt.title('Frequency Response of Band-Pass FIR Filter')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain (dB)')
# plt.grid(True)
# plt.ylim(-80, 10)
# plt.xlim(0, fs / 2)
# plt.tight_layout()
# plt.show()
