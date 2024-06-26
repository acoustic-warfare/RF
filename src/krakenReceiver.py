import numpy as np
import SoapySDR as sp
from SoapySDR import *
from scipy.fft import fft
from scipy.signal import butter, sosfilt
#from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
import pyargus.directionEstimation as pa
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
        fc = self.center_freq
        fs = 4*fc
        fn = 0.2*fs
        bandwidth = 0.6*fc
        wn = [np.finfo(float).eps, (bandwidth/2) / fn] 
        sos = butter(4, wn, btype='bandpass', output='sos')
        self.filter = sos

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
    
        self.buffer = sosfilt(self.filter, self.buffer)

    def plot_fft(self):

        if self.num_devices > 1:
            fig, axes = plt.subplots(self.num_devices, 2, figsize=(15, 5 * self.num_devices))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
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
        spatial_corr_matrix = np.dot(self.buffer, self.buffer.conj().T)
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

# def init():
#     for lines in [line, line1, line2, line_doa]:
#         lines.set_data([], [])

#     for ax in axs.flat[:3]:
#         ax.set_xlim(-sample_rate / 2, sample_rate / 2)

#     axs[1,1].set_xlim(0,179)
#     return line, line1, line2, line_doa

# def update(frame):
#     kraken.read_streams()

#     doa_data = kraken.music()
#     doa_data = np.divide(np.abs(doa_data),np.max(np.abs(doa_data)))
#     line_doa.set_data(np.linspace(0,179,180), doa_data)
#     axs[1,1].set_ylim(np.min(doa_data),1.1)

#     freqs = np.fft.fftfreq(num_samples, d=1 / sample_rate)
#     line_list = [line, line1, line2]
#     for i, lines in enumerate(line_list):
#         samples = kraken.buffer[i]
#         fft_samples = fft(samples)
#         abs_fft_samples = np.abs(fft_samples)

#         lines.set_data(freqs, abs_fft_samples)
#         axs.flat[i].set_ylim(0, 1.1*np.max(abs_fft_samples))

#     return line, line1, line2, line_doa

def plot_doa(doa_data):
    plt.ion()
    plt.plot(np.linspace(0,179,180), doa_data)
    plt.pause(0.0001)
    plt.draw()
    plt.clf()

# class RealTimePlot(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setGeometry(100, 100, 800, 600)

#         self.series = QLineSeries()
#         self.chart = QChart()
#         self.chart.addSeries(self.series)
#         self.chart.createDefaultAxes()
        
#         self.chart_view = QChartView(self.chart)
#         self.chart_view.setRenderHint(QPainter.Antialiasing)

#         layout = QVBoxLayout()
#         layout.addWidget(self.chart_view)
        
#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)
        
#         self.x = np.linspace(0, 179, 180)
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start()

#     def update_plot(self):
#         kraken.read_streams()
#         doa_data = kraken.music()
#         doa_data = np.divide(np.abs(doa_data),np.max(np.abs(doa_data)))
#         self.series.clear()
#         for i in range(len(self.x)):
#             self.series.append(self.x[i], doa_data[i])

# if __name__ == "__main__":
#     num_samples = 1024*256
#     sample_rate = 2.048e6
#     center_freq = 433.9e6
#     bandwidth =  2e5 
#     gain = 40
#     y = np.array([0,0,0])
#     x = np.array([0,1,2])
#     antenna_distance = 0.35

#     kraken = KrakenReceiver(center_freq, num_samples, 
#                            sample_rate, bandwidth, gain, antenna_distance, x, y, num_devices=3)
#     app = QApplication(sys.argv)
#     window = RealTimePlot()
#     window.show()
#     sys.exit(app.exec_())


if __name__ == "__main__":
    num_samples = 1024*256
    sample_rate = 2.048e6
    center_freq = 433.9e6
    bandwidth =  2e5 
    gain = 40
    y = np.array([0,0])
    x = np.array([0,1])
    antenna_distance = 0.35

    kraken = KrakenReceiver(center_freq, num_samples, 
                           sample_rate, bandwidth, gain, antenna_distance, x, y, num_devices=2)

   
    while True:
        kraken.read_streams()
        doa_data = kraken.music()
        doa_data = np.divide(np.abs(doa_data),np.max(np.abs(doa_data)))
        print(np.argmax(doa_data))
        plt.ion()
        plt.plot(np.linspace(0,179,180), doa_data)
        #plt.pause(np.finfo(float).eps)
        plt.draw()
        plt.clf()
    

    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # (line,) = axs[0, 0].plot([], [])  #FFT for channel 0
    # axs[0, 0].grid()
    # axs[0, 0].set_title("Real-Time FFT of Received Samples (Channel 0)")
    # axs[0, 0].set_xlabel("Frequency (Hz)")
    # axs[0, 0].set_ylabel("Amplitude")

    # # Top-right subplot (0, 1)
    # (line1,) = axs[0, 1].plot([], [])  #FFT for channel 1
    # axs[0, 1].grid()
    # axs[0, 1].set_title("Real-Time FFT of Received Samples (Channel 1)")
    # axs[0, 1].set_xlabel("Frequency (Hz)")
    # axs[0, 1].set_ylabel("Amplitude")

    # # Bottom-left subplot (1, 0)
    # (line2,) = axs[1, 0].plot([], [])  #FFT for channel 2
    # axs[1, 0].grid()
    # axs[1, 0].set_title("Real-Time FFT of Received Samples (Channel 2)")
    # axs[1, 0].set_xlabel("Frequency (Hz)")
    # axs[1, 0].set_ylabel("Amplitude")

    # # Bottom-right subplot (1, 1)
    # (line_doa,) = axs[1, 1].plot([], [])  #Direction of Arrival Estimation
    # axs[1, 1].grid()
    # axs[1, 1].set_title('Direction of Arrival Estimation')
    # axs[1, 1].set_xlabel('Incident Angle [deg]')
    # axs[1, 1].set_ylabel('Amplitude')
    

    # ani = FuncAnimation(fig, update, init_func=init, frames=100, interval=500, blit=False)

    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
  