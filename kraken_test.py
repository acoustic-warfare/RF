import numpy as np
import SoapySDR as sp
from SoapySDR import *
from scipy.fft import fft
#from scipy.signal import signal 
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

class KrakenReceiver():
    def __init__(self, center_freq, num_samples, sample_rate, bandwidth, gain, num_devices=5):
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)
        self.devices, self.streams = self._setup_devices()

    def _setup_device(self,i):
        device = sp.Device(dict(driver="rtlsdr")) 
        device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        device.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        device.setGain(SOAPY_SDR_RX, 0, self.gain)
        device.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        return device
    
    def _setup_devices(self):
        devices = np.zeros(self.num_devices, dtype=object)
        streams = np.zeros(self.num_devices, dtype=object)
        for i in range(self.num_devices):
            device = self._setup_device(i)
            devices[i] = device
            rx_stream = device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
            device.activateStream(rx_stream)
            streams[i] = rx_stream
        return devices, streams

    def close_streams(self):
        for i, device in enumerate(self.devices):
            device.deactivateStream(self.streams[i])
            device.closeStream(self.streams[i])

    def _read_stream(self, device, time):
        sr = self.devices[device].readStream(
            self.streams[device], [self.buffer[device]], self.num_samples, 0, time)
        
        #print(f"Device {device}: \n Samples = {sr.ret} \n Timestamp = {sr.timeNs}\n Flag = {sr.flags}\n")

    def read_streams(self):
        current_time_ns = int(time.time() * 1e9)
        start_time_ns = int(current_time_ns + 5e9)
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(self._read_stream, i, start_time_ns) for i in range(self.num_devices)]
            for future in futures:
                future.result()


    def calc_direction_test(self):
        self.read_streams()
        ch0_samples = self.buffer[0]
        ch1_samples = self.buffer[1]
        ch2_samples = self.buffer[2] 
       
        fft_ch0_samples = fft(ch0_samples)
        fft_ch1_samples = fft(ch1_samples)
        fft_ch2_samples = fft(ch2_samples)

        freq = np.fft.fftfreq(self.num_samples, d=1/sample_rate)

        abs_fft_ch0_samples = np.abs(fft_ch0_samples)
        #abs_fft_ch1_samples = np.abs(fft_ch1_samples)
        #abs_fft_ch2_samples = np.abs(fft_ch2_samples)
        
        max_index = np.argmax(abs_fft_ch0_samples)
        # Set all elements to zero except for the one at max index

        phase0 = np.angle(fft_ch0_samples[max_index])
        phase1 = np.angle(fft_ch1_samples[max_index])
        phase2 = np.angle(fft_ch2_samples[max_index])

        p21 = phase2 - phase1
        p20 = phase2 - phase0
            
        doa = np.arctan(p21/(2*p20-p21))

        print(freq[max_index])

        print(f"p21 = {p21}")
        print(f"p20 = {p20}")
        print(doa)
        print("\n")

    def plot_fft_test(self):
        freqs = np.fft.fftfreq(self.num_samples, d=1/self.sample_rate)

        ch0_samples = self.buffer[0]
        ch1_samples = self.buffer[1]

        fft_ch0_samples = fft(ch0_samples)
        fft_ch1_samples = fft(ch1_samples)
        abs_fft_ch0_samples = np.abs(fft_ch0_samples)
        abs_fft_ch1_samples = np.abs(fft_ch1_samples)

        fft_ch0_samples[abs_fft_ch0_samples < 20] = 0
        fft_ch1_samples[abs_fft_ch1_samples < 20] = 0
        ch0_angle = np.angle(fft_ch0_samples, deg=True)
        ch1_angle = np.angle(fft_ch1_samples, deg=True)

        max_index = np.argmax(abs_fft_ch0_samples)
        print(f"amplitude = {abs_fft_ch0_samples[max_index]}")

        fig, ((ax, ax1), (ax_phase, ax1_phase)) = plt.subplots(2, 2, figsize=(18, 8))
        #((ax, ax1), (ax_phase, ax1_phase)) = plt.subplots(1, 2, figsize=(15,6))

        ax.plot(freqs, abs_fft_ch0_samples) 
        ax.grid()
        ax.set_title("Real-Time FFT of Received Samples (Channel 0)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")

        ax1.plot(freqs, abs_fft_ch1_samples)
        ax1.grid()
        ax1.set_title("Real-Time FFT of Received Samples (Channel 1)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")


        ax_phase.plot(freqs, ch0_angle, 'o') 
        ax_phase.grid()
        ax_phase.set_title("Real-Time phase of Received Samples (Channel 0)")
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("phase (radians)")

        ax1_phase.plot(freqs, ch1_angle, 'o')
        ax1_phase.grid()
        ax1_phase.set_title("Real-Time phase of Received Samples (Channel 1)")
        ax1_phase.set_xlabel("Frequency (Hz)")
        ax1_phase.set_ylabel("Phase (radians)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    num_samples = 256*1024
    sample_rate = 1e6
    center_freq = 433.9e6
    bandwidth = 200e3
    gain = 50
    kraken = KrakenReceiver(center_freq, num_samples, 
                            sample_rate, bandwidth, gain, num_devices=3)
    # while True:
    #     kraken.calc_direction_test()
    kraken.read_streams()
    kraken.close_streams()
    kraken.plot_fft_test()