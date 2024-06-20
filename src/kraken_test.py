import sys

sys.path.append("/usr/local/lib")

import numpy as np
import SoapySDR as sp
from SoapySDR import *
from scipy.fft import fft

# from scipy.signal import signal
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor


class KrakenReceiver:
    def __init__(
        self, center_freq, num_samples, sample_rate, bandwidth, gain, num_devices=5
    ):
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)
        self.devices, self.streams = self._setup_devices()

    def _setup_device(self, i):
        device = sp.Device(dict(driver="rtlsdr"))
        device.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        device.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        device.setGain(SOAPY_SDR_RX, 0, self.gain)
        device.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        print(f"Num antennas: {device.getNumChannels(SOAPY_SDR_RX)}")
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

        print()
        return devices, streams

    def close_streams(self):
        for i, device in enumerate(self.devices):
            device.deactivateStream(self.streams[i])
            device.closeStream(self.streams[i])

    def _read_stream(self, device, time):
        sr = self.devices[device].readStream(
            self.streams[device], [self.buffer[device]], self.num_samples, 0, time
        )

        print(
            f"Device {device}: \n Samples = {sr.ret} \n Timestamp = {sr.timeNs}\n Flag = {sr.flags}\n"
        )

    def read_streams(self):
        current_time_ns = int(time.time() * 1e9)
        start_time_ns = int(current_time_ns + 5e9)
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [
                executor.submit(self._read_stream, i, start_time_ns)
                for i in range(self.num_devices)
            ]
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

        freq = np.fft.fftfreq(self.num_samples, d=1 / sample_rate)

        abs_fft_ch0_samples = np.abs(fft_ch0_samples)
        # abs_fft_ch1_samples = np.abs(fft_ch1_samples)
        # abs_fft_ch2_samples = np.abs(fft_ch2_samples)

        max_index = np.argmax(abs_fft_ch0_samples)
        # Set all elements to zero except for the one at max index

        phase0 = np.angle(fft_ch0_samples[max_index])
        phase1 = np.angle(fft_ch1_samples[max_index])
        phase2 = np.angle(fft_ch2_samples[max_index])

        p21 = phase2 - phase1
        p20 = phase2 - phase0

        doa = np.arctan(p21 / (2 * p20 - p21))

        print(freq[max_index])

        print(f"p21 = {p21}")
        print(f"p20 = {p20}")
        print(doa)
        print("\n")

    def plot_fft(self):

        if self.num_devices > 1:
            fig, axes = plt.subplots(
                self.num_devices, 2, figsize=(15, 5 * self.num_devices)
            )
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = np.array([axes])  # Ensure axes is always a 2D array

        for i in range(self.num_devices):
            # FFT of the samples
            freqs = np.fft.fftfreq(self.num_samples, d=1 / self.sample_rate)
            # fft_samples = np.fft.fftshift(fft(self.buffer[i]))
            fft_samples = fft(self.buffer[i])
            axes[i, 0].plot(freqs, np.abs(fft_samples))
            axes[i, 0].grid()
            axes[i, 0].set_title(f"FFT of Received Samples (Channel {i})")
            axes[i, 0].set_xlabel("Frequency (Hz)")
            axes[i, 0].set_ylabel("Amplitude")

            # Phase of the samples
            fft_samples[np.abs(fft_samples) < 30] = 0
            phases = np.angle(fft_samples, deg=True)
            phases[np.angle(fft_samples) == 0] = np.NaN
            axes[i, 1].plot(freqs, phases, "o")
            axes[i, 1].grid()
            axes[i, 1].set_xlim(-self.sample_rate / 2, self.sample_rate / 2)
            axes[i, 1].set_ylim(-180, 180)
            axes[i, 1].set_title(f"Phase of Received Samples (Channel {i})")
            axes[i, 1].set_xlabel("Frequency (Hz)")
            axes[i, 1].set_ylabel("Phase (radians)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    num_samples = 1000000  # 256*1024
    sample_rate = 2.048e6
    center_freq = 102.7e6
    bandwidth = 200e3
    gain = 50
    kraken = KrakenReceiver(
        center_freq, num_samples, sample_rate, bandwidth, gain, num_devices=2
    )
    # while True:
    #     kraken.calc_direction_test()
    kraken.read_streams()
    kraken.close_streams()
    kraken.plot_fft()
