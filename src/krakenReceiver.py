import numpy as np
import SoapySDR as sp
from SoapySDR import *
from scipy.fft import fft
#from scipy.signal import signal 
import matplotlib.pyplot as plt
import time
import pyargus.directionEstimation as pa

from concurrent.futures import ThreadPoolExecutor

#signals([433e6], [30] ,self.num_devices, self.num_samples)#

class KrakenReceiver():
    def __init__(self, center_freq, num_samples, sample_rate, bandwidth, gain, num_devices=5):
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.buffer = signals([self.center_freq], [90,45] ,self.num_devices, self.num_samples)#np.zeros((self.num_devices, num_samples), dtype=np.complex64)
        self.devices, self.streams = self._setup_devices()

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
            device.activateStream(rx_stream)
            streams[i] = rx_stream

        return devices, streams

    # def close_streams(self):
    #     for i, device in enumerate(self.devices):
    #         device.deactivateStream(self.streams[i])
    #         device.closeStream(self.streams[i])
    #         del device

    # def _read_stream(self, device, time):
    #     sr = self.devices[device].readStream(
    #         self.streams[device], [self.buffer[device]], self.num_samples, 0, time)
        
    #     #print(f"Device {device}: \n Samples = {sr.ret} \n Timestamp = {sr.timeNs}\n Flag = {sr.flags}\n")

    # def read_streams(self):
    #     current_time_ns = int(time.time() * 1e9)
    #     start_time_ns = int(current_time_ns + 5e9)
    #     with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
    #         futures = [executor.submit(self._read_stream, i, start_time_ns) for i in range(self.num_devices)]
    #         for future in futures:
    #             future.result()

    def plot_fft(self):

        if self.num_devices > 1:
            fig, axes = plt.subplots(self.num_devices, 2, figsize=(15, 5 * self.num_devices))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = np.array([axes])  #Ensure axes is always a 2D array

        for i in range(self.num_devices):
            #FFT of the samples
            freqs = np.fft.fftfreq(self.num_samples, d=1/self.sample_rate)
            #fft_samples = np.fft.fftshift(fft(self.buffer[i]))
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

    def _create_scanning_vectors(self, x, y, theta_coverage, offset=0):
        
        scanning_vectors = np.zeros((self.num_devices, theta_coverage), dtype=np.complex64)
        for theta in range(theta_coverage):
            theta_rad = np.deg2rad(theta)
            scanning_vectors[:, theta] = np.exp(
                1j * 2 * np.pi * (x * np.cos(theta_rad + offset) + y * np.sin(theta_rad + offset)))

        return scanning_vectors
    

    def music(self):

        spatial_corr_matrix = pa.corr_matrix_estimate(self.buffer.T)

        #wave_length = 0.6909
        antenna_distance_m = 0.25
        
        x = np.array([0,1,2]) * antenna_distance_m
        y = np.array([0,0,0])
        scanning_vectors = self._create_scanning_vectors(x, y, 360)

        # print("\n\n")

        # eigvals, _ = np.linalg.eig(spatial_corr_matrix)
        # print("Eigenvalues of the spatial correlation matrix:", eigvals)
        # cond_number = np.linalg.cond(spatial_corr_matrix)
        # print("Condition Number:", cond_number)

        # print("\n\n")

        # #Regularize the correlation matrix
        # regularization_parameter = 1e3
        # spatial_corr_matrix = spatial_corr_matrix + regularization_parameter * np.eye(spatial_corr_matrix.shape[0])

        # eigvals, _ = np.linalg.eig(spatial_corr_matrix)
        # print("Eigenvalues of the spatial correlation matrix:", eigvals)
        # cond_number = np.linalg.cond(spatial_corr_matrix)
        # print("Condition Number:", cond_number)

        # print("\n\n")

        doa = pa.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, self.num_devices)

        print(doa)

    def mem(self):
        spatial_corr_matrix = pa.corr_matrix_estimate(self.buffer.T)
        antenna_distance_m = 0.25
        x = np.array([0,1,2,3,4]) * antenna_distance_m
        y = np.array([0,0,0,0,0])
        scanning_vectors = self._create_scanning_vectors(x, y, 180)

        doa = pa.DOA_MEM(spatial_corr_matrix,scanning_vectors)
        return doa

    def capon(self):
        
        spatial_corr_matrix = pa.corr_matrix_estimate(self.buffer.T)
        antenna_distance_m = 0.25
        x = np.array([0,1,2,3,4]) * antenna_distance_m
        y = np.array([0,0,0,0,0])
        scanning_vectors = self._create_scanning_vectors(x, y, 180)

        doa = pa.DOA_Capon(spatial_corr_matrix,scanning_vectors)
        return doa

    def bartlett(self):
        spatial_corr_matrix = pa.corr_matrix_estimate(self.buffer.T)
        antenna_distance_m = 0.25
        x = np.array([0,1,2,3,4]) * antenna_distance_m
        y = np.array([0,0,0,0,0])
        scanning_vectors = self._create_scanning_vectors(x, y, 180)

        doa = pa.DOA_Capon(spatial_corr_matrix,scanning_vectors)
        return doa

    def lpm(self):
        spatial_corr_matrix = pa.corr_matrix_estimate(self.buffer.T)

        print(spatial_corr_matrix)

        antenna_distance_m = 0.25
        x = np.array([0,1,2,3,4]) * antenna_distance_m
        y = np.array([0,0,0,0,0])
        scanning_vectors = self._create_scanning_vectors(x, y, 180)

        doa = pa.DOA_LPM(spatial_corr_matrix,scanning_vectors, 2)
        return(doa)

    

def get_device_info():
        #Enumerate devices to get their arguments including serial numbers
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

    sensor_positions = np.arange(num_sensors)
    signals = np.zeros((num_sensors, num_snapshots), dtype=complex)
    
    for f, angle in zip(frequencies, angles):
        signal = np.exp(1j * 2 * np.pi * f * np.arange(num_snapshots) / num_snapshots)
        steering_vector = np.exp(1j * 2 * np.pi * sensor_positions[:, np.newaxis] * np.sin(np.radians(angle)) / wavelength)
        signals += steering_vector @ signal[np.newaxis, :]
    
    noise = np.sqrt(noise_power) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    return signals + noise

if __name__ == "__main__":
    num_samples = 256*1024
    sample_rate = 2.048e6 #1.024e6  
    center_freq = 433e6
    bandwidth =  2e5 #1.024e6
    gain = 40
    kraken = KrakenReceiver(center_freq, num_samples, 
                           sample_rate, bandwidth, gain, num_devices=5)
    while True:
        #kraken.read_streams()
        #print(kraken.buffer)
        doa_data = kraken.mem()
        pa.DOA_plot(doa_data, np.linspace(0, 179, 180)) #d=0.25
        plt.show()
    #kraken.read_streams()   
    #kraken.plot_fft()
    #get_device_info()
  