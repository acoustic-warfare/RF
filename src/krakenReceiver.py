import numpy as np
import SoapySDR as sp
from SoapySDR import *
from scipy.fft import fft
#from scipy.signal import signal 
import matplotlib.pyplot as plt
import time
from pyargus.directionEstimation import DOA_MUSIC
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

    def close_streams(self):
        for i, device in enumerate(self.devices):
            device.deactivateStream(self.streams[i])
            device.closeStream(self.streams[i])
            del device

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

    def _create_scanning_vectors(self, distance, theta_coverage):
        
        scanning_vectors = np.zeros((self.num_devices, theta_coverage), dtype=np.complex64)
        for theta in range(theta_coverage):
            theta_rad = np.deg2rad(theta)
            scanning_vector = np.exp(1j * 2 * np.pi * distance * 
                             np.sin(theta_rad) * np.arange(self.num_devices)) 

            #print(scanning_vector)   
            #scanning_vector /= np.linalg.norm(scanning_vector)
            scanning_vectors[:, theta] = scanning_vector

        return scanning_vectors
    

    def _create_scanning_vectors2(self, wave_length, theta_coverage):
        
        scanning_vectors = np.zeros((self.num_devices, theta_coverage), dtype=np.complex64)
        for theta in range(theta_coverage):
            theta_rad = np.deg2rad(theta)
            scanning_vector = np.exp(1j * 2 * np.pi * np.outer(np.arange(self.num_devices), np.sin(theta_rad)) /wave_length)   
            #scanning_vector /= np.linalg.norm(scanning_vector)
            #scanning_vectors[:, theta] = scanning_vector

        return scanning_vectors


    def doa_music(self):
        
        spatial_corr_matrix = (1/self.num_samples) * np.dot(self.buffer, self.buffer.conj().T) 
        #spatial_corr_matrix = np.real(spatial_corr_matrix)

        #Regularize the correlation matrix
        # regularization_parameter = 1e1
        # spatial_corr_matrix = spatial_corr_matrix + regularization_parameter * np.eye(spatial_corr_matrix.shape[0])

        wave_length = 0.6909
        distance_m = 0.25
        distance = distance_m/wave_length
        scanning_vectors = self._create_scanning_vectors2(wave_length, 180)

        #print(scanning_vectors.shape)
        #print(spatial_corr_matrix.shape)

        eigvals, _ = np.linalg.eig(spatial_corr_matrix)
        print("Eigenvalues of the spatial correlation matrix:", eigvals)
        #eigvals, _ = np.linalg.eig(regularized_matrix)
        #print("Eigenvalues of the regularized spatial correlation matrix:", eigvals)

        doa = DOA_MUSIC(spatial_corr_matrix, scanning_vectors, self.num_devices)

        print(doa)

def get_device_info():
        # Enumerate devices to get their arguments including serial numbers
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



if __name__ == "__main__":
    num_samples = 256*1024
    sample_rate = 2.048e6 #1.024e6  
    center_freq = 102.7e6
    bandwidth =  2e5 #1.024e6
    gain = 40
    kraken = KrakenReceiver(center_freq, num_samples, 
                           sample_rate, bandwidth, gain, num_devices=2)
    while True:
        kraken.read_streams()
        kraken.doa_music()
    # kraken.read_streams()   
    # kraken.plot_fft()
    #get_device_info()
  