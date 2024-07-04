import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import sys
import pyargus.directionEstimation as pa
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from rtlsdr import *
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor
from pylab import *


sdr = RtlSdr()

serial_numbers = RtlSdr.get_device_serial_addresses()
print(serial_numbers)

#sdr.num_samples = 1024*128
sdr.sample_rate = 2.048e6
sdr.center_freq = 433e6
sdr.bandwidth =  2e5 
sdr.gain = 40

#Design a FIR filter using the firwin function
numtaps = 51  # Number of filter taps (filter length)
fc = sdr.center_freq
fs = 4*fc
bandwidth = 0.3*fc
highcut = bandwidth/2  # Upper cutoff frequency (Hz)
taps = signal.firwin(numtaps, [highcut], fs=fs, pass_zero=True)
filter = taps


num = [0.0, 1.0]
den = [1e-6 , 1.0]          
dt = 1e-6
discrete_system = signal.cont2discrete((num, den), dt)
b = np.array(discrete_system[0].flatten(), dtype=np.float64)
a = np.array(discrete_system[1].flatten(), dtype=np.float64)

samples = sdr.read_samples(256*1024)

fir_filter_Samples = signal.lfilter(filter, 1.0, samples)
lti_filter_Samples = signal.lfilter(b, a, samples)
sdr.close()

psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')

show()

class KrakenReceiver():

    """
    Represents a Kraken receiver system for signal processing.

    Attributes:
    num_devices : int
        Number of receiver devices.
    center_freq : float
        Center frequency of the signal.
    num_samples : int
        Number of samples per device per capture.
    sample_rate : float
        Sampling rate in samples per second.
    bandwidth : float
        Bandwidth of the signal.
    gain : float
        Gain of the receiver devices.
    buffer : numpy.ndarray
        Buffer to store captured signal data, shape (num_devices, num_samples).
    devices : numpy.ndarray
        List of device objects representing the receiver devices.
    streams : numpy.ndarray
        List of streams associated with each device for data capture.
    x : float
        X-coordinate of the receiver location (scaled by antenna distance).
    y : float
        Y-coordinate of the receiver location.
    num_devices : int, optional
        Number of receiver devices (default is 5).
    filter : scipy filter object
        A signal processing filter
    """
    def __init__(self, center_freq, num_samples, sample_rate, bandwidth, gain, antenna_distance, x, y, 
                 num_devices=5, simulation = 0, f_type = 'LTI', detection_range = 360):
        
        self.num_devices = num_devices
        self.center_freq = center_freq
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.f_type = f_type
        self.devices, self.streams = (0,0) #self._setup_devices()

        self.simulation = simulation
        self.x = x * antenna_distance
        self.y = y * antenna_distance
        self.detection_range = detection_range

        if simulation:
            pass
            #self.buffer = signals_linear([self.center_freq], [30] ,self.num_devices, self.num_samples, self.x, antenna_distance)
            #self.buffer = signals_circular([self.center_freq], [300] ,self.num_devices, self.num_samples, self.x, self.y, antenna_distance)
        else:
            self.buffer = np.zeros((self.num_devices, num_samples), dtype=np.complex64)


