import numpy as np
from rtlsdr import RtlSdr
from scipy.fft import fft

# Create an SDR device instance
sdr = RtlSdr()

# Configure SDR settings
sdr.sample_rate = 2.048e6  #Hz
sdr.center_freq = 102.7e6    #Hz
sdr.gain = 'auto'
#num_samples = 

#Capture samples
samples = sdr.read_samples(256*1024)

fft_result = fft(samples)
freq = np.fft.fftfreq(256*1024, d=1/2.048e6)

import matplotlib.pyplot as plt
#plt.plot(freq, np.abs(fft_result))
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_result)))
plt.ylabel("Amplitude")
plt.xlabel("Frequency(Hz)")
plt.show()

# Clean up
sdr.close()
