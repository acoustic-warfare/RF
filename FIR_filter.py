import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the parameters of the band-pass FIR filter
numtaps = 101  # Number of filter taps (filter length)
fc = 433.9e6
fs = 4*fc  # Sampling rate (Hz)'
bandwidth = 0.05*fc
zero = np.finfo(float).eps
lowcut = zero  # Lower cutoff frequency (Hz)
highcut = bandwidth/2  # Upper cutoff frequency (Hz)


# Design a band-pass FIR filter using the firwin function
taps = signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False, window='hamming')

# Compute the frequency response of the filter
freq_response = np.abs(np.fft.fft(taps, 1000))  # Compute FFT and take magnitude

freqz = np.fft.fftfreq(freq_response.size, 1/fs)

# # Apply the filter to the signal
# filtered_x = signal.lfilter(taps, 1.0, x)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(freqz, 20 * np.log10(freq_response))
plt.title('Frequency Response of Band-Pass FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.ylim(-80, 10)
plt.xlim(0, fs / 2)
plt.tight_layout()
plt.show()

