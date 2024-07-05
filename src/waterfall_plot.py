import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import matplotlib.pyplot as plt
import numpy as np
import adi

# Create radio
'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 1024
rx_lo = 433e6
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 40
tx_lo = rx_lo
tx_gain = -3
fc0 = int(200e3)

'''Create Radio'''
sdr = adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Radio'''
sdr.rx_enabled_channels = [0]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(fc0*3)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(NumSamples)

fft_size = 1024

data = sdr.rx()

for i in range(1024):
    data = sdr.rx()

num_rows = len(data) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[i*fft_size:(i+1)*fft_size])))**2)

plt.imshow(spectrogram, aspect='auto', extent = [samp_rate/-2/1e6, samp_rate/2/1e6, len(data)/samp_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.show()