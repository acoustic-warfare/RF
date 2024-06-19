from hackrf import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import welch
import time


# Initialize HackRF
def init_hackrf():
    hrf = HackRF()
    hrf.sample_rate = 10e6  # 1 MHz to 6 GHz
    hrf.center_freq = 0
    return hrf


hrf = init_hackrf()

# Set up plot
fig, ax = plt.subplots()
(line,) = ax.plot([], [])
ax.set_xlim(-hrf.sample_rate / 2 / 1e6, hrf.sample_rate / 2 / 1e6)
ax.set_ylim(-110, -60)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Relative power (dB)")


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    global hrf
    retries = 3
    for i in range(retries):
        try:
            samples = hrf.read_samples(2e6)
            freqs, Pxx = welch(
                samples, fs=hrf.sample_rate, nperseg=1024, return_onesided=False
            )
            line.set_data(
                np.fft.fftshift(freqs) / 1e6, 10 * np.log10(np.fft.fftshift(Pxx))
            )
            return (line,)
        except IOError as e:
            print(f"Error in hackrf_start_rx: {e}")
            if i < retries - 1:
                print("Retrying...")
                time.sleep(1)  # Wait a bit before retrying
            else:
                print("Reinitializing HackRF...")
                hrf.close()
                hrf = init_hackrf()
    return (line,)


ani = FuncAnimation(
    fig, update, init_func=init, blit=True, interval=1000, cache_frame_data=False
)
plt.show()

# Close HackRF when done
hrf.close()
