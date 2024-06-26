import sys
sys.path.append("/usr/lib/python3/dist-packages/")
import adi
import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def butter_bandpass(lowcut, highcut, fs, order=5):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def init():

    line.set_data([], [])
    ax.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax.set_ylim(0, 1) 

    line1.set_data([], [])
    ax1.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax1.set_ylim(0, 1)

    line2.set_data([], [])
    ax2.set_xlim(-1,1)  #Full frequency range
    ax2.set_ylim(-1, 1)

    line_phase.set_data([], [])
    ax_phase.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax_phase.set_ylim(-1, 1)

    line1_phase.set_data([], [])
    ax1_phase.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax1_phase.set_ylim(-1, 1)

    line2_phase.set_data([], [])
    ax2_phase.set_xlim(-1, 1)  #Full frequency range
    ax2_phase.set_ylim(-90, 90)

    return line, line1, line2, line_phase, line1_phase, line2_phase

def update(frame):
    samples: complex = sdr.rx()
    ch0_samples= samples[0]
    ch1_samples = samples[1]

    # ch0_samples = butter_bandpass_filter(ch0_samples_unfiltered, 1, 2, sdr.rx_lo)
    # ch1_samples = butter_bandpass_filter(ch1_samples_unfiltered, 1, 2, sdr.rx_lo)


    freqs = np.fft.fftfreq(5000, d=1/samp_rate)


    fft_ch0_samples = fft.fft(ch0_samples)
    fft_ch1_samples = fft.fft(ch1_samples)
    abs_fft_ch0_samples = np.abs(fft_ch0_samples)
    abs_fft_ch1_samples = np.abs(fft_ch1_samples)
    
    # fft_ch0_samples[abs_fft_ch0_samples < 30] = 0
    # fft_ch1_samples[abs_fft_ch1_samples < 30] = 0
    max_index_ch0 = np.argmax(abs_fft_ch0_samples)
    fft_ch0_samples[abs_fft_ch0_samples != max(abs_fft_ch0_samples)] = 0
    fft_ch0_max = fft_ch0_samples[max_index_ch0]
    # fft_ch1_samples[abs_fft_ch1_samples != max(abs_fft_ch1_samples)] = 0
    
    # Set all elements to zero except for the one at max index
    #fft_ch1_samples_max = np.zeros_like(fft_ch1_samples)
    
    #fft_ch1_samples_max[max_index_ch0] = fft_ch1_samples[max_index_ch0]

    fft_ch1_max = fft_ch1_samples[max_index_ch0]

    ch0_angle = np.angle(fft_ch0_max, deg=False)
    # if (ch0_angle < 0):
    #     ch0_angle = 2*np.pi + ch0_angle
    ch1_angle = np.angle(fft_ch1_max, deg=False)
    # if (ch1_angle < 0):
    #     ch1_angle = 2*np.pi + ch1_angle

    #if (ch0_angle > ch1_angle):
    #    phase_difference = ch0_angle - ch1_angle
    #else:
    #    phase_difference = ch1_angle - ch0_angle
#
    #if (phase_difference > np.pi):
    #    phase_difference = np.pi - phase_difference

    phase_difference = ch1_angle - ch0_angle



    # Ensure the phase difference is in the range [-pi, pi]
    phase_difference = np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
    #phase_difference = np.degrees(phase_difference)

    index = (np.arcsin((phase_difference*wavelength)/(2*np.pi*d)))
    index = index 
    print("Angle of arrival is: ", np.degrees(index))


    print("The sin value is ", abs(np.sin(index)))
    print("The cos value is ", np.cos(index))


    #sdr.tx_destroy_buffer()
    
    line.set_data(freqs, abs_fft_ch0_samples)
    line_phase.set_data(freqs, ch0_angle)
    line1.set_data(freqs, abs_fft_ch1_samples)  
    line1_phase.set_data(freqs, ch1_angle)  
    if (index < 0):
        index = np.pi - np.abs(index)
    line2.set_data(np.cos(index), np.sin(index))
    line2_phase.set_data(freqs, np.degrees(index))

    ylim = np.max(np.maximum(abs_fft_ch1_samples, abs_fft_ch0_samples))
    phase_ylim_high = np.max(np.maximum(ch0_angle, ch1_angle))
    phase_ylim_low = np.min(np.minimum(ch0_angle, ch1_angle))

    ax.set_ylim(0, ylim * 1.1)
    ax1.set_ylim(0, ylim * 1.1)
    ax2.set_ylim(-1, 1)
    ax_phase.set_ylim(phase_ylim_low * 1.1, phase_ylim_high * 1.1)
    ax1_phase.set_ylim(phase_ylim_low * 1.1, phase_ylim_high * 1.1)
    ax2_phase.set_ylim(-90, 90)



    return line, line1, line2, line_phase, line1_phase, line2_phase


# Main script
if __name__ == "__main__":
    # Create radio
    sdr = adi.ad9361("ip:192.168.2.1")
    samp_rate = 20e6
    fc0 = int(200e3)

    '''Configure Rx properties'''
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(5e6)
    sdr.rx_lo = int(433.9e6)
    sdr.rx_buffer_size = 5000
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(40)
    sdr.rx_hardwaregain_chan1 = int(40)
    sdr.sample_rate = int(1e6) 



    s = int(sdr.sample_rate)
    N = 2**16
    ts = 1 / float(int(sdr.sample_rate))
    t = np.arange(0, N * ts, ts)
    i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
    q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
    iq0 = i0 + 1j * q0
    sdr.tx([iq0,iq0])


    c = 3e8
    freq = 433.9e6
    d = 0.35
    wavelength = c/freq    

    fs = int(sdr.sample_rate)
    N = 2**16
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
    q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
    iq0 = i0 + 1j * q0


    fig, ((ax, ax1, ax2), (ax_phase, ax1_phase, ax2_phase)) = plt.subplots(2, 3, figsize=(18,8))
    #fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15,6))

    line, = ax.plot([], []) 
    ax.grid()
    ax.set_title("Real-Time FFT of Received Samples (Channel 0)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    line1, = ax1.plot([], [])
    ax1.grid()
    ax1.set_title("Real-Time FFT of Received Samples (Channel 1)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")

    x = np.linspace(0, 2 * np.pi, 50)
    ax2.plot(np.cos(x), np.sin(x), "k", lw=0.3)
    line2, = ax2.plot(0, 0, "o")
    ax2.set_title("The unit circle")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    line_phase, = ax_phase.plot([], [], 'o') 
    ax_phase.grid()
    ax_phase.set_title("Real-Time phase of Received Samples (Channel 0)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("phase (radians)")

    line1_phase, = ax1_phase.plot([], [], 'o')
    ax1_phase.grid()
    ax1_phase.set_title("Real-Time phase of Received Samples (Channel 1)")
    ax1_phase.set_xlabel("Frequency (Hz)")
    ax1_phase.set_ylabel("Phase (radians)")

    line2_phase, = ax2_phase.plot([], [], 'o')
    ax2_phase.grid()
    ax2_phase.set_title("Angle of arrival")
    ax2_phase.set_xlabel("Frequency (Hz)")
    ax2_phase.set_ylabel("Phase (radians)")


    ani = animation.FuncAnimation(fig, update, init_func=init, frames=100, interval=500, blit=False)
    plt.show()
