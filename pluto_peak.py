"""
Jon Kraft, Oct 30 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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



def calcTheta(phase):
    # calculates the steering angle for a given phase delta (phase is in deg)
    # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
    arcsin_arg = np.deg2rad(phase)*3E8/(2*np.pi*rx_lo*d)
    arcsin_arg = max(min(1, arcsin_arg), -1)     # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.arcsin(arcsin_arg)
    print("Theta: ", calc_theta)
    return calc_theta

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

def init():

    line.set_data([], [])
    ax.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax.set_ylim(0, 1) 

    line1.set_data([], [])
    ax1.set_xlim(-samp_rate / 2, samp_rate / 2)  #Full frequency range
    ax1.set_ylim(0, 1)

    line2.set_data([], [])
    ax2.set_xlim(-1,1)  #Full frequency range
    #ax2.set_ylim(0, 1)

    return line, line1, line2

def update(frame):
    sdr.tx([iq0,iq0])  # Send Tx data.

    '''Collect Data'''
    for i in range(20):  
        # let Pluto run for a bit, to do all its calibrations, then get a buffer
        data = sdr.rx()

    data = sdr.rx()
    ch0_samples=data[0]
    ch1_samples=data[1]
    peak_sum = []
    delay_phases = np.arange(-180, 180, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = ch1_samples* np.exp(1j*np.deg2rad(phase_delay+phase_cal))
        delayed_sum = dbfs(ch0_samples + delayed_Rx_1)
        peak_sum.append(np.max(delayed_sum[signal_start:signal_end]))
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    print("Phase shift is: ", (peak_delay))
    steer_angle = calcTheta(peak_delay) + np.pi/2


    freqs = np.fft.fftfreq(4096, d=1/samp_rate)


    fft_ch0_samples = fft.fft(ch0_samples)
    fft_ch1_samples = fft.fft(ch1_samples)
    abs_fft_ch0_samples = np.abs(fft_ch0_samples)
    abs_fft_ch1_samples = np.abs(fft_ch1_samples)
    
    # fft_ch0_samples[abs_fft_ch0_samples < 30] = 0
    # fft_ch1_samples[abs_fft_ch1_samples < 30] = 0



    # Ensure the phase difference is in the range [-pi, pi]
    print("Angle of arrival is: ", np.degrees(steer_angle))


    print("The sin value is ", abs(np.sin(steer_angle)))
    print("The cos value is ", np.cos(steer_angle))


    #sdr.tx_destroy_buffer()
    
    line.set_data(freqs, abs_fft_ch0_samples)
    line1.set_data(freqs, abs_fft_ch1_samples)  
    
    # if(index> np.pi/2):
    #     index = np.pi/2 + (np.pi-index)
    # else:
    #     index = np.pi/2 - index
    line2.set_data(np.cos(steer_angle), np.sin(steer_angle))

    ylim = np.max(np.maximum(abs_fft_ch1_samples, abs_fft_ch0_samples))

    ax.set_ylim(0, ylim * 1.1)
    ax1.set_ylim(0, ylim * 1.1)
    ax2.text(-2, -14, "{} deg".format(steer_angle))    
    #ax2.set_ylim(0, 1)

    sdr.tx_destroy_buffer()


    return line, line1, line2


# Main script
if __name__ == "__main__":
    # Create radio
    '''Setup'''
    samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
    NumSamples = 2**12
    rx_lo = 433.7e6
    rx_mode = "manual"  # can be "manual" or "slow_attack"
    rx_gain0 = 40
    rx_gain1 = 40
    tx_lo = rx_lo
    tx_gain = -3
    fc0 = int(200e3)
    phase_cal = -128
    num_scans = 5

    ''' Set distance between Rx antennas '''
    d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
    wavelength = 3E8/rx_lo              # wavelength of the RF carrier
    d = d_wavelength*wavelength         # distance between elements in meters
    print("Set distance between Rx Antennas to ", int(d*1000), "mm")

    '''Create Radio'''
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    '''Configure properties for the Radio'''
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0*3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_hardwaregain_chan1 = int(rx_gain1)
    sdr.rx_buffer_size = int(NumSamples)
    sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
    sdr.tx_rf_bandwidth = int(fc0*3)
    sdr.tx_lo = int(tx_lo)
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = int(tx_gain)
    sdr.tx_hardwaregain_chan1 = int(-88)
    sdr.tx_buffer_size = int(2**18)

    '''Program Tx and Send Data'''
    fs = int(sdr.sample_rate)
    N = 2**16
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
    q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
    iq0 = i0 + 1j * q0
    
    # Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
    xf = np.fft.fftfreq(NumSamples, ts)
    xf = np.fft.fftshift(xf)/1e6
    signal_start = int(NumSamples*(samp_rate/2+fc0/2)/samp_rate)
    signal_end = int(NumSamples*(samp_rate/2+fc0*2)/samp_rate)

    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(18,8))

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
    
    x = np.linspace(0, np.pi, 50)
    ax2.plot(np.cos(x), np.sin(x), "k", lw=0.3)
    line2, = ax2.plot(0, 1, "o")
    ax2.set_title("The unit circle")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3.plot([], [])



    ani = animation.FuncAnimation(fig, update, init_func=init, frames=100, interval=500, blit=False)
    plt.show()





    # if Plot_Compass==False:
    #     fig, (ax1, ax2) = plt.subplots(2)
    #     # plt.plot(delay_phases, peak_sum)
    #     # plt.axvline(x=peak_delay, color='r', linestyle=':')
    #     # plt.text(-180, -26, "Peak signal occurs with phase shift = {} deg".format(round(peak_delay,1)))
    #     # plt.text(-180, -28, "If d={}mm, then steering angle = {} deg".format(int(d*1000), steer_angle))
    #     # plt.ylim(top=0, bottom=-30)        
    #     # plt.xlabel("phase shift [deg]")
    #     # plt.ylabel("Rx0 + Rx1 [dBfs]")
    #     # plt.draw()
    #     # plt.show()
    #     fig.suptitle("Plotting FFT:s")
    #     ax1.plot(np.fft.fft(Rx_0))
    #     ax2.plot(np.fft.fft(Rx_1))
    #     plt.draw()
    #     plt.show()
    # else:
    #     fig = plt.figure(figsize=(3,3))
    #     ax = plt.subplot(111,polar=True)
    #     ax.set_theta_zero_location('N')
    #     ax.set_theta_direction(-1)
    #     ax.set_thetamin(-90)
    #     ax.set_thetamax(90)
    #     ax.set_rlim(bottom=-20, top=0)
    #     ax.set_yticklabels([])
    #     ax.vlines(np.deg2rad(steer_angle),0,-20)
    #     ax.text(-2, -14, "{} deg".format(steer_angle))
    #     plt.draw()
    #     plt.show()

