import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from struct import pack, unpack

"""Direction finding with PlutoSDR using beamforming. This also contains how to save the data collected from the sdr to a file. 
This direction finding is inspired by pysdr:s examples. Check out https://pysdr.org/content/doa.html for a better understanding. """

def create_new_file(num):
    #Creates a new file
    basename = "file" + str(num)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = "_".join([basename, suffix]) 
    f = open(filename, "x")
    return filename

def save_data(filename, input):
    #Saves the data to file
    data = ', '.join(str(e) for e in input)
    f = open(filename, "a")
    f.write(data + "\n")
    f.close()

def w_mvdr(theta, buffer):
   """This is the method for Minimum Variance Distortionless Response (MVDR)"""
   buffer_array = np.array(buffer)
   s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
   s = s.reshape(-1,1) # make into a column vector (size 3x1)
   R = (buffer_array @ buffer_array.conj().T)/buffer_array.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
   Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
   return w

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
    sdr.writeStream(txStream, [iq0], len(iq0))

    #create buffer
    
    #receive some samples
    sr = sdr.readStream(rxStream, buffs, num_samples)
    rx1_data = buffs[0]
    rx2_data = buffs[1]
    print("The first filenames is: ", filenames[0])
    print("The first filename is: ", filenames[1])
    save_data(filenames[0], buffs[0])
    save_data(filenames[1], buffs[1])
    rx1_fft = (fft.fft(rx1_data))
    abs_rx1_fft = abs(rx1_fft)
    rx2_fft = (fft.fft(rx2_data))
    abs_rx2_fft = abs(rx2_fft)
    print(sr.ret) #num samples or error code
    print(sr.flags) #flags set by receive operation
    print(sr.timeNs) #timestamp for receive buffer

    # The beamforming
    theta_scan = np.linspace(-1*np.pi, np.pi, 9000) # 9000 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        w = w_mvdr(theta_i, buffs) # 3x1
        X_weighted = w.conj().T @ buffs # apply weights
        power_dB = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        results.append(power_dB)
    results -= np.max(results) # normalize

    theta = theta_scan[np.argmax(results)]
    print("Theta before if: ", np.degrees(theta))
    if (abs(theta)>(np.pi/2)):
        if (theta > 0):
            theta = np.pi - theta
        else:
            theta = (-1)*np.pi - theta

    thetaInDegrees = np.degrees(theta)
    print("Theta after if: ", thetaInDegrees)

    freqs = np.fft.fftfreq(1024, d=1/samp_rate)
    line.set_data(freqs, rx1_fft)
    line1.set_data(freqs, rx2_fft)  
    
    # if(index> np.pi/2):
    #     index = np.pi/2 + (np.pi-index)
    # else:
    #     index = np.pi/2 - index
    line2.set_data(np.cos(theta), np.sin(theta))

    ylim = np.max(np.maximum(abs_rx1_fft, abs_rx2_fft))

    ax.set_ylim(0, ylim * 1.1)
    ax1.set_ylim(0, ylim * 1.1)
    ax2.text(-2, -14, "{} deg".format(theta))    
    #ax2.set_ylim(0, 1)

    sdr.deactivateStream(txStream) #stop streaming
    sdr.closeStream(txStream)

    return line, line1, line2


if __name__ == "__main__":
#create filenames
    filenames = []
    for i in range(2):
        filenames.append(create_new_file(i)) 
        print("Filename: ", filenames[i])

# Create radio
    '''Setup'''
    samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
    NumSamples = 2**12
    rx_lo = 433e6
    rx_mode = "manual"  # can be "manual" or "slow_attack"
    rx_gain = 40
    tx_lo = rx_lo
    tx_gain = -3
    fc0 = int(200e3)
    Nr = 2

    ''' Set distance between Rx antennas '''
    d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
    wavelength = 3E8/rx_lo              # wavelength of the RF carrier
    d = d_wavelength*wavelength         # distance between elements in meters
    print("Set distance between Rx Antennas to ", int(d*1000), "mm")

    #enumerate devices
    results = SoapySDR.Device.enumerate()
    for result in results: print(result)

    #create device instance
    #args can be user defined or from the enumeration result
    args = dict(driver="plutosdr")
    sdr = SoapySDR.Device(args)

    '''Program Tx'''
    fs = int(samp_rate)
    N = 2**16
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
    q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
    iq0 = i0 + 1j * q0

    #apply settings
    sdr.setSampleRate(SOAPY_SDR_TX, 0, 2e6)
    sdr.setFrequency(SOAPY_SDR_TX, 0, 433e6)

    #setup a stream (complex floats)
    txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    sdr.activateStream(txStream) #start streaming

    #Preparing to receive by setting up the channels
    for channel in [0, 1]:
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, channel, samp_rate)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, rx_lo)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, channel, 50)

    #setup a stream (complex floats)
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0,1])
    sdr.activateStream(rxStream) #start streaming

    #create a re-usable buffer for rx samples
    num_samples = 1024
    buffs = [np.array([0]*num_samples, np.complex64), np.array([0]*num_samples, np.complex64)]

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



    sdr.deactivateStream(rxStream) #stop streaming
    sdr.closeStream(rxStream)
