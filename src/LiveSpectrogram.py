from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import adi


class LiveSpectrogram():
    
    def __init__(self, segment_size, overlap, window_function, frames, num_samples, sample_rate, sdr, rx_lo):
        self.window_function = window_function
        self.sdr = sdr
        self.segment_size = segment_size
        self.overlap = overlap
        self.frames = frames
        self.sample_rate = sample_rate
        self.increment = 0
        self.num_samples = num_samples
        self.step_size = self.segment_size - self.overlap
        self.num_segments = (num_samples- self.overlap) // self.step_size
        self.big_spectrogram = []
        self.spectrogram = np.zeros((frames, segment_size // 2 + 1))
        self.rx_lo = rx_lo

        self.fig, self.ax = plt.subplots()
    
        self.im = self.ax.imshow(self.spectrogram, aspect='auto', extent = [self.rx_lo-bandwidth/2, self.rx_lo+bandwidth/2, num_samples/self.rx_lo, 0], cmap = "viridis", vmin = 30, vmax = 70, origin='lower')
        
        self.fig.colorbar(self.im, ax=self.ax, label='Magnitude (dB)')
        self.ax.set_xlabel('Frequency (kHz)')
        self.ax.set_ylabel('Time (s)')
        self.ax.set_title('Live Spectrogram')

    def creating_spectrogram(self, samples):
        segments = []
        # print("num of samples ", self.num_samples)
        # print("Num of segments: ", self.num_segments)
        for i in range(self.num_segments):
            segments.append(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*self.overlap:i*self.overlap + self.segment_size]))**2))) #from pySDR
            #segments[i] = 10*np.log10(np.abs((np.fft.rfft(samples[i*overlap:i*overlap + self.segment_size])))**2) #one liner.
        #print("Segements: ", segments)
        return segments
 
    
    def update(self, frame):
        samples = self.sdr.rx()
        self.big_spectrogram.append(self.creating_spectrogram(samples))
        if (len(self.big_spectrogram) > segment_size):
            self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])[0]
        else:
            self.spectrogram = np.array(self.big_spectrogram[self.increment:(self.increment + segment_size)])[0]


        
        self.increment = self.increment+1
        #print(self.spectrogram)
        # self.spectrogram = np.array(self.queue)
        self.im.set_array(self.spectrogram)
        return self.im,

if __name__ == '__main__':
   
    #Creating Pluto
    samp_rate = 30e6    # must be <=30.72 MHz if both channels are enabled
    num_samples = 32768
    rx_lo = 105e6
    rx_mode = "manual"  # can be "manual" or "slow_attack"
    rx_gain0 = 40
    fft_size = 256 
    bandwidth = int(30e6)

    '''Create Radio'''
    sdr = adi.ad9361(uri='ip:192.168.2.1')

    '''Configure properties for the Radio'''
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = bandwidth
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)

    sdr.rx_buffer_size = int(num_samples)


    segment_size = 256
    overlap = 128
    window_function = np.hanning(segment_size)
    frames = 60
    # sample_rate = 1e6
    # t = np.arange(1024*1000)/sample_rate # time vector
    # f = 50e3 # freq of tone
    # samples = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))
    live_spectrogram = LiveSpectrogram(segment_size, overlap, window_function, frames, num_samples, samp_rate, sdr, rx_lo)
    ani = animation.FuncAnimation(live_spectrogram.fig, live_spectrogram.update, 
                                  frames=range(num_samples // overlap - frames), interval=1000, blit=True
    )
    plt.show()



