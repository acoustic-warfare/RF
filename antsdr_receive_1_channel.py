import sys
sys.path.append('/usr/local/lib/python3/dist-packages')
import uhd
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def receive_samples(usrp, num_samples, bandwidth, center_freq, sample_rate):
    usrp.set_rx_rate(sample_rate)
    usrp.set_rx_freq(center_freq)  
    usrp.set_rx_bandwidth(bandwidth)  
    usrp.set_rx_gain(50) 

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")  
    rx_streamer = usrp.get_rx_stream(st_args)

    rx_metadata = uhd.types.RXMetadata()
    buffer = np.zeros((num_samples,), dtype=np.complex64)

    #Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # Receive samples
    recv_samples = rx_streamer.recv(buffer, rx_metadata, timeout=5.0)

    # Stop streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)

    return buffer[:recv_samples]

def init():
    line.set_data([], [])
    ax.set_xlim(-sample_rate / 2, sample_rate / 2)  #Full frequency range
    ax.set_ylim(0,1)

    line_phase.set_data([], [])
    ax_phase.set_xlim(-sample_rate / 2, sample_rate / 2)  #Full frequency range
    ax_phase.set_ylim(-1,1)

    return line,

def update(frame):
    samples = receive_samples(usrp, num_samples, bandwidth, center_freq, sample_rate)
    freqs = np.fft.fftfreq(num_samples, d=1/sample_rate)
    print(f"Num samples: {len(samples)}")

    samples = fft.fft(samples)
    samples_abs = np.abs(samples)

    samples[samples_abs < 300] = 0
        
    samples_angle = np.angle(samples, deg=True)
    
    ax.set_ylim(0, np.max(samples_abs) * 1.1)
    ax_phase.set_ylim(np.min(samples_angle) *1.1, np.max(samples_angle) *1.1)
    line.set_data(freqs, samples_abs)
    line_phase.set_data(freqs, samples_angle)
    
    return line, line_phase

if __name__ == "__main__":
    usrp = uhd.usrp.MultiUSRP()  #Create a USRP device instance
    num_samples = 500000
    center_freq = 102.7e6  #Center frequency in Hz 
    sample_rate = 1e6 #Sample rate in samples per second 
    bandwidth = 200e3

    fig, (ax, ax_phase) = plt.subplots(1,2, figsize=(15,6))
    line, = ax.plot([], []) 
    ax.grid()
    ax.set_title("Real-Time FFT of Received Samples")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    line_phase, = ax_phase.plot([], [], 'o') 
    ax_phase.grid()
    ax_phase.set_title("Real-Time phase of Received Samples")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("phase (deg)")

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=100, interval=500, blit=False)
    plt.show()