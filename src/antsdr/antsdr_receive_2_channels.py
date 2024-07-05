import sys

sys.path.append("/usr/local/lib/python3/dist-packages")

import uhd
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def receive_samples(usrp, num_samples, bandwidth, center_freq, sample_rate):
    # Channel 0 setup
    usrp.set_rx_rate(sample_rate, 0)
    usrp.set_rx_freq(center_freq, 0)
    usrp.set_rx_bandwidth(bandwidth, 0)
    usrp.set_rx_gain(50, 0)
    # Channel 1 setup
    usrp.set_rx_rate(sample_rate, 1)
    usrp.set_rx_bandwidth(bandwidth, 1)
    usrp.set_rx_freq(center_freq, 1)
    usrp.set_rx_gain(56, 1)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0, 1]
    # st_args.recv_frame_size = 512 * 1024
    rx_streamer = usrp.get_rx_stream(st_args)

    rx_metadata = uhd.types.RXMetadata()
    buffer = np.zeros((2, num_samples), dtype=np.complex64)

    # Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = (
        False  # Ensure that streaming starts immediately and aligned
    )
    stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(
        usrp.get_time_now().get_real_secs() + 0.5
    )
    # stream_cmd.time_spec += uhd.libpyuhd.types.time_spec.from_ticks(10000, sample_rate)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # Receive samples
    rx_streamer.recv(buffer, rx_metadata, timeout=5.0)

    # Stop streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)

    return buffer


def init():
    line.set_data([], [])
    ax.set_xlim(-sample_rate / 2, sample_rate / 2)  # Full frequency range
    ax.set_ylim(0, 1)

    line1.set_data([], [])
    ax1.set_xlim(-sample_rate / 2, sample_rate / 2)  # Full frequency range
    ax1.set_ylim(0, 1)

    line_phase.set_data([], [])
    ax_phase.set_xlim(-sample_rate / 2, sample_rate / 2)  # Full frequency range
    ax_phase.set_ylim(-1, 1)

    line1_phase.set_data([], [])
    ax1_phase.set_xlim(-sample_rate / 2, sample_rate / 2)  # Full frequency range
    ax1_phase.set_ylim(-1, 1)

    return line, line1, line_phase, line1_phase


def update(frame):
    samples = receive_samples(usrp, num_samples, bandwidth, center_freq, sample_rate)
    ch0_samples = samples[0]
    ch1_samples = samples[1]
    freqs = np.fft.fftfreq(num_samples, d=1 / sample_rate)
    print(f"Channel 0 num samples: {len(ch0_samples)}")
    print(f"Channel 1 num samples: {len(ch1_samples)}")

    fft_ch0_samples = fft.fft(ch0_samples)
    fft_ch1_samples = fft.fft(ch1_samples)
    abs_fft_ch0_samples = np.abs(fft_ch0_samples)
    abs_fft_ch1_samples = np.abs(fft_ch1_samples)

    # fft_ch0_samples[abs_fft_ch0_samples < 30] = 0
    # fft_ch1_samples[abs_fft_ch1_samples < 30] = 0
    max_index_ch0 = np.argmax(abs_fft_ch0_samples)
    fft_ch0_samples[abs_fft_ch0_samples != max(abs_fft_ch0_samples)] = 0
    # fft_ch1_samples[abs_fft_ch1_samples != max(abs_fft_ch1_samples)] = 0

    # Set all elements to zero except for the one at max index
    fft_ch1_samples_max = np.zeros_like(fft_ch1_samples)
    fft_ch1_samples_max[max_index_ch0] = fft_ch1_samples[max_index_ch0]

    ch0_angle = np.angle(fft_ch0_samples, deg=True)
    ch1_angle = np.angle(fft_ch1_samples_max, deg=True)

    line.set_data(freqs, abs_fft_ch0_samples)
    line_phase.set_data(freqs, ch0_angle)
    line1.set_data(freqs, abs_fft_ch1_samples)
    line1_phase.set_data(freqs, ch1_angle)

    ylim = np.max(np.maximum(abs_fft_ch1_samples, abs_fft_ch0_samples))
    phase_ylim_high = np.max(np.maximum(ch0_angle, ch1_angle))
    phase_ylim_low = np.min(np.minimum(ch0_angle, ch1_angle))

    ax.set_ylim(0, ylim * 1.1)
    ax1.set_ylim(0, ylim * 1.1)
    ax_phase.set_ylim(phase_ylim_low * 1.1, phase_ylim_high * 1.1)
    ax1_phase.set_ylim(phase_ylim_low * 1.1, phase_ylim_high * 1.1)

    return line, line1, line_phase, line1_phase


# Main script
if __name__ == "__main__":
    usrp = uhd.usrp.MultiUSRP()  # Create a USRP device instance
    num_samples = 50000
    center_freq = 102.7e6  # Center frequency in Hz
    sample_rate = 1e5  # Sample rate in samples per second
    bandwidth = 200e3

    fig, ((ax, ax1), (ax_phase, ax1_phase)) = plt.subplots(2, 2, figsize=(18, 8))
    # fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15,6))

    (line,) = ax.plot([], [])
    ax.grid()
    ax.set_title("Real-Time FFT of Received Samples (Channel 0)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    (line1,) = ax1.plot([], [])
    ax1.grid()
    ax1.set_title("Real-Time FFT of Received Samples (Channel 1)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")

    (line_phase,) = ax_phase.plot([], [], "o")
    ax_phase.grid()
    ax_phase.set_title("Real-Time phase of Received Samples (Channel 0)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("phase (radians)")

    (line1_phase,) = ax1_phase.plot([], [], "o")
    ax1_phase.grid()
    ax1_phase.set_title("Real-Time phase of Received Samples (Channel 1)")
    ax1_phase.set_xlabel("Frequency (Hz)")
    ax1_phase.set_ylabel("Phase (radians)")

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=100, interval=500, blit=False
    )
    plt.show()
