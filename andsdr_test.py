import sys

sys.path.append("/usr/local/lib/python3/dist-packages")
import uhd
import numpy as np
import matplotlib.pyplot as plt

usrp = uhd.usrp.MultiUSRP()

total_num_samples = 1500  # number of samples received
buffer_length = 100

center_freq = 102.7e6  # Hz
sample_rate = 1e6  # Hz
gain = 50  # dB

usrp.set_rx_rate(sample_rate, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_rx_gain(gain, 0)

# Set up the stream and receive buffer
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
# st_args.channels = [0]
st_args.channels = [0, 1]
# print(st_args.args)
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)

streamer_max_samples = streamer.get_max_num_samps()
print("streamer maxnum: " + str(streamer_max_samples))

# recv_buffer = np.zeros((1, buffer_length), dtype=np.complex64)
recv_buffer = np.zeros((2, buffer_length), dtype=np.complex64)


# Start Stream
# stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
# stream_cmd.stream_now = True
# stream_cmd
stream_cmd.stream_now = False
# stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(0.1)
# stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(super(MultiUSRP, self).get_reak_secs() + 0.05)
stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(
    usrp.get_time_now().get_real_secs() + 0.50
)
stream_cmd.num_samps = total_num_samples
streamer.issue_stream_cmd(stream_cmd)

# Receive Samples
# samples = np.zeros(num_samps, dtype=np.complex64)
# for i in range(num_samps//1000):
#    nsamps = streamer.recv(recv_buffer, metadata)
#    print("nsamps:" + str(nsamps))
#    samples[i*1000:(i+1)*1000] = recv_buffer[0]
#    print(samples[0:10])

# Receive Samples.  recv() will return zeros, then our samples, then more zeros, letting us know it's done
waiting_to_start = True  # keep track of where we are in the cycle (see above comment)
nsamps = 0
i = 0
samples = np.zeros(total_num_samples, dtype=np.complex64)
while nsamps != 0 or waiting_to_start:
    nsamps = streamer.recv(recv_buffer, metadata)
    # print(metadata.start_of_burst)
    print(
        "more fragments: "
        + str(metadata.more_fragments)
        + ", sum:"
        + str(np.sum(recv_buffer))
    )
    # print(recv_buffer)
    if (
        metadata.error_code != uhd.types.RXMetadataErrorCode.none
        and metadata.error_code != uhd.types.RXMetadataErrorCode.timeout
    ):
        print("Error:" + str(metadata.error_code))

    # print("nsamps:" + str(nsamps))
    if nsamps and waiting_to_start:
        waiting_to_start = False
        # samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
        # print(recv_buffer[0][0:nsamps])
    # elif nsamps:
    # samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
    # print(recv_buffer[0][0:nsamps])
    i += nsamps
    print("nsamps:" + str(nsamps) + ", i=", str(i))

# streamer.recv(recv_buffer, metadata)


# Stop Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)

print(len(samples))
print(samples)
print(np.sum(samples))
plt.plot(abs(samples), marker="o", linestyle="-", color="b")
plt.title("Sample Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# print(samples[0:1000])
