from shmemIface import *
from iq_header import *


class KrakenReceiver():
    def __init__(self, num_antennas, center_freq, gain):

        self.daq_center_freq = center_freq  # MHz
        self.daq_rx_gain = gain  # [dB]

        #Shared memory setup
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        daq_path = os.path.join(os.path.dirname(root_path), "src/kraken/heimdall/heimdall_daq_fw")
        self.daq_shmem_control_path = os.path.join(os.path.join(daq_path, "Firmware"), "_data_control/")
        self.init_data_iface()

        self.iq_samples = np.empty(0)
        self.iq_header = IQHeader()
        self.num_antennas = num_antennas  # Number of receiver channels, updated after establishing connection

    def init_data_iface(self):
        # Open shared memory interface to capture the DAQ firmware output
        self.in_shmem_iface = inShmemIface(
            "delay_sync_iq", self.daq_shmem_control_path, read_timeout=5.0
        )
        if not self.in_shmem_iface.init_ok:
            self.in_shmem_iface.destory_sm_buffer()
            raise RuntimeError("Shared Memory Init Failed")
        print("Successfully Initilized Shared Memory")

    def get_iq_online(self):
        """
        This function obtains a new IQ data frame through the Ethernet IQ data or the shared memory interface
        """

        active_buff_index = self.in_shmem_iface.wait_buff_free()
        if active_buff_index < 0 or active_buff_index > 1:
            # If we cannot get the new IQ frame then we zero the stored IQ header
            self.iq_header = IQHeader()
            self.iq_samples = np.empty(0)
            raise RuntimeError(f"Terminating.., signal: {active_buff_index}")
            #return -1

        buffer = self.in_shmem_iface.buffers[active_buff_index]


        iq_header_bytes = buffer[:1024].tobytes()
        self.iq_header.decode_header(iq_header_bytes)

        # Initialization from header - Set channel numbers
        if self.M == 0:
            self.M = self.iq_header.active_ant_chs

        incoming_payload_size = (
            self.iq_header.cpi_length * self.iq_header.active_ant_chs * 2 * (self.iq_header.sample_bit_depth // 8)
        )

        shape = (self.iq_header.active_ant_chs, self.iq_header.cpi_length)
        iq_samples_in = buffer[1024 : 1024 + incoming_payload_size].view(dtype=np.complex64).reshape(shape)

        # Reuse the memory allocated for self.iq_samples if it has the
        # correct shape
        if self.iq_samples.shape != shape:
            self.iq_samples = np.empty(shape, dtype=np.complex64)

        np.copyto(self.iq_samples, iq_samples_in)

        self.in_shmem_iface.send_ctr_buff_ready(active_buff_index)

    def receive_iq_frame(self):
        """
        Called by the get_iq_online function. Receives IQ samples over the establed Ethernet connection
        """
        total_received_bytes = 0
        recv_bytes_count = 0
        iq_header_bytes = bytearray(self.iq_header.header_size)  # allocate array
        view = memoryview(iq_header_bytes)  # Get buffer

        print("Starting IQ header reception")

        # while total_received_bytes < self.iq_header.header_size:
        #     # Receive into buffer
        #     recv_bytes_count = self.socket_inst.recv_into(view, self.iq_header.header_size - total_received_bytes)
        #     view = view[recv_bytes_count:]  # reset memory region
        #     total_received_bytes += recv_bytes_count

        self.iq_header.decode_header(iq_header_bytes)
        # Uncomment to check the content of the IQ header
        # self.iq_header.dump_header()

        incoming_payload_size = (
            self.iq_header.cpi_length * self.iq_header.active_ant_chs * 2 * int(self.iq_header.sample_bit_depth / 8)
        )
        if incoming_payload_size > 0:
            # Calculate total bytes to receive from the iq header data
            total_bytes_to_receive = incoming_payload_size
            receiver_buffer_size = 2**18

            #self.logger.debug("Total bytes to receive: {:d}".format(total_bytes_to_receive))
            print(f"Total bytes to receive: {total_bytes_to_receive}")

            total_received_bytes = 0
            recv_bytes_count = 0
            iq_data_bytes = bytearray(total_bytes_to_receive + receiver_buffer_size)  # allocate array
            view = memoryview(iq_data_bytes)  # Get buffer

            # while total_received_bytes < total_bytes_to_receive:
            #     # Receive into buffer
            #     recv_bytes_count = self.socket_inst.recv_into(view, receiver_buffer_size)
            #     view = view[recv_bytes_count:]  # reset memory region
            #     total_received_bytes += recv_bytes_count

            print("IQ Data Succesfully Received")

            # Convert raw bytes to Complex float64 IQ samples
            self.iq_samples = np.frombuffer(iq_data_bytes[0:total_bytes_to_receive], dtype=np.complex64).reshape(
                self.iq_header.active_ant_chs, self.iq_header.cpi_length
            )

            self.iq_frame_bytes = bytearray() + iq_header_bytes + iq_data_bytes
            return self.iq_samples
        else:
            return 0
        

kraken = KrakenReceiver(5, 433.9, 40)
