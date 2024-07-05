from heimdall_daq_firmware.iq_eth_sink import *
from heimdall_daq_firmware.shmemIface import *
from heimdall_daq_firmware.delay_sync import *



class KrakenReceiver():
    def __init__(self, logging_level=10):
       
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        self.daq_center_freq = 100  # MHz
        self.daq_rx_gain = 0  # [dB]
        self.daq_agc = False

        #Shared memory setup
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        daq_path = os.path.join(os.path.dirname(root_path), "heimdall_daq_fw")
        self.daq_shmem_control_path = os.path.join(os.path.join(daq_path, "Firmware"), "_data_control/")
        self.init_data_iface()


        self.iq_frame_bytes = None
        self.iq_samples = np.empty(0)
        self.iq_header = IQHeader()
        self.M = 0  # Number of receiver channels, updated after establishing connection

    def init_data_iface(self):
        # Open shared memory interface to capture the DAQ firmware output
        self.in_shmem_iface = inShmemIface( 
            "delay_sync_iq", self.daq_shmem_control_path, read_timeout=5.0
        )
        if not self.in_shmem_iface.init_ok:
            self.logger.critical("Shared memory initialization failed")
            self.in_shmem_iface.destory_sm_buffer()
            return -1
        return 0
    
    def get_iq_online(self):
       
        active_buff_index = self.in_shmem_iface.wait_buff_free()
        if active_buff_index < 0 or active_buff_index > 1:
            self.logger.info("Terminating.., signal: {:d}".format(active_buff_index))
            # If we cannot get the new IQ frame then we zero the stored IQ header
            self.iq_header = IQHeader()
            self.iq_samples = np.empty(0)
            return -1

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
        print(self.iq_samples)
        self.in_shmem_iface.send_ctr_buff_ready(active_buff_index)


kraken = KrakenReceiver()
