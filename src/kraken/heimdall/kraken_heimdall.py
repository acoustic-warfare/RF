from shmemIface import *
from iq_header import *



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

