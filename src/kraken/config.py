import configparser
import numpy as np

def kraken_config(center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type, detection_range, waraps):
    """
    Configures and writes the DAQ (Data Acquisition) chain settings to an INI file.

    Parameters:
    -----------
    center_freq : float
        Center frequency for data acquisition.
    num_samples : int
        Number of samples to be acquired.
    sample_rate : float
        Sampling rate for data acquisition.
    antenna_distance : float
        Distance between adjacent antennas in the array.
    x : list of float
        List of x-coordinates for the antenna array elements.
    y : list of float
        List of y-coordinates for the antenna array elements.
    f_type : str
        Type of the function used in processing (e.g., 'FFT', 'MUSIC').
    detection_range : int
        Range of angles to scan for detection, in degrees.

    """
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    config.set('daq', 'center_freq', str(center_freq))
    config.set('daq', 'sample_rate', str(sample_rate))
    config.set('daq', 'gain', str(gain))
    config.set('daq', 'daq_buffer_size', str(num_samples))
    config.set('pre_processing', 'cpi_size', str(num_samples))
    config.set('variables', 'antenna_distance', str(antenna_distance))
    config.set('variables', 'f_type', f_type)
    config.set('variables', 'detection_range', str(detection_range))
    config.set('variables', 'x', ','.join(map(str, x)))
    config.set('variables', 'y', ','.join(map(str, y)))
    config.set('variables', 'waraps', str(waraps))

    with open('heimdall_daq_fw/Firmware/daq_chain_config.ini', 'w') as configfile:
        config.write(configfile)

def read_kraken_config():
    """
    Reads the DAQ (Data Acquisition) chain settings from an INI file.

    Returns:
    --------
    center_freq : int
        Center frequency for data acquisition.
    num_samples : int
        Number of samples to be acquired.
    sample_rate : int
        Sampling rate for data acquisition.
    antenna_distance : float
        Distance between adjacent antennas in the array.
    x : ndarray
        Array of x-coordinates for the antenna array elements.
    y : ndarray
        Array of y-coordinates for the antenna array elements.
    f_type : str
        Type of the function used in processing (e.g., 'FFT', 'MUSIC').
    detection_range : int
        Range of angles to scan for detection, in degrees.
    """
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    center_freq = config.getint('daq', 'center_freq')
    sample_rate = config.getint('daq', 'sample_rate')
    num_samples = config.getint('pre_processing', 'cpi_size')
    antenna_distance = config.getfloat('variables','antenna_distance')

    x_str = config.get('variables', 'x')
    x = np.fromstring(x_str, sep=',')
    y_str = config.get('variables', 'y')
    y = np.fromstring(y_str, sep=',')
    
    f_type = config.get('variables', 'f_type')
    detection_range = config.getint('variables', 'detection_range')
    waraps = config.getboolean('variables', 'waraps')
    
    return center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type, detection_range, waraps

num_samples = 1024*64 # 1048576 # 
sample_rate = int(2.048e6)
center_freq = int(433.9e6)
gain = 40
# Linear Setupself.num_antennas = 0  
y = np.array([0, 0, 0, 0, 0])
x = np.array([-2, -1, 0, 1, 2])
antenna_distance = 0.175
detection_range = 180
waraps = False

kraken_config(center_freq, num_samples, sample_rate, antenna_distance, x, y, 'FIR', detection_range, waraps)