import configparser
import numpy as np

def kraken_config(center_freq, num_samples, sample_rate, antenna_distance, 
                  x, y, array_type, f_type, waraps):
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
    waraps : bool
        True to stream to waraps False if not
    """
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    config.set('daq', 'center_freq', str(center_freq))
    config.set('daq', 'sample_rate', str(sample_rate))
    config.set('daq', 'gain', str(gain))
    config.set('daq', 'daq_buffer_size', str(num_samples))
    config.set('pre_processing', 'cpi_size', str(num_samples))
    config.set('pre_processing', 'fir_tap_size', str(51))
    config.set('pre_processing', 'fir_relative_bandwidth', str(0.1))
    config.set('variables', 'f_type', f_type)
    config.set('variables', 'x', ','.join(map(str, x)))
    config.set('variables', 'y', ','.join(map(str, y)))
    config.set('variables', 'antenna_distance', str(antenna_distance))
    config.set('variables', 'array_type', array_type)
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
    waraps : bool
        True to stream to waraps False if not
    """
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    center_freq = config.getint('daq', 'center_freq')
    sample_rate = config.getint('daq', 'sample_rate')
    num_samples = config.getint('pre_processing', 'cpi_size')
    antenna_distance = config.getfloat('variables','antenna_distance')
    array_type = config.get('variables', 'array_type')

    x_str = config.get('variables', 'x')
    x = np.fromstring(x_str, sep=',')
    y_str = config.get('variables', 'y')
    y = np.fromstring(y_str, sep=',')
    
    f_type = config.get('variables', 'f_type')
    waraps = config.getboolean('variables', 'waraps')
    
    return center_freq, num_samples, sample_rate, antenna_distance, x, y, array_type, f_type, waraps

num_samples = 1024*64 # 1048576 # 
sample_rate = int(2.048e6)
center_freq = int(433.9e6)
gain = 40
#Linear Setup
# y = np.array([0, 0, 0, 0, 0])
# x = np.array([-2, -1, 0, 1, 2])
# antenna_distance = 0.175
# Circular setup
ant0 = [1,    0]
ant1 = [0.3090,    0.9511]
ant2 = [-0.8090,    0.5878]
ant3 = [-0.8090,   -0.5878]
ant4 = [0.3090,   -0.9511]
y = np.array([ant0[1], ant1[1], ant2[1], ant3[1], ant4[1]])
x = np.array([ant0[0], ant1[0], ant2[0], ant3[0], ant4[0]])
antenna_distance =  0.35
antenna_distance = antenna_distance / 2.0 / np.sin(72.0*np.pi/180.0)

waraps = False

kraken_config(center_freq, num_samples, sample_rate, antenna_distance, x, y, 'UCA', 'FIR', waraps)