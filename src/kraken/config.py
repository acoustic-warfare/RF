import configparser
import numpy as np

def write_list_to_config():
    pass

def kraken_config(center_freq, sample_rate, gain, num_samples, antenna_distance, x, y, f_type, detection_range):
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

    with open('heimdall_daq_fw/Firmware/daq_chain_config.ini', 'w') as configfile:
        config.write(configfile)

def read_kraken_config():
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
    f_type = config.get('variables', 'detection_range')
    
    return center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type, detection_range

num_samples = 1024*64 # 1048576 # 
sample_rate = int(2.048e6)
center_freq = int(434.4e6)
gain = 40
# Linear Setup
y = np.array([0, 0, 0, 0, 0])
x = np.array([0, 1, 2, 3, 4])
antenna_distance = 0.175
detection_range = 180

kraken_config(center_freq, sample_rate, gain, num_samples, antenna_distance, x, y, 'FIR', detection_range)