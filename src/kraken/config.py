import configparser
import numpy as np

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
    config.set('variables', 'x', ','.join(map(str, x)))
    config.set('variables', 'y', ','.join(map(str, y)))
    config.set('variables', 'detection_range', str(detection_range))

    with open('heimdall_daq_fw/Firmware/daq_chain_config.ini', 'w') as configfile:
        config.write(configfile)

def read_kraken_config():
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    center_freq = config.getint('daq', 'center_freq')
    sample_rate = config.getint('daq', 'sample_rate')
    num_samples = config.getint('pre_processing', 'cpi_size')
    antenna_distance = config.getfloat('variables','antenna_distance')
    detection_range = config.getint('variables', 'detection_range')

    x_str = config.get('variables', 'x')
    x = np.fromstring(x_str, sep=',')
    y_str = config.get('variables', 'y')
    y = np.fromstring(y_str, sep=',')
    
    f_type = config.get('variables', 'f_type')
    
    return center_freq, num_samples, sample_rate, antenna_distance, x, y, f_type, detection_range

antenna_distance = 0.175
# ant0 = [1,    0]
# ant1 = [0.3090,    0.9511]
# ant2 = [-0.8090,    0.5878]
# ant3 = [-0.8090,   -0.5878]
# ant4 = [0.3090,   -0.9511]
# y = np.array([ant0[1], ant1[1], ant2[1], ant3[1], ant4[1]])
# x = np.array([ant0[0], ant1[0], ant2[0], ant3[0], ant4[0]])
# antenna_distance = antenna_distance / 2.0 / np.sin(36.0*np.pi/180.0)

num_samples = 1024*64 # 1048576 # 
sample_rate = int(2.048e6)
center_freq = int(433.9e6)
gain = 40
# Linear Setup
y = np.array([0, 1, 0, 1, 0])
x = np.array([0, 1, 2, 3, 4])
detection_range = 360


kraken_config(center_freq, sample_rate, gain, num_samples, antenna_distance, x, y, 'FIR', detection_range)