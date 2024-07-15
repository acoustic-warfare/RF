import configparser

def kraken_config(center_freq, sample_rate, gain, num_samples):
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    config.set('daq', 'center_freq', str(center_freq))
    config.set('daq', 'sample_rate', str(sample_rate))
    config.set('daq', 'gain', str(gain))
    config.set('daq', 'daq_buffer_size', str(num_samples))
    config.set('pre_processing', 'cpi_size', str(num_samples))

    with open('heimdall_daq_fw/Firmware/daq_chain_config.ini', 'w') as configfile:
        config.write(configfile)

def read_kraken_config():
    config = configparser.ConfigParser()
    config.read('heimdall_daq_fw/Firmware/daq_chain_config.ini')

    center_freq = config.getint('daq', 'center_freq')
    sample_rate = config.getint('daq', 'sample_rate')
    #gain = config.getint('daq', 'gain')
    num_samples = config.getint('pre_processing', 'cpi_size')
    
    return center_freq, num_samples, sample_rate

num_samples = 1024*64 # 1048576 # 
sample_rate = int(2.048e6)
center_freq = int(433.9e6)
gain = 40

kraken_config(center_freq, sample_rate, gain, num_samples)