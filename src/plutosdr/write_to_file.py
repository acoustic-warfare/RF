import datetime  
import adi  
import os 
import time 
"""Create and configure SDR"""
sdr = adi.ad9361(uri='ip:192.168.2.1')  
samp_rate = 2e6  
NumSamples = 2**12  # Number of samples to capture
rx_lo = 433e6  
rx_mode = "manual"  # Gain control mode, can be "manual" or "slow_attack"
rx_gain0 = 40  
fc0 = int(200e3)  # Cutoff frequency in Hz, used to set the bandwidth

# Configure the SDR settings
sdr.rx_enabled_channels = [0]  # Enable only channel 0 for receiving
sdr.sample_rate = int(samp_rate)  
sdr.rx_rf_bandwidth = int(fc0*3)  
sdr.rx_lo = int(rx_lo) 
sdr.gain_control_mode = rx_mode  
sdr.rx_hardwaregain_chan0 = int(rx_gain0) 
sdr.rx_buffer_size = int(NumSamples)  
sdr._rxadc.set_kernel_buffers_count(1)  # Set kernel buffers to 1 to avoid stale data

# Function to create and store images in a specified folder (implementation placeholder)
def create_images(folder_name):
    pass

# Function to create a new folder for recordings
def create_folder():
    absolute_path = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
    
    # Define the relative path for the recordings
    relative_path = "recordings/"
    
    # Create a unique folder name using the current date and time
    folder_name = 'recordings_' + time.strftime("%Y%m%d_%H%M%S")
    
    # Combine the absolute path with the folder name to get the full path
    full_path = os.path.join(absolute_path, relative_path, folder_name)
    
    # Create the directory for storing recordings
    os.makedirs(full_path)  # Ensure all intermediate-level directories are created
    print(f"Full path created: {full_path}")  # Print the full path for debugging
    return full_path

# Function to create a new file with a unique name based on the current date and time
def create_new_file(full_path):
    # Create the base name for the file
    basename = os.path.join(full_path, "myfile")
    
    # Generate a unique suffix using the current date and time
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Combine the base name and suffix to get the full filename
    filename = "_".join([basename, suffix])
    
    # Create a new file with the generated filename
    f = open(filename, "x")
    return filename

# Function to save data to a specified file
def save_data(filename, input):
    # Open the file in append mode
    f = open(filename, "a")
    
    # Write the input data to the file followed by a newline
    f.write(input + "\n")
    
    # Close the file
    f.close()

# Creates new folder and return the path
full_path = create_folder()
# Create a new file and get its filename
filename = create_new_file(full_path)

# Infinite loop to continuously save data from SDR to the file
while(True):
    
    data = sdr.rx()
    input = ' '.join(str(e) for e in data)
    save_data(filename, input)
