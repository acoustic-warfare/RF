import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers

#enumerate devices
results = SoapySDR.Device.enumerate()# {"driver": "hackrf"})
for result in results: print(result)

exit(0)

#create device instance
#args can be user defined or from the enumeration result
args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)

#query device info
print(sdr.listAntennas(SOAPY_SDR_RX, 0))
print(sdr.listGains(SOAPY_SDR_RX, 0))
freqs = sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
for freqRange in freqs: print(freqRange)

#apply settings
sdr.setSampleRate(SOAPY_SDR_TX, 0, 2e6)
sdr.setFrequency(SOAPY_SDR_TX, 0, 433e6)

#setup a stream (complex floats)
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
sdr.activateStream(txStream) #start streaming

# '''Program Tx and Send Data'''
fs = int(sdr.getSampleRate(SOAPY_SDR_TX, 0))
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * 200e3) * 2 ** 14
q0 = np.sin(2 * np.pi * t * 200e3) * 2 ** 14
iq0 = np.array(i0 + 1j * q0, np.complex64)

#create a re-usable buffer for rx samples
#buff = numpy.array([0]*1024, numpy.complex64)

#receive some samples
for i in range(2**12):
    sr = sdr.writeStream(txStream, [iq0], len(iq0))
    print(sr.ret) #num samples or error code
    print(sr.flags) #flags set by receive operation
    print(sr.timeNs) #timestamp for receive buffer

#shutdown the stream
sdr.deactivateStream(txStream) #stop streaming
sdr.closeStream(txStream)


