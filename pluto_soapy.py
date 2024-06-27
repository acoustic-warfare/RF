# Should work with PyQt5 / PySide2 / PySide6 as well
from PyQt6 import QtWidgets
import pyqtgraph as pg
import numpy as np
import numpy  # use numpy for buffers
import SoapySDR as sp

sp.loadModule(
    "/usr/local/lib/SoapySDR/modules0.8-3/libPlutSDRSupport.so")


def init():
    # enumerate devices
    results = sp.Device.enumerate()
    for result in results:
        print(result)

    # create device instance
    # args can be user defined or from the enumeration result
    args = dict(driver="rtlsdr")
    sdr = sp.Device(args)

    # query device info
    print(sdr.listAntennas(sp.SOAPY_SDR_RX, 0))
    print(sdr.listGains(sp.SOAPY_SDR_RX, 0))
    freqs = sdr.getFrequencyRange(sp.SOAPY_SDR_RX, 0)
    for freqRange in freqs:
        print(freqRange)

    # apply settings
    sdr.setSampleRate(sp.SOAPY_SDR_RX, 0, 1e6)
    sdr.setFrequency(sp.SOAPY_SDR_RX, 0, 912.3e6)

    # setup a stream (complex floats)
    rxStream = sdr.setupStream(sp.SOAPY_SDR_RX, sp.SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)  # start streaming

    # create a re-usable buffer for rx samples
    buff = numpy.array([0]*1024, numpy.complex64)

    # receive some samples
    for i in range(10):
        sr = sdr.readStream(rxStream, [buff], len(buff))
        print(sr.ret)  # num samples or error code
        print(sr.flags)  # flags set by receive operation
        print(sr.timeNs)  # timestamp for receive buffer

    # shutdown the stream
    sdr.deactivateStream(rxStream)  # stop streaming
    sdr.closeStream(rxStream)


def update():
    pass


def visualize():
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication([])

    # Define a top-level widget to hold everything
    w = QtWidgets.QWidget()
    w.setWindowTitle('PyQtGraph example')

    # Create some widgets to be placed inside
    btn = QtWidgets.QPushButton('press me')
    text = QtWidgets.QLineEdit('enter text')
    listw = QtWidgets.QListWidget()
    plot = pg.PlotWidget()

    # Create a grid layout to manage the widgets size and position
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)

    # Add widgets to the layout in their proper positions
    layout.addWidget(btn, 0, 0)  # button goes in upper-left
    layout.addWidget(text, 1, 0)  # text edit goes in middle-left
    layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
    # plot goes on right side, spanning 3 rows
    layout.addWidget(plot, 0, 1, 3, 1)
    # Display the widget as a new window
    w.show()

    # Start the Qt event loop
    app.exec()  # or app.exec_() for PyQt5 / PySide2


if __name__ == "__main__":
    visualize()
    input("")
