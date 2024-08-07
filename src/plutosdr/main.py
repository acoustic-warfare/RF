import argparse
import time
import traceback
from agent import Agent
from data.config import AgentConfig, MqttConfig
from QtSpectrogram import LiveSpectrogram, RealTimePlotter
import adi
import sys
from PyQt5 import QtWidgets, QtCore
import time
from rtmp_streamer import PyRtmpStreamer



def main():
    # Creates the parameters for the PlutoSDR and LiveSpectrogram
    samp_rate = 30e6
    num_samples = 65536
    center_freq = 105e6  # Center Frequency
    rx_mode = "manual"
    rx_gain = 70
    bandwidth = int(30e6)
    waraps = True

    # Creates the PlutoSDR and sets the properties
    sdr = adi.ad9361(uri='ip:192.168.2.1')
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = bandwidth
    sdr.rx_lo = int(center_freq)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(num_samples)

    # Number of FFTs/segments that the window will contain
    frames = 120

    stream_name = "rtmp://ome.waraps.org/app/plutosdr"
    
    streamer = PyRtmpStreamer(1280, 720, stream_name)

    liveSpectrogram = LiveSpectrogram(frames, num_samples, samp_rate, sdr, center_freq, bandwidth)
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter(liveSpectrogram, waraps, streamer)

    # Puts and runs the LiveSpectrogram in another thread
    liveSpectrogram.data_ready.connect(plotter.update_plot)
    thread = QtCore.QThread()
    liveSpectrogram.moveToThread(thread)
    thread.started.connect(liveSpectrogram.start)
    thread.start()




    if waraps:


        def start_agent():
            try:
                parser = argparse.ArgumentParser()
                parser.add_argument("-n", "--name", help="Give the agents a name")
                parser.add_argument("-u", "--units", help="Input how many agents to create")
                args = parser.parse_args()

                if args.name:
                    AgentConfig.NAME = args.name
                    agent_topic: str = f"waraps/unit/{AgentConfig.DOMAIN}/{AgentConfig.SIM_REAL}/{AgentConfig.NAME}"
                    MqttConfig.WARAPS_TOPIC_BASE = agent_topic
                    MqttConfig.WARAPS_LISTEN_TOPIC = f"{agent_topic}/exec/command"

                my_agent = Agent(liveSpectrogram, streamer)

                # Listen for incoming messages and send agent data 'rate' times per second
                rate: float = 1.0 / my_agent.logic.rate
                while True:
                    my_agent.check_task()
                    my_agent.send_heartbeat()
                    my_agent.send_sensor_info()
                    my_agent.send_position()
                    my_agent.send_speed()
                    my_agent.send_course()
                    my_agent.send_heading()
                    my_agent.send_direct_execution_info()

                    time.sleep(rate)

            except Exception:
                print(traceback.format_exc())

        agent_thread = QtCore.QThread()
        agent_thread.run = start_agent
        agent_thread.start()


    
    plotter.show()


    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
