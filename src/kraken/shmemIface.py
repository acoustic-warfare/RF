"""
    HeIMDALL DAQ Firmware
    Python based shared memory interface implementations

    Author: Tamás Pető
    License: GNU GPL V3

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import os
import time
from multiprocessing import shared_memory
from struct import pack, unpack

import numpy as np

A_BUFF_READY = 1
B_BUFF_READY = 2
INIT_READY = 10
TERMINATE = 255
SLEEP_TIME_BETWEEN_READ_ATTEMPTS = 0.01  # seconds

class inShmemIface:
    def __init__(self, shmem_name, ctr_fifo_path="_data_control/", read_timeout=None):
        self.init_ok = True
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.num_read_attempts = int(read_timeout / SLEEP_TIME_BETWEEN_READ_ATTEMPTS) if read_timeout else 1
        self.shmem_name = shmem_name

        self.memories = []
        self.buffers = []
        fw_fifo_flags = os.O_RDONLY | os.O_NONBLOCK if read_timeout else os.O_RDONLY

        try:
            self.fw_ctr_fifo = os.open(ctr_fifo_path + "fw_" + shmem_name, fw_fifo_flags)
            self.bw_ctr_fifo = os.open(ctr_fifo_path + "bw_" + shmem_name, os.O_WRONLY)
        except OSError as err:
            self.logger.critical("OS error: {0}".format(err))
            self.logger.critical("Failed to open control fifos")
            self.bw_ctr_fifo = None
            self.fw_ctr_fifo = None
            self.init_ok = False
        if self.fw_ctr_fifo is not None:
            signal = self.read_fw_ctr_fifo()
            if signal and signal == INIT_READY:
                self.memories.append(shared_memory.SharedMemory(name=shmem_name + "_A"))
                self.memories.append(shared_memory.SharedMemory(name=shmem_name + "_B"))
                self.buffers.append(
                    np.ndarray(
                        (self.memories[0].size,),
                        dtype=np.uint8,
                        buffer=self.memories[0].buf,
                    )
                )
                self.buffers.append(
                    np.ndarray(
                        (self.memories[1].size,),
                        dtype=np.uint8,
                        buffer=self.memories[1].buf,
                    )
                )
            else:
                self.init_ok = False

    def send_ctr_buff_ready(self, active_buffer_index):
        if active_buffer_index == 0:
            os.write(self.bw_ctr_fifo, pack("B", A_BUFF_READY))
        elif active_buffer_index == 1:
            os.write(self.bw_ctr_fifo, pack("B", B_BUFF_READY))

    def destory_sm_buffer(self):
        for memory in self.memories:
            memory.close()

        if self.fw_ctr_fifo is not None:
            os.close(self.fw_ctr_fifo)

        if self.bw_ctr_fifo is not None:
            os.close(self.bw_ctr_fifo)

    def wait_buff_free(self):
        signal = self.read_fw_ctr_fifo()
        if not signal:
            return -1
        elif signal == A_BUFF_READY:
            return 0
        elif signal == B_BUFF_READY:
            return 1
        elif signal == TERMINATE:
            return TERMINATE
        else:
            return -1

    def read_fw_ctr_fifo(self):
        for _ in range(self.num_read_attempts):
            try:
                signal = unpack("B", os.read(self.fw_ctr_fifo, 1))[0]
            except BlockingIOError:
                time.sleep(SLEEP_TIME_BETWEEN_READ_ATTEMPTS)
            else:
                return signal
        return None
