"""
Captures Arduino data from the serial port to std output.

Copyright (c) 2022~2024 Lindo St. Angel
"""
import time
import sys

import serial

def get_arduino_data(port) -> list:
    """ Get bytes from Arduino.

    Elements are:
       rms voltage,
       AC voltage present flag,
       rms current, real power, apparent power for phase 0,
       agc gain state for phase 0,
       rms current, real power, apparent power for phase 1,
       agc gain state for phase 1.
    """
    ser_bytes = port.readline()
    # Decode them as utf-8.
    decoded_bytes = ser_bytes.decode('utf-8').rstrip()
    return list(decoded_bytes.split(','))

if __name__ == '__main__':
    print('Running. Press CTRL-C to exit.')

    with serial.Serial('/dev/ttyACM0', 115200, timeout=1) as ser:
        if ser.isOpen():
            time.sleep(1.0) # wait for port to be ready
            print(f'Connected to port {ser.port}.')
            ser.reset_input_buffer()
            while True:
                try:
                    if ser.in_waiting > 0:
                        # Get bytes from Arduino.
                        sample = get_arduino_data(ser)
                        print(sample)
                except KeyboardInterrupt:
                    print('Got CTRL-C. Exiting.')
                    sys.exit(0)
                time.sleep(0.01)
