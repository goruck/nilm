"""
Captures Arduino data from the serial port to a CSV file.

Copyright (c) 2022 Lindo St. Angel
"""
import serial, time, csv, datetime
from progress.bar import Bar

CSV_FILE_NAME = '/mnt/usbstorage/nilm/garage/samples.csv'
#datetime, unix timestamp(UTC), rms voltage, {rms current, real power, apparent power} for each phase
CSV_ROW_NAMES = ['DT','TS','V','I1','W1','VA1','I2','W2','VA2']

SAMPLE_MAX = 5 * 10800 # 5 days @ 8 sec sampling period

def get_arduino_data(port) -> list:
    # Get bytes from Arduino.
    ser_bytes = port.readline()
    # Decode them as utf-8.
    decoded_bytes = ser_bytes.decode('utf-8').rstrip()
    # Split into individual elements and convert to float.
    # Elements are:
    #   rms voltage,
    #   rms current, real power, apparent power for phase 0,
    #   rms current, real power, apparent power for phase 1
    sample = [float(d) for d in decoded_bytes.split(',')]
    # Insert datetime.
    sample.insert(0, datetime.datetime.now())
    # Insert timestamp.
    sample.insert(1, round(time.time(), 2))
    #print(f'sample = {sample}')
    return sample

if __name__ == '__main__':
    print('Running. Press CTRL-C to exit.')

    with serial.Serial('/dev/ttyACM0', 115200, timeout=1) as ser:
        if ser.isOpen():
            time.sleep(1.0) # wait for port to be ready
            print(f'Connected to port {ser.port}.')
            with open(CSV_FILE_NAME, 'w') as csv_file:
                print(f'Opened {CSV_FILE_NAME} for writing {SAMPLE_MAX} samples.')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_ROW_NAMES)
                ser.reset_input_buffer()
                sample_num = 0
                try:
                    with Bar('Processing', max=SAMPLE_MAX) as bar:
                        while sample_num < SAMPLE_MAX:
                            if ser.in_waiting > 0:
                                # Get bytes from Arduino.
                                sample = get_arduino_data(ser)
                                # Write to csv file.
                                csv_writer.writerow(sample)
                                sample_num +=1
                                bar.next()
                except KeyboardInterrupt:
                    print('Got CTRL-C. Exiting.')
                print('Done.')