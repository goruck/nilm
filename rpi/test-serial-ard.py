# test serial link to arduino
import serial, time, csv

CSV_FILE_NAME = 'samples.csv'

if __name__ == '__main__':
    print('Running. Press CTRL-C to exit.')

    with serial.Serial('/dev/ttyACM0', 115200, timeout = 1) as ser:
        if ser.isOpen():
            time.sleep(0.5) # wait for port to be ready
            print(f'Connected to port {ser.port}.')
            with open(CSV_FILE_NAME, 'w') as csv_file:
                print(f'Opened {CSV_FILE_NAME} for writing.')
                csv_writer = csv.writer(csv_file)
                sample_num = 0
                ser.reset_input_buffer()
                try:
                    while True:
                        if ser.in_waiting > 0:
                            ser_bytes = ser.readline()
                            print(ser_bytes)
                            #decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode('utf-8'))
                            #print(decoded_bytes)
                            #csv_writer.writerow([sample_num, decoded_bytes])
                            #sample_num +=1
                except KeyboardInterrupt:
                    print('Got CTRL-C. Exiting.')