# test serial link to arduino
import serial, time, csv

CSV_FILE_NAME = 'samples.csv'
#
CSV_ROW_NAMES = ['S','V','F','I1','W1','VA1','Wh1','PF1','I2','W2','VA2','Wh2','PF2']

if __name__ == '__main__':
    print('Running. Press CTRL-C to exit.')

    with serial.Serial('/dev/ttyACM0', 115200, timeout = 1) as ser:
        if ser.isOpen():
            time.sleep(1.0) # wait for port to be ready
            print(f'Connected to port {ser.port}.')
            with open(CSV_FILE_NAME, 'w') as csv_file:
                print(f'Opened {CSV_FILE_NAME} for writing.')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_ROW_NAMES)
                sample_num = 0
                ser.reset_input_buffer()
                try:
                    while True:
                        if ser.in_waiting > 0:
                            ser_bytes = ser.readlines()
                            #print(ser_bytes)
                            decoded_bytes = [float(b.decode('utf-8').rstrip()) for b in ser_bytes]
                            decoded_bytes.insert(0, sample_num)
                            print(decoded_bytes)
                            csv_writer.writerow(decoded_bytes)
                            sample_num +=1
                except KeyboardInterrupt:
                    print('Got CTRL-C. Exiting.')