# Electrical panel to Arduino hardware interface

This folder contains material related to the interface between the electrical panel and the Arduino hardware interface. See main README for a general overview and below for additional details.

## Design Considerations
[OpenEnergyMonitor](https://openenergymonitor.org/) in general was a great resource for this project and in particular the information found in [How to build an Arduino energy monitor - measuring mains voltage and current](https://learn.openenergymonitor.org/electricity-monitoring/ctac) was used to guide the design of this interface and the analog signal conditioning circuitry.

The current sensing circuitry includes a variable gain stage that is controlled by an automatic gain loop running on the Arduino. This improves signal-to-noise through the analog-to-digital conversion and improves downstream machine learning and training performance.

This subsystem was calibrated with an external power meter, yielding parameters which are used in the Arduino code that extracts the metrics used in downstream processing.

## Schematic

![Alt text](../img/pan-ard-inf-v1.1.jpg?raw=true "Analog Signal Conditioning Schematic")

## Layout

![Alt text](../img/pan-ard-inf-lo-v1.1.jpg?raw=true "Analog Signal Conditioning Schematic")

## References

1. [Atmel ATmega640/V-1280/V-1281/V-2560/V-2561/V Datasheet](https://ww1.microchip.com/downloads/en/devicedoc/atmel-2549-8-bit-avr-microcontroller-atmega640-1280-1281-2560-2561_datasheet.pdf)
2. [Arduino MEGA 2560 Documentation](https://store-usa.arduino.cc/products/arduino-mega-2560-rev3?selectedStore=us)
3. [Atmel AVR465: Single-Phase Power/Energy Meter with Tamper Detection](https://ww1.microchip.com/downloads/en/Appnotes/Atmel-2566-Single-Phase-Power-Energy-Meter-with-Tamper-Detection_Ap-Notes_AVR465.pdf)