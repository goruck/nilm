# Electrical panel to Arduino hardware interface

This folder contains material related to the interface between the electrical panel and the Arduino hardware interface. See main README for a general overview and below for additional details.

## Design Considerations
[OpenEnergyMonitor](https://openenergymonitor.org/) in general was a great resource for this project and in particular the information found in [How to build an Arduino energy monitor - measuring mains voltage and current](https://learn.openenergymonitor.org/electricity-monitoring/ctac) was used to guilde the deisgn of this interface and the analog signal conditioning circuitry.

There are two main parts to this interface:
* An op-amp based circuit that buffers a bias line (AREF) from the Arduino that is used to set the midpoint of the voltages going into its analog-to-digital (ADC) converters.
* Voltage scaling and filtering RC circuits for each current phase and the mains voltage signals. Each signal is scaled to match the ADC range, level shifted to match the midpoint of the ADC range and bandlimited to prevent aliasing.

This subsystem was calibrated with an external power meter, yielding parameters which are used in the Arduino code that extracts the metrics used in downstream processing.

## Current Schematic

![Alt text](../img/analog-signal-conditioning.jpg?raw=true "Analog Signal Conditioning Schematic")

## References

1. [Atmel ATmega640/V-1280/V-1281/V-2560/V-2561/V Datasheet](https://ww1.microchip.com/downloads/en/devicedoc/atmel-2549-8-bit-avr-microcontroller-atmega640-1280-1281-2560-2561_datasheet.pdf)
2. [Arduino MEGA 2560 Documentation](https://store-usa.arduino.cc/products/arduino-mega-2560-rev3?selectedStore=us)