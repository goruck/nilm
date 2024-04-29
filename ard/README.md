# ard

This folder contains the Arduino sketch that is responsible for sending voltage and current measurements to the Raspberry Pi based compute. See main README for a general overview.

The sketch uses the `EmonLibCM` library to continuously measure in the background the voltage and current inputs and then then calculates a true average quantity for each and then informs the `ard.ino` sketch that the measurements are available and should be read and processed downstream. Automatic gain control of the analog front end is done every sample period.

The voltage, current and phase calibration values were arrived at per the steps described on the Open Energy Monitor website and the emonLibCM User Documentation.

Follow the instructions found at `EmonLibCM`'s github to install v2.4.0 in your Arduino library folder. You will also need to install the OneWire library which you can do using the Arduino Library manager or manually install it via a `git clone` or a Zip file to the Arduino library folder. The Arduino is usually found at `~/Arduino`.

## References

* [EmonLibCM](https://github.com/openenergymonitor/EmonLibCM)

* [EmonLibCM User Documentation](https://github.com/openenergymonitor/EmonLibCM/blob/master/emonLibCM%20User%20Doc.pdf)

* [Open Energy Monitor](https://learn.openenergymonitor.org/)