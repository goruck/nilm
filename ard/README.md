# ard

This folder contains the Arduino sketch that is responsible for sending voltage and current measurements to the Raspberry Pi based compute. See main README for a general overview.

The sketch uses the `emonLibCM` library to continuously measure in the background the voltage and current inputs and then then calculates a true average quantity for each and then informs the `ard.ino` sketch that the measurements are available and should be read and processed downstream.

The voltage, current and phase calibration values were arrived at per the steps described on the Open Energy Monitor website and the emonLibCM User Documentation. The values I used are shown below in the code snippet below taken from `ard.ino`.

```C++
EmonLibCM_SetADC_VChannel(0, 132.0);     // ADC Input channel, voltage calibration
EmonLibCM_SetADC_IChannel(1, 58.0, 4.6); // ADC Input channel, current calibration, phase calibration
EmonLibCM_SetADC_IChannel(2, 58.0, 4.2); // The current channels will be read in this order
```

## References

* [emonLibCM](https://github.com/openenergymonitor/EmonLibCM)

* [emonLibCM User Documentation](https://github.com/openenergymonitor/EmonLibCM/blob/master/emonLibCM%20User%20Doc.pdf)

* [Learn Open Energy Monitor](https://learn.openenergymonitor.org/)