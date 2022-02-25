/*

Continually sends the following over the serial port:
  RMS line voltage (one phase only)
  RMS current (both phases)
  Calculated real power (both phases)
  Calcuated apparent power (both phases).

Copyright (c) 2022 Lindo St. Angel

*/

#include <Arduino.h>
#include "emonLibCM.h"

void setup() 
{

  Serial.begin(115200);

  EmonLibCM_SetADC_VChannel(0, 132.0);         // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(1, 58.0, 4.6);     // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(2, 58.0, 4.2);     // The current channels will be read in this order

  EmonLibCM_setADC_VRef(INTERNAL2V56);         // ADC Reference voltage (set to 2.56V)
  EmonLibCM_ADCCal(2.56);                      // ADC Cal voltage (set to 2.56V)
  
  EmonLibCM_cycles_per_second(60);             // Line frequency (set to 60 Hz)

  EmonLibCM_datalog_period(0.1);               // Interval over which stats are reported (in secs)

  EmonLibCM_Init();                            // Start continuous monitoring

}

void loop()
{

  if (EmonLibCM_Ready())
  {
    
    Serial.print(EmonLibCM_getVrms());Serial.print(",");

    Serial.print(EmonLibCM_getIrms(0),3);Serial.print(",");
    Serial.print(EmonLibCM_getRealPower(0));Serial.print(",");
    Serial.print(EmonLibCM_getApparentPower(0));Serial.print(",");
    Serial.print(EmonLibCM_getIrms(1),3);Serial.print(",");
    Serial.print(EmonLibCM_getRealPower(1));Serial.print(",");
    Serial.print(EmonLibCM_getApparentPower(1));

    Serial.println(); // Completes one sample.

  }

}