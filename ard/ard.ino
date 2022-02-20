/*

*/

#include <Arduino.h>
#include "emonLibCM.h"

void setup() 
{

  //Serial.begin(9600);
  //Serial.println("Set baud=115200");
  //Serial.end();
  Serial.begin(115200);
  
  Serial.println("\nStaring monitoring...");

  EmonLibCM_SetADC_VChannel(0, 132.0);                    // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(1, 58.0, 4.6);                // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(2, 58.0, 4.2);                // The current channels will be read in this order

  EmonLibCM_setADC_VRef(INTERNAL2V56);                    // ADC Reference voltage (set to 2.56V)
  EmonLibCM_ADCCal(2.56);                                 // ADC Cal voltage (set to 2.56V)
  
  EmonLibCM_cycles_per_second(60);                        // Line frequency is 60Hz
  
  EmonLibCM_min_startup_cycles(10);                       // Number of cycles to let ADC run before starting first actual measurement

  EmonLibCM_datalog_period(10.0);                         // Set interval over which stats are reports

  EmonLibCM_Init();                                       // Start continuous monitoring

}

void loop()
{

  if (EmonLibCM_Ready())
  {
    
    Serial.println(EmonLibCM_getVrms());
    Serial.println(EmonLibCM_getLineFrequency(),2);
    for (byte ch=0; ch<2; ch++)
    {
        Serial.println(EmonLibCM_getIrms(ch),3);
        Serial.println(EmonLibCM_getRealPower(ch));
        Serial.println(EmonLibCM_getApparentPower(ch));
        Serial.println(EmonLibCM_getWattHour(ch));
        Serial.println(EmonLibCM_getPF(ch),4);
        delay(10);
    }

    /*
    Serial.println(EmonLibCM_acPresent()?"AC present ":"AC missing ");
    delay(5);

    Serial.print(" V=");Serial.print(EmonLibCM_getVrms());
    Serial.print(" f=");Serial.println(EmonLibCM_getLineFrequency(),2);

    for (byte ch=0; ch<2; ch++)
    {
        Serial.print("Ch ");Serial.print(ch+1);
        Serial.print(" I=");Serial.print(EmonLibCM_getIrms(ch),3);
        Serial.print(" W=");Serial.print(EmonLibCM_getRealPower(ch));
        Serial.print(" VA=");Serial.print(EmonLibCM_getApparentPower(ch));
        Serial.print(" Wh=");Serial.print(EmonLibCM_getWattHour(ch));
        Serial.print(" pf=");Serial.print(EmonLibCM_getPF(ch),4);
        Serial.println();
        delay(10);
    }
    */
  }
}