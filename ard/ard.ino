/*

*/

#include <Arduino.h>
#include "emonLibCM.h"

void setup() 
{

  Serial.begin(9600);
  Serial.println("Set baud=115200");
  Serial.end();
  Serial.begin(115200);
  
  Serial.println("\nStaring monitoring..."); 

  EmonLibCM_SetADC_VChannel(0, 120.64);                    // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(1, 90.91, 4.6);                // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(2, 90.91, 4.2);                // The current channels will be read in this order

  //analogReference(INTERNAL1V1);
  EmonLibCM_setADC_VRef(INTERNAL1V1);                      // ADC Reference voltage (set to 1.1V)
  
  EmonLibCM_cycles_per_second(60);                         // mains frequency 50Hz, 60Hz
  
  EmonLibCM_min_startup_cycles(10);                        // number of cycles to let ADC run before starting first actual measurement

  EmonLibCM_Init();                                        // Start continuous monitoring.

}

void loop()
{

  if (EmonLibCM_Ready())
  {

    Serial.println(EmonLibCM_acPresent()?"AC present ":"AC missing ");
    delay(5);

    Serial.print(" V=");Serial.print(EmonLibCM_getVrms());
    Serial.print(" f=");Serial.println(EmonLibCM_getLineFrequency(),2);           

    for (byte ch=0; ch<4; ch++)
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

  }
}