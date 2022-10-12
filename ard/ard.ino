//  Main NILM Arduino sketch.
//
//  Continually sends the following over the serial port:
//    RMS line voltage (one phase only)
//    RMS current (both phases)
//    Calculated real power (both phases)
//    Calculated apparent power (both phases).
//
//  Automatic gain control of the analog front end is done every sample period.
//
//  Copyright (c) 2022 Lindo St. Angel

#include <Arduino.h>
#include "emonLibCM.h"

// Define MCU GPIOs for gain control bits.
const byte kCt1G0 = 42; // ATMEGA2560 pin 42 (PL7) - GPIO42 - CT1 GAIN0
const byte kCt1G1 = 43; // ATMEGA2560 pin 41 (PL6) - GPIO43 - CT1 GAIN1
const byte kCt2G0 = 49; // ATMEGA2560 pin 35 (PL0) - GPIO49 - CT2 GAIN0
const byte kCt2G1 = 48; // ATMEGA2560 pin 36 (PL1) - GPIO48 - CT2 GAIN1

// Define ADC channels.
#define NUM_ADC_I_CHAN 2 // Number of ADC channels used for current sensing
#define V_CHAN  (byte) 0 // Voltage sense
#define I1_CHAN (byte) 1 // Current sense 1
#define I2_CHAN (byte) 2 // Current sense 2

// Set ADC full scale range to 2.56 Volts.
#define ADC_FS (double) 2.56

// Analog front end gains.
#define LOW_GAIN  (double) -1.3
#define MID_GAIN  (double) -10.2
#define HIGH_GAIN (double) -63.0

// Define current transformer max reading. 
#define CT_MAX (double) 100 // 100 Amps rms.

// Define calibration constants for Emon lib power calculations. 
#define V_AMP_CAL       (double) 180.0
#define I1_AMP_CAL_LOW  (double) 110.9
#define I1_AMP_CAL_MID  (double) 14.0
#define I1_AMP_CAL_HIGH (double) 2.29
#define I1_PH_CAL       (double) 4.6
#define I2_AMP_CAL_LOW  (double) 110.9 // was 144.0
#define I2_AMP_CAL_MID  (double) 14.0
#define I2_AMP_CAL_HIGH (double) 2.29
#define I2_PH_CAL       (double) 4.2

// Define current transformer gain control bit tuple
typedef struct gain_tuple
{
  int g1;
  int g0; // LSB
} GainTuple;

// Create a gain control tuple for each current transformer
GainTuple constexpr kCt1GainCtrl = {.g1 = kCt1G1, .g0 = kCt1G0};
GainTuple constexpr kCt2GainCtrl = {.g1 = kCt2G1, .g0 = kCt2G0};

// Analog front end gain settings.
typedef enum gain_settings
{
  kLowGain,
  kMidGain,
  kHighGain,
  kInvalidGain
} GainSettings;

// AGC inputs.
typedef struct agc_params
{
  byte channel;
  double v_rms;
  double i_rms;
  double apparent_power;
} AgcParams;

// Function prototypes.
GainSettings GetGainSettings(GainTuple gain_control);
double GetThreshold(GainSettings gain, double voltage);
void AutomaticGainControl(AgcParams *agcData);
inline void SetLowGain(GainTuple gain_control);
inline void SetMidGain(GainTuple gain_control);
inline void SetHighGain(GainTuple gain_control);
inline bool GetLowGain(GainTuple gain_control);
inline bool GetMidGain(GainTuple gain_control);
inline bool GetHighGain(GainTuple gain_control);

// Set analog gain to low.
inline void SetLowGain(GainTuple gain_control)
{
  digitalWrite(gain_control.g1, LOW);
  digitalWrite(gain_control.g0, LOW);
  return;
}

// Set analog gain to mid.
inline void SetMidGain(GainTuple gain_control)
{
  digitalWrite(gain_control.g1, LOW);
  digitalWrite(gain_control.g0, HIGH);
  return;
}

// Set analog gain to high.
inline void SetHighGain(GainTuple gain_control)
{
  digitalWrite(gain_control.g1, HIGH);
  digitalWrite(gain_control.g0, HIGH);
  return;
}

// Return true if analog gain is set to low.
inline bool GetLowGain(GainTuple gain_control)
{
  return ((digitalRead(gain_control.g1) == LOW) && (digitalRead(gain_control.g0) == LOW));
}

// Return true if analog gain is set to mid.
inline bool GetMidGain(GainTuple gain_control)
{
  return ((digitalRead(gain_control.g1) == LOW) && (digitalRead(gain_control.g0) == HIGH));
}

// Return true if analog gain is set to high.
inline bool GetHighGain(GainTuple gain_control)
{
  return ((digitalRead(gain_control.g1) == HIGH) && (digitalRead(gain_control.g0) == HIGH));
}

// Calculate threshold for switching analog front end gain.
double GetThreshold(GainSettings gain, double voltage)
{
  double threshold = CT_MAX * voltage; // Max Apparent Power = max mains Irms * mains Vrms

  // Adjust for analog front end gain. 
  switch (gain)
  {
  case kLowGain:
    threshold /= -LOW_GAIN;
    break;
  case kMidGain:
    threshold /= -MID_GAIN;
    break;
  case kHighGain:
    threshold /= -HIGH_GAIN;
    break;
  default:
    break;
  }

  return threshold;
}

// Get current analog front end gain setings.
GainSettings GetGainSettings(GainTuple gain_control)
{
  if (GetLowGain(gain_control))
  {
    return kLowGain;
  }
  else if (GetMidGain(gain_control))
  {
    return kMidGain;
  }
  else if (GetHighGain(gain_control))
  {
    return kHighGain;
  }
  else
  {
    return kInvalidGain;
  }
}

// Automatic gain control of analog front end.
//
// Apparent power is compared against a threshold to change gain.
// The threshold is set by the maximum apparent power of the current
// transformer and mains voltage. Before a new gain is applied, a check
// is done to ensure the ADC input is not overloaded. This would cause
// incorrect readings, potentially preventing agc loop closure. The current
// channels are recalibrated each time the gain is adjusted. 
void AutomaticGainControl(AgcParams *AgcData)
{
  double v_adc;
  double amp_cal_low, amp_cal_mid, amp_cal_high, phase_cal;
  double threshold;
  GainTuple gain_control;
  GainSettings gain;
  byte adc_input = AgcData->channel + 1;

  // ADC full scale RMS voltage, 0.905 Vrms @2.56V FS.
  double constexpr kAdcFsVrms (ADC_FS / 2.828427125);

  // Threshold scale factors for switching analog gain settings.
  double const kUpperThreshold = 0.95;
  double const kLowerThreshold = 0.85;

  if (adc_input == I1_CHAN)
  {
    amp_cal_low = I1_AMP_CAL_LOW;
    amp_cal_mid = I1_AMP_CAL_MID;
    amp_cal_high = I1_AMP_CAL_HIGH;
    phase_cal = I1_PH_CAL;
    gain_control = kCt1GainCtrl;
  }
  else if (adc_input == I2_CHAN)
  {
    amp_cal_low = I2_AMP_CAL_LOW;
    amp_cal_mid = I2_AMP_CAL_MID;
    amp_cal_high = I2_AMP_CAL_HIGH;
    phase_cal = I2_PH_CAL;
    gain_control = kCt2GainCtrl;
  }
  else
  {
    /* invalid */
  }

  gain = GetGainSettings(gain_control);
  threshold = GetThreshold(gain, AgcData->v_rms);

  if (AgcData->apparent_power > threshold * kUpperThreshold)
  {
    // Decrement gains.
    if (gain == kHighGain)
    {
      SetMidGain(gain_control);
      EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_mid, phase_cal);
    }
    else if (gain == kMidGain)
    {
      SetLowGain(gain_control);
      EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_low, phase_cal);
    }
    else
    {/* already at low gain, do nothing */}
  }
  else if (AgcData->apparent_power < threshold * kLowerThreshold)
  {
    // Increment gains.
    if (gain == kLowGain)
    {
      v_adc = AgcData->i_rms / amp_cal_low;   // approx rms voltage at ADC input
      if (v_adc * -MID_GAIN < kAdcFsVrms)  // validity check
      {
        SetMidGain(gain_control);
        EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_mid, phase_cal);
      }
    }
    else if (gain == kMidGain)
    {
      v_adc = AgcData->i_rms / amp_cal_mid;   // approx rms voltage at ADC input
      if (v_adc * -HIGH_GAIN < kAdcFsVrms) // validity check
      {
        SetHighGain(gain_control);
        EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_high, phase_cal);
      }
    }
    else
    {/* already at high gain, do nothing */}
  }
}

void setup()
{
  // Set GPIOs as outputs.
  pinMode(kCt1G0, OUTPUT);
  pinMode(kCt1G1, OUTPUT);
  pinMode(kCt2G0, OUTPUT);
  pinMode(kCt2G1, OUTPUT);

  // Set initial analog front end gains.
  SetLowGain(kCt1GainCtrl);
  SetLowGain(kCt2GainCtrl);

  Serial.begin(115200);

  EmonLibCM_SetADC_VChannel(V_CHAN, V_AMP_CAL);                   // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(I1_CHAN, I1_AMP_CAL_LOW, I1_PH_CAL);  // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(I2_CHAN, I2_AMP_CAL_LOW, I2_PH_CAL);  // The current channels will be read in this order

  EmonLibCM_setADC_VRef(INTERNAL2V56);                            // ADC Reference voltage (set to 2.56V)
  EmonLibCM_ADCCal(ADC_FS);                                       // ADC Cal voltage (set to 2.56V)
  
  EmonLibCM_cycles_per_second(60);                                // Line frequency (set to 60 Hz)

  EmonLibCM_datalog_period(8.0);                                  // Interval over which stats are reported (in secs)

  EmonLibCM_Init();                                               // Start continuous monitoring
}

void loop()
{
  AgcParams AgcData;
  AgcParams *AgcDataPtr = &AgcData;
  double i_rms, v_rms;
  double apparent_power;

  if (EmonLibCM_Ready())
  {
    v_rms = EmonLibCM_getVrms();
    Serial.print(v_rms);Serial.print(",");

    for (size_t i = 0; i < NUM_ADC_I_CHAN; i++)
    {
      i_rms = EmonLibCM_getIrms(i);
      apparent_power = EmonLibCM_getApparentPower(i);
     
      Serial.print(i_rms,3);Serial.print(",");
      Serial.print(EmonLibCM_getRealPower(i));Serial.print(",");
      Serial.print(apparent_power);
      if (i < NUM_ADC_I_CHAN - 1) Serial.print(",");

      // Run automatic gain control of analog front end.
      AgcDataPtr->channel = i;
      AgcDataPtr->v_rms = v_rms;
      AgcDataPtr->i_rms = i_rms;
      AgcDataPtr->apparent_power = apparent_power;
      AutomaticGainControl(AgcDataPtr);
    }

    Serial.println(); // Outputs one sample.

    delay(50);
  }
}