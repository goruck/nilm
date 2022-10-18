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
constexpr byte kCt1G0 = 42; // ATMEGA2560 pin 42 (PL7) - GPIO42 - CT1 GAIN0
constexpr byte kCt1G1 = 43; // ATMEGA2560 pin 41 (PL6) - GPIO43 - CT1 GAIN1
constexpr byte kCt2G0 = 49; // ATMEGA2560 pin 35 (PL0) - GPIO49 - CT2 GAIN0
constexpr byte kCt2G1 = 48; // ATMEGA2560 pin 36 (PL1) - GPIO48 - CT2 GAIN1

// ADC channel to signal sense mapping.
constexpr byte kVChan = 0;       // Voltage sense
constexpr byte kNumAdcIChan = 2; // #ADC channels for current sensing
constexpr byte kI1Chan = 1;      // Current sense 1
constexpr byte kI2Chan = 2;      // Current sense 2

// Set ADC full scale range to 2.56 Volts.
constexpr double kAdcFs = 2.56;

// Analog front end gains.
constexpr double kLowGain = -1.3;
constexpr double kMidGain = -10.2;
constexpr double kHighGain = -63.0;

// Define current transformer max reading. 
constexpr double kCtMax = 100.0; // 100 Amps rms.

// Define calibration constants for Emon lib power calculations. 
constexpr double kVAmpCal = 198.0;
constexpr double kI1AmpCalLow = 110.9;
constexpr double kI1AmpCalMid = 14.0;
constexpr double kI1AmpCalHigh = 2.29;
constexpr double kI1PhCal = 4.6;
constexpr double kI2AmpCalLow = 110.9; // was 144.0
constexpr double kI2AmpCalMid = 14.0;
constexpr double kI2AmpCalHigh = 2.29;
constexpr double kI2PhCal = 4.2;

// Define current transformer gain control bit tuple
typedef struct gain_tuple
{
  int g1;
  int g0; // LSB
} GainTuple;

// Create a gain control tuple for each current transformer
GainTuple constexpr kCt1GainCtrl = {.g1 = kCt1G1, .g0 = kCt1G0};
GainTuple constexpr kCt2GainCtrl = {.g1 = kCt2G1, .g0 = kCt2G0};

// Analog front end gain states.
typedef enum gain_state
{
  kLow,
  kMid,
  kHigh,
  kInvalid
} GainState;

// AGC inputs.
typedef struct agc_params
{
  byte channel;
  double v_rms;
  double i_rms;
  double apparent_power;
} AgcParams;

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
  return (digitalRead(gain_control.g1) == LOW) && (digitalRead(gain_control.g0) == LOW);
}

// Return true if analog gain is set to mid.
inline bool GetMidGain(GainTuple gain_control)
{
  return (digitalRead(gain_control.g1) == LOW) && (digitalRead(gain_control.g0) == HIGH);
}

// Return true if analog gain is set to high.
inline bool GetHighGain(GainTuple gain_control)
{
  return (digitalRead(gain_control.g1) == HIGH) && (digitalRead(gain_control.g0) == HIGH);
}

// Calculate threshold for switching analog front end gain.
double GetThreshold(GainState gain_state, double voltage)
{
  double threshold = kCtMax * voltage; // Max Apparent Power = max mains Irms * mains Vrms

  // Adjust for analog front end gain. 
  switch (gain_state)
  {
  case kLow:
    threshold /= -kLowGain;
    break;
  case kMid:
    threshold /= -kMidGain;
    break;
  case kHigh:
    threshold /= -kHighGain;
    break;
  default:
    break;
  }

  return threshold;
}

// Get current analog front end gain setings.
GainState GetGainSettings(GainTuple gain_control)
{
  if (GetLowGain(gain_control))
  {
    return kLow;
  }
  else if (GetMidGain(gain_control))
  {
    return kMid;
  }
  else if (GetHighGain(gain_control))
  {
    return kHigh;
  }
  else
  {
    return kInvalid;
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
  GainState gain_state;
  byte adc_input = AgcData->channel + 1;

  // Compute ADC full scale RMS voltage, 0.905 Vrms @2.56V FS.
  // NB: This is an approximation since waveforms are not pure sinusoids.
  double constexpr kAdcFsVrms = kAdcFs / 2.828427125;

  // Threshold scale factors for switching analog gain settings.
  double constexpr kUpperThreshold = 0.95;
  double constexpr kLowerThreshold = 0.85;

  if (adc_input == kI1Chan)
  {
    amp_cal_low = kI1AmpCalLow;
    amp_cal_mid = kI1AmpCalMid;
    amp_cal_high = kI1AmpCalHigh;
    phase_cal = kI1PhCal;
    gain_control = kCt1GainCtrl;
  }
  else if (adc_input == kI2Chan)
  {
    amp_cal_low = kI2AmpCalLow;
    amp_cal_mid = kI2AmpCalMid;
    amp_cal_high = kI2AmpCalHigh;
    phase_cal = kI2PhCal;
    gain_control = kCt2GainCtrl;
  }

  gain_state = GetGainSettings(gain_control);
  threshold = GetThreshold(gain_state, AgcData->v_rms);

  if (AgcData->apparent_power > threshold * kUpperThreshold)
  {
    // Decrement gains.
    if (gain_state == kHigh)
    {
      SetMidGain(gain_control);
      EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_mid, phase_cal);
    }
    else if (gain_state == kMid)
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
    if (gain_state == kLow)
    {
      v_adc = AgcData->i_rms / amp_cal_low; // approx rms voltage at ADC input
      if (v_adc * -kMidGain < kAdcFsVrms)   // validity check
      {
        SetMidGain(gain_control);
        EmonLibCM_ReCalibrate_IChannel(adc_input, amp_cal_mid, phase_cal);
      }
    }
    else if (gain_state == kMid)
    {
      v_adc = AgcData->i_rms / amp_cal_mid; // approx rms voltage at ADC input
      if (v_adc * -kHighGain < kAdcFsVrms)  // validity check
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
  constexpr double kAssumedVrms= 120.0;  // Assumed line voltage when none is detected
  constexpr unsigned int kLineFreq = 60; // AC line frequency in Hz
  constexpr float kDataLogPeriod = 8.0;  // Interval in seconds over which data is reported

  // Set GPIOs as outputs to control analog front end gain.
  pinMode(kCt1G0, OUTPUT);
  pinMode(kCt1G1, OUTPUT);
  pinMode(kCt2G0, OUTPUT);
  pinMode(kCt2G1, OUTPUT);

  // Set initial analog front end gains.
  SetLowGain(kCt1GainCtrl);
  SetLowGain(kCt2GainCtrl);

  Serial.begin(115200);

  // Configure Emon lib.
  EmonLibCM_setAssumedVrms(kAssumedVrms);
  EmonLibCM_SetADC_VChannel(kVChan, kVAmpCal);                // ADC Input channel, voltage calibration
  EmonLibCM_SetADC_IChannel(kI1Chan, kI1AmpCalLow, kI1PhCal); // ADC Input channel, current calibration, phase calibration
  EmonLibCM_SetADC_IChannel(kI2Chan, kI2AmpCalLow, kI2PhCal); // The current channels will be read in this order
  EmonLibCM_setADC_VRef(INTERNAL2V56);                        // ADC Reference voltage
  EmonLibCM_ADCCal(kAdcFs);                                   // ADC Cal voltage
  EmonLibCM_cycles_per_second(kLineFreq);
  EmonLibCM_datalog_period(kDataLogPeriod);
  EmonLibCM_Init();                                           // Start continuous monitoring
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

    for (size_t i = 0; i < kNumAdcIChan; i++)
    {
      i_rms = EmonLibCM_getIrms(i);
      apparent_power = EmonLibCM_getApparentPower(i);
     
      Serial.print(i_rms,3);Serial.print(",");
      Serial.print(EmonLibCM_getRealPower(i));Serial.print(",");
      Serial.print(apparent_power);
      if (i < kNumAdcIChan - 1) Serial.print(",");

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