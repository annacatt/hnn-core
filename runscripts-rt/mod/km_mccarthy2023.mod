COMMENT
Conceptual model:  M current (slow, non-inactivating K current) of the
  VIP interneuron model by M. M. McCarthy (VIPwithT.cpp, 2023).

Kinetics follow Mainen & Sejnowski (1996) with the temperature factor
already folded in (2.3^((37-23)/10) = 3.209) and a maximal rate of
1e-4 /ms, exactly as written in VIPwithT.cpp:

  alpha(v) =  3.209e-4 * (v+30) / (1 - exp(-(v+30)/9))
  beta(v)  = -3.209e-4 * (v+30) / (1 - exp( (v+30)/9))
  dw/dt    = alpha*(1-w) - beta*w
  ik       = gbar * w * (v - ek)

There is no dependence on celsius (the C++ model has none).
In VIPwithT.cpp gm = 1 mS/cm2  ->  gbar = 0.001 S/cm2.

NEURON implementation: written for runscripts-rt (hnn-tuning format).
ENDCOMMENT

NEURON {
  SUFFIX km_mccarthy2023
  USEION k READ ek WRITE ik
  RANGE gbar, g, ik
  GLOBAL rmax, vhalf, k_slope
}

UNITS {
  (S) = (siemens)
  (mV) = (millivolt)
  (mA) = (milliamp)
}

PARAMETER {
  gbar = 0.001 (S/cm2)
  rmax = 3.209e-4 (/ms/mV)  : 3.209 (Q10 factor at 37 C) * 1e-4
  vhalf = -30 (mV)
  k_slope = 9 (mV)
}

ASSIGNED {
  v (mV)
  ek (mV)
  ik (mA/cm2)
  g (S/cm2)
  alpha (/ms)
  beta (/ms)
}

STATE { w }

BREAKPOINT {
  SOLVE states METHOD cnexp
  g = gbar * w
  ik = g * (v - ek)
}

INITIAL {
  rates(v)
  w = alpha / (alpha + beta)
}

DERIVATIVE states {
  rates(v)
  w' = alpha * (1 - w) - beta * w
}

PROCEDURE rates(v (mV)) {
  UNITSOFF
  alpha = rmax * k_slope * vtrap((v - vhalf) / k_slope)
  beta  = rmax * k_slope * vtrap(-(v - vhalf) / k_slope)
  UNITSON
}

: x / (1 - exp(-x)), with the limit 1 at x = 0
FUNCTION vtrap(x) {
  if (fabs(x) < 1e-6) {
    vtrap = 1 + x / 2
  } else {
    vtrap = x / (1 - exp(-x))
  }
}
