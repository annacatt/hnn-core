COMMENT
Conceptual model:  low-threshold (T-type) Ca current of the VIP
  interneuron model by M. M. McCarthy (VIPwithT.cpp, 2023), whose gating
  is taken from her thalamocortical cell model (TCcell.cpp).

  i  = gbar * m^2 * h * G(v)

  G(v) is the Goldman-Hodgkin-Katz flux term exactly as in VIPwithT.cpp,
  with fixed concentrations (no Ca dynamics in the C++ model):

    x    = 2 F v / (R T)                      (v in volts)
    G(v) = 2 F * x / (1 - exp(-x)) * (cai0 - cao0 * exp(-x))

  minf = 1 / (1 + exp(-(v+57)/6.2))
  hinf = 1 / (1 + exp( (v+81)/4))
  taum = 0.612 + 1 / (exp((v+16.8)/18.2) + exp(-(v+132)/16.7))
  tauh = exp((v+467)/66.6)            for v < -81 mV
       = 28 + exp(-(v+22)/10.5)       otherwise

Conversion of the strength parameter:
  In VIPwithT.cpp the current (uA/cm2) is Pca * VIParea * m^2 h G(v)
  with Pca = 80 and VIParea = 0.00029, i.e. 0.0232.
  NEURON currents are in mA/cm2, so gbar = 0.0232 * 1e-3 = 2.32e-5.
  gbar keeps the C++ units (scaled); it is not a conductance in S/cm2.

Implemented as a NONSPECIFIC_CURRENT so it does not change cai/cao or
interact with other Ca mechanisms, as in the C++ model. No dependence on
celsius: the temperature T is fixed at 310 K as in the C++ code.

NEURON implementation: written for runscripts-rt (hnn-tuning format).
ENDCOMMENT

NEURON {
  SUFFIX cat_mccarthy2023
  NONSPECIFIC_CURRENT i
  RANGE gbar, i, cai0, cao0
  GLOBAL tempK
}

UNITS {
  (mV) = (millivolt)
  (mA) = (milliamp)
}

PARAMETER {
  gbar = 2.32e-5          : Pca * VIParea * 1e-3 (see COMMENT)
  cai0 = 1e-8             : intracellular Ca (M), 10 nM, as in the C++ model
  cao0 = 0.003            : extracellular Ca (M), 3 mM
  tempK = 310             : temperature (K)
  FARADAY_C = 96489       : C/mol, value used in the C++ model
  R_GAS = 8.31            : J/(mol K), value used in the C++ model
}

ASSIGNED {
  v (mV)
  i (mA/cm2)
  minf
  hinf
  taum (ms)
  tauh (ms)
}

STATE { m h }

BREAKPOINT {
  SOLVE states METHOD cnexp
  i = gbar * m * m * h * ghk(v)
}

INITIAL {
  rates(v)
  m = minf
  h = hinf
}

DERIVATIVE states {
  rates(v)
  m' = (minf - m) / taum
  h' = (hinf - h) / tauh
}

PROCEDURE rates(v (mV)) {
  UNITSOFF
  minf = 1 / (1 + exp(-(v + 57) / 6.2))
  hinf = 1 / (1 + exp((v + 81) / 4))
  taum = 0.612 + 1 / (exp((v + 16.8) / 18.2) + exp(-(v + 132) / 16.7))
  if (v < -81) {
    tauh = exp((v + 467) / 66.6)
  } else {
    tauh = 28 + exp(-(v + 22) / 10.5)
  }
  UNITSON
}

FUNCTION ghk(v (mV)) {
  LOCAL x, ex
  UNITSOFF
  x = 2 * FARADAY_C * v * 0.001 / (R_GAS * tempK)
  ex = exp(-x)
  ghk = 2 * FARADAY_C * efun(x) * (cai0 - cao0 * ex)
  UNITSON
}

: x / (1 - exp(-x)), with the limit 1 at x = 0
FUNCTION efun(x) {
  if (fabs(x) < 1e-6) {
    efun = 1 + x / 2
  } else {
    efun = x / (1 - exp(-x))
  }
}
