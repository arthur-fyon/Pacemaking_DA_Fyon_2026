#=
This file contains all DA model gating functions
=#

# Gating functions
boltz(V, xhalf, xk) = 1 / (1 + exp(-(V-xhalf)/xk))

# Na-current
m_inf(V) = boltz(V, -30.09, 13.2)
m_inf_true(V) = boltz(V, -10., 13.2)
m_inf_true_Yang(V) = boltz(V, -30., 2)
tau_m(V) = 0.01 + 1.0 / ((-(15.6504 + 0.4043*V)/(exp(-19.565 -0.5052*V)-1.0)) + 3.0212*exp(-7.4630e-3*V))
h_inf(V) = boltz(V, -54., -12.8)
tau_h(V) = 0.4 + 1.0 / ((5.0754e-4*exp(-6.3213e-2*V)) + 9.7529*exp(0.13442*V))
hs_inf(V) = boltz(V, -54.8, -1.57)
tau_hs(V) = 20 + 580 / (1 + exp(V))

# Kd-current
n_inf(V) = boltz(V, -25., 12.)
tau_n(V) = (27.2598 / (1+exp(-(V+61.1253)/4.4429))) * (1 / (1+exp((V+36.8869)/9.7083)) + 0.0052) + 0.8876

# L-type Ca-current
l_inf(V) = boltz(V, -45., 7.5)
l_inf_true(V) = boltz(V, -10., 7.5)
tau_l(V) = 1 / ((-0.020876*(V+39.726))/(exp(-(V+39.726)/4.711)-1) + 0.19444*exp(-(V+15.338)/224.21))

# A-type K-current
p_inf(V) = boltz(V, -35.1, 13.4)
tau_p(V) = (95.5813 / (1+exp(-(V+71.5402)/26.0594))) * (1 / (1+exp((V+62.5026)/6.5199)) -0.5108) + 48.2438
q1_inf(V) = boltz(V, -80., -6.)
tau_q1(V) = 6.1*exp(0.015*V)
q2_inf(V) = boltz(V, -80., -6.)
tau_q2(V) = 294.0087 + 55.8321*(1/(1+exp((V+52.5933)/4.9104)) - 5.2348) * (1/(1+exp(V-84.8594)/35.3239))

# ERG K-current
alphao(V) = 0.0036 * exp(0.0759*V)
betao(V) = 1.2523e-5 * exp(-0.0671*V)
alphai(V) = 91.11 * exp(0.1189*V)
betai(V) = 12.6 * exp(0.0733*V)

# H current
mH_inf(V) = boltz(V, -77.6, -17.317)
tau_mH(V) = 26.21 + 3136/(1+exp(-(V+22.686)/29.597))

# New current
mPacemaker_inf(V) = boltz(V, -13.31099526924295, 5.997692378993812)
