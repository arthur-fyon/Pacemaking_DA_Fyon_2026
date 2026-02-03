#=
================================================================================
DA_kinetics.jl - Gating Variable Kinetics for Dopaminergic Neuron Model
================================================================================

This file contains all gating functions for the dopaminergic (DA) neuron model,
based on the Canavier et al. (2014) formulation. The kinetics describe the 
voltage-dependent activation and inactivation of various ionic currents.

Ionic currents modeled:
    - Na⁺  : Transient sodium current (m, h, hs gating)
    - Kd   : Delayed-rectifier potassium current (n gating)
    - CaL  : L-type calcium current (l gating)
    - KA   : A-type potassium current (p, q1, q2 gating)
    - KERG : ERG potassium current (o, i gating)
    - H    : Hyperpolarization-activated current (mH gating)
    - Pacemaker : Pacemaking current (mPacemaker gating)

References:
    Canavier et al. (2014) - Dopaminergic neuron model

Units:
    - Voltage (V): mV
    - Time constants (τ): ms
=#

# ==============================================================================
# Boltzmann Function
# ==============================================================================

"""
    boltz(V, xhalf, xk)

Compute the Boltzmann steady-state activation/inactivation function.

# Arguments
- `V::Real`: Membrane potential (mV)
- `xhalf::Real`: Half-activation voltage (mV)
- `xk::Real`: Slope factor (mV), positive for activation, negative for inactivation

# Returns
- Steady-state value ∈ [0, 1]
"""
boltz(V, xhalf, xk) = 1 / (1 + exp(-(V - xhalf) / xk))

# ==============================================================================
# Transient Sodium Current (INa)
# ==============================================================================

"""
    m_inf(V)

Steady-state activation for Na⁺ current (shifted kinetics for model fitting).
"""
m_inf(V) = boltz(V, -30.09, 13.2)

"""
    m_inf_true(V)

Steady-state activation for Na⁺ current (physiological kinetics).
"""
m_inf_true(V) = boltz(V, -10.0, 13.2)

"""
    m_inf_true_Yang(V)

Steady-state activation for Na⁺ current (Yang et al. variant).
"""
m_inf_true_Yang(V) = boltz(V, -30.0, 2)

"""
    tau_m(V)

Time constant for Na⁺ activation (ms).
"""
tau_m(V) = 0.01 + 1.0 / ((-(15.6504 + 0.4043 * V) / (exp(-19.565 - 0.5052 * V) - 1.0)) + 3.0212 * exp(-7.4630e-3 * V))

"""
    h_inf(V)

Steady-state fast inactivation for Na⁺ current.
"""
h_inf(V) = boltz(V, -54.0, -12.8)

"""
    tau_h(V)

Time constant for Na⁺ fast inactivation (ms).
"""
tau_h(V) = 0.4 + 1.0 / ((5.0754e-4 * exp(-6.3213e-2 * V)) + 9.7529 * exp(0.13442 * V))

"""
    hs_inf(V)

Steady-state slow inactivation for Na⁺ current.
"""
hs_inf(V) = boltz(V, -54.8, -1.57)

"""
    tau_hs(V)

Time constant for Na⁺ slow inactivation (ms).
"""
tau_hs(V) = 20 + 580 / (1 + exp(V))

# ==============================================================================
# Delayed-Rectifier Potassium Current (IKd)
# ==============================================================================

"""
    n_inf(V)

Steady-state activation for delayed-rectifier K⁺ current.
"""
n_inf(V) = boltz(V, -25.0, 12.0)

"""
    tau_n(V)

Time constant for delayed-rectifier K⁺ activation (ms).
"""
tau_n(V) = (27.2598 / (1 + exp(-(V + 61.1253) / 4.4429))) * (1 / (1 + exp((V + 36.8869) / 9.7083)) + 0.0052) + 0.8876

# ==============================================================================
# L-type Calcium Current (ICaL)
# ==============================================================================

"""
    l_inf(V)

Steady-state activation for L-type Ca²⁺ current (shifted kinetics).
"""
l_inf(V) = boltz(V, -45.0, 7.5)

"""
    l_inf_true(V)

Steady-state activation for L-type Ca²⁺ current (physiological kinetics).
"""
l_inf_true(V) = boltz(V, -10.0, 7.5)

"""
    tau_l(V)

Time constant for L-type Ca²⁺ activation (ms).
"""
tau_l(V) = 1 / ((-0.020876 * (V + 39.726)) / (exp(-(V + 39.726) / 4.711) - 1) + 0.19444 * exp(-(V + 15.338) / 224.21))

# ==============================================================================
# A-type Potassium Current (IKA)
# ==============================================================================

"""
    p_inf(V)

Steady-state activation for A-type K⁺ current.
"""
p_inf(V) = boltz(V, -35.1, 13.4)

"""
    tau_p(V)

Time constant for A-type K⁺ activation (ms).
"""
tau_p(V) = (95.5813 / (1 + exp(-(V + 71.5402) / 26.0594))) * (1 / (1 + exp((V + 62.5026) / 6.5199)) - 0.5108) + 48.2438

"""
    q1_inf(V)

Steady-state fast inactivation for A-type K⁺ current.
"""
q1_inf(V) = boltz(V, -80.0, -6.0)

"""
    tau_q1(V)

Time constant for A-type K⁺ fast inactivation (ms).
"""
tau_q1(V) = 6.1 * exp(0.015 * V)

"""
    q2_inf(V)

Steady-state slow inactivation for A-type K⁺ current.
"""
q2_inf(V) = boltz(V, -80.0, -6.0)

"""
    tau_q2(V)

Time constant for A-type K⁺ slow inactivation (ms).
"""
tau_q2(V) = 294.0087 + 55.8321 * (1 / (1 + exp((V + 52.5933) / 4.9104)) - 5.2348) * (1 / (1 + exp(V - 84.8594) / 35.3239))

# ==============================================================================
# ERG Potassium Current (IKERG)
# ==============================================================================

"""
    alphao(V)

Forward rate constant for ERG K⁺ channel opening (ms⁻¹).
"""
alphao(V) = 0.0036 * exp(0.0759 * V)

"""
    betao(V)

Backward rate constant for ERG K⁺ channel opening (ms⁻¹).
"""
betao(V) = 1.2523e-5 * exp(-0.0671 * V)

"""
    alphai(V)

Forward rate constant for ERG K⁺ channel inactivation (ms⁻¹).
"""
alphai(V) = 91.11 * exp(0.1189 * V)

"""
    betai(V)

Backward rate constant for ERG K⁺ channel inactivation (ms⁻¹).
"""
betai(V) = 12.6 * exp(0.0733 * V)

# ==============================================================================
# Hyperpolarization-Activated Current (IH)
# ==============================================================================

"""
    mH_inf(V)

Steady-state activation for H-current.
"""
mH_inf(V) = boltz(V, -77.6, -17.317)

"""
    tau_mH(V)

Time constant for H-current activation (ms).
"""
tau_mH(V) = 26.21 + 3136 / (1 + exp(-(V + 22.686) / 29.597))

# ==============================================================================
# Pacemaking Current
# ==============================================================================

"""
    mPacemaker_inf(V)

Steady-state activation for the pacemaking current.
"""
mPacemaker_inf(V) = boltz(V, -13.31099526924295, 5.997692378993812)
