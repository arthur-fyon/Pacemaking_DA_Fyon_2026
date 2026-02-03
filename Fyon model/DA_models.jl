#=
================================================================================
DA_models.jl - Differential Equations for Dopaminergic Neuron Models
================================================================================

This file contains the ordinary differential equations (ODEs) describing various
versions of the dopaminergic (DA) neuron model based on Canavier et al. (2014).

Model variants:
    - DA_ODE                          : Original model with shifted kinetics
    - DA_ODE_nohs                     : Model without slow Na⁺ inactivation (hs)
    - DA_ODE_true_NaLCN               : Corrected kinetics with NaLCN pacemaker
    - DA_ODE_true                     : Corrected kinetics with fitted pacemaker
    - DA_ODE_true_notinstant          : Non-instantaneous pacemaker activation
    - DA_ODE_true_notinstant_transient: Time-varying pacemaker conductance

State variables (13-14 dimensions):
    V    : Membrane potential (mV)
    m    : Na⁺ activation
    h    : Na⁺ fast inactivation
    hs   : Na⁺ slow inactivation
    l    : L-type Ca²⁺ activation
    n    : Delayed-rectifier K⁺ activation
    p    : A-type K⁺ activation
    q1   : A-type K⁺ fast inactivation
    q2   : A-type K⁺ slow inactivation
    o    : ERG K⁺ open state
    i    : ERG K⁺ inactivated state
    mH   : H-current activation
    Ca   : Intracellular calcium concentration (mM)
    (mPacemaker : Pacemaker activation, in some variants)

References:
    Canavier et al. (2014) - Dopaminergic neuron model
=#

include("DA_kinetics.jl")

# ==============================================================================
# Original DA Model (Canavier 2014) - Current-Clamp Mode
# ==============================================================================

"""
    DA_ODE(du, u, p, t)

Original dopaminergic neuron model with shifted activation kinetics.

# Parameters (p)
1. `Iapp(t)`  : Applied current function (pA)
2. `gNa`      : Na⁺ maximal conductance
3. `gCaL`     : L-type Ca²⁺ maximal conductance
4. `gKd`      : Delayed-rectifier K⁺ maximal conductance
5. `gKA`      : A-type K⁺ maximal conductance
6. `gKERG`    : ERG K⁺ maximal conductance
7. `gKSK`     : SK K⁺ maximal conductance
8. `gH`       : H-current maximal conductance
9. `gLNS`     : Non-specific leak conductance
10. `gLCa`    : Ca²⁺ leak conductance

# State Variables (u)
13-dimensional: [V, m, h, hs, l, n, p, q1, q2, o, i, mH, Ca]
"""
function DA_ODE(du, u, p, t)
    # Parameters
    Iapp  = p[1](t)  # Applied current (time-dependent)
    gNa   = p[2]     # Sodium current maximal conductance
    gCaL  = p[3]     # L-type calcium current maximal conductance
    gKd   = p[4]     # Delayed-rectifier potassium current maximal conductance
    gKA   = p[5]     # A-type potassium current maximal conductance
    gKERG = p[6]     # ERG potassium current maximal conductance
    gKSK  = p[7]     # SK current maximal conductance
    gH    = p[8]     # H-current maximal conductance
    gLNS  = p[9]     # Leak non-specific current maximal conductance
    gLCa  = p[10]    # Leak calcium current maximal conductance

    # State variables
    V    = u[1]      # Membrane potential
    m    = u[2]      # Sodium current activation
    h    = u[3]      # Sodium current inactivation
    hs   = u[4]      # Sodium current slow inactivation
    l    = u[5]      # L-type calcium current activation
    n    = u[6]      # Delayed-rectifier potassium current activation
    p    = u[7]      # A-type potassium current activation
    q1   = u[8]      # A-type potassium current fast inactivation
    q2   = u[9]      # A-type potassium current slow inactivation
    o    = u[10]     # ERG potassium current open state
    i    = u[11]     # ERG potassium current inactivated state
    mH   = u[12]     # H-current activation
    Ca   = u[13]     # Intracellular calcium concentration
    
    # Calcium-dependent SK current and pump
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics
    du[1] = 1/C * (
        - gNa * m^3 * h * hs * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics
    du[2]  = (1 / tau_m(V))  * (m_inf(V)  - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)  - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V) - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf(V)  - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)  - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)  - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V) - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    
    # Calcium dynamics
    du[13] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# DA Model Without Slow Na⁺ Inactivation
# ==============================================================================

"""
    DA_ODE_nohs(du, u, p, t)

Dopaminergic neuron model without slow Na⁺ inactivation (hs) in the membrane
potential equation. The hs variable is still tracked but does not affect INa.

See `DA_ODE` for parameter and state variable descriptions.
"""
function DA_ODE_nohs(du, u, p, t)
    # Parameters
    Iapp  = p[1](t)
    gNa   = p[2]
    gCaL  = p[3]
    gKd   = p[4]
    gKA   = p[5]
    gKERG = p[6]
    gKSK  = p[7]
    gH    = p[8]
    gLNS  = p[9]
    gLCa  = p[10]

    # State variables
    V    = u[1]
    m    = u[2]
    h    = u[3]
    hs   = u[4]
    l    = u[5]
    n    = u[6]
    p    = u[7]
    q1   = u[8]
    q2   = u[9]
    o    = u[10]
    i    = u[11]
    mH   = u[12]
    Ca   = u[13]
    
    # Calcium-dependent terms
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics (note: hs removed from INa)
    du[1] = 1/C * (
        - gNa * m^3 * h * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics
    du[2]  = (1 / tau_m(V))  * (m_inf(V)  - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)  - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V) - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf(V)  - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)  - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)  - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V) - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    
    # Calcium dynamics
    du[13] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# Corrected DA Model with NaLCN Pacemaker Current
# ==============================================================================

"""
    DA_ODE_true_NaLCN(du, u, p, t)

Corrected dopaminergic neuron model with physiological activation kinetics 
and a sodium leak channel (NaLCN) as the pacemaking current.

# Additional Parameters
11. `gNaLCN` : NaLCN pacemaking current maximal conductance

Uses corrected activation curves: `m_inf_true` and `l_inf_true`.
"""
function DA_ODE_true_NaLCN(du, u, p, t)
    # Parameters
    Iapp   = p[1](t)
    gNa    = p[2]
    gCaL   = p[3]
    gKd    = p[4]
    gKA    = p[5]
    gKERG  = p[6]
    gKSK   = p[7]
    gH     = p[8]
    gLNS   = p[9]
    gLCa   = p[10]
    gNaLCN = p[11]    # Pacemaking current (NaLCN)

    # State variables
    V    = u[1]
    m    = u[2]
    h    = u[3]
    hs   = u[4]
    l    = u[5]
    n    = u[6]
    p    = u[7]
    q1   = u[8]
    q2   = u[9]
    o    = u[10]
    i    = u[11]
    mH   = u[12]
    Ca   = u[13]
    
    # Calcium-dependent terms
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics
    du[1] = 1/C * (
        - gNa * m^3 * h * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        - gNaLCN * (V - VNa)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics (using corrected kinetics)
    du[2]  = (1 / tau_m(V))  * (m_inf_true(V) - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)      - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V)     - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf_true(V) - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)      - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)      - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V)     - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V)     - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    
    # Calcium dynamics
    du[13] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# Corrected DA Model with Fitted Pacemaker Current
# ==============================================================================

"""
    DA_ODE_true_instant(du, u, p, t)

Corrected dopaminergic neuron model with physiological activation kinetics
and a fitted pacemaking current with instantaneous activation.

# Additional Parameters
11. `gPacemaker` : Pacemaking current maximal conductance

The pacemaker current uses `mPacemaker_inf(V)` for voltage-dependent activation
and reverses at `EPacemaker`.
"""
function DA_ODE_true_instant(du, u, p, t)
    # Parameters
    Iapp       = p[1](t)
    gNa        = p[2]
    gCaL       = p[3]
    gKd        = p[4]
    gKA        = p[5]
    gKERG      = p[6]
    gKSK       = p[7]
    gH         = p[8]
    gLNS       = p[9]
    gLCa       = p[10]
    gPacemaker = p[11]    # Pacemaking current

    # State variables
    V    = u[1]
    m    = u[2]
    h    = u[3]
    hs   = u[4]
    l    = u[5]
    n    = u[6]
    p    = u[7]
    q1   = u[8]
    q2   = u[9]
    o    = u[10]
    i    = u[11]
    mH   = u[12]
    Ca   = u[13]
    
    # Calcium-dependent terms
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics
    du[1] = 1/C * (
        - gNa * m^3 * h * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        - gPacemaker * mPacemaker_inf(V) * (V - EPacemaker)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics (using corrected kinetics)
    du[2]  = (1 / tau_m(V))  * (m_inf_true(V) - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)      - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V)     - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf_true(V) - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)      - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)      - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V)     - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V)     - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    
    # Calcium dynamics
    du[13] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# Corrected DA Model with Non-Instantaneous Pacemaker Activation
# ==============================================================================

"""
    DA_ODE_true_notinstant(du, u, p, t)

Corrected dopaminergic neuron model with non-instantaneous pacemaker activation.
The pacemaker gating variable `mPacemaker` has its own time constant.

# Additional Parameters
11. `gPacemaker` : Pacemaking current maximal conductance
12. `tau`        : Time constant multiplier for pacemaker activation

# State Variables
14-dimensional: [V, m, h, hs, l, n, p, q1, q2, o, i, mH, mPacemaker, Ca]
"""
function DA_ODE_true_notinstant(du, u, p, t)
    # Parameters
    Iapp       = p[1](t)
    gNa        = p[2]
    gCaL       = p[3]
    gKd        = p[4]
    gKA        = p[5]
    gKERG      = p[6]
    gKSK       = p[7]
    gH         = p[8]
    gLNS       = p[9]
    gLCa       = p[10]
    gPacemaker = p[11]
    tau        = p[12]    # Pacemaker time constant multiplier

    # State variables
    V          = u[1]
    m          = u[2]
    h          = u[3]
    hs         = u[4]
    l          = u[5]
    n          = u[6]
    p          = u[7]
    q1         = u[8]
    q2         = u[9]
    o          = u[10]
    i          = u[11]
    mH         = u[12]
    mPacemaker = u[13]    # Pacemaker activation (dynamic)
    Ca         = u[14]
    
    # Calcium-dependent terms
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics
    du[1] = 1/C * (
        - gNa * m^3 * h * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        - gPacemaker * mPacemaker * (V - EPacemaker)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics
    du[2]  = (1 / tau_m(V))  * (m_inf_true(V) - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)      - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V)     - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf_true(V) - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)      - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)      - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V)     - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V)     - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = (1 / (tau * tau_m(V))) * (mPacemaker_inf(V) - mPacemaker)
    
    # Calcium dynamics
    du[14] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# Corrected DA Model with Time-Varying Pacemaker Conductance
# ==============================================================================

"""
    DA_ODE_true_notinstant_transient(du, u, p, t)

Corrected dopaminergic neuron model with non-instantaneous pacemaker activation
and time-varying pacemaker conductance (for transient stimulation protocols).

# Additional Parameters
11. `gPacemaker(t)` : Time-dependent pacemaking current maximal conductance (nS)
12. `tau`           : Time constant for pacemaker activation

# State Variables
14-dimensional: [V, m, h, hs, l, n, p, q1, q2, o, i, mH, mPacemaker, Ca]
"""
function DA_ODE_true_notinstant_transient(du, u, p, t)
    # Parameters
    Iapp       = p[1](t)
    gNa        = p[2]
    gCaL       = p[3]
    gKd        = p[4]
    gKA        = p[5]
    gKERG      = p[6]
    gKSK       = p[7]
    gH         = p[8]
    gLNS       = p[9]
    gLCa       = p[10]
    gPacemaker = p[11](t)    # Time-dependent pacemaker conductance
    tau        = p[12]

    # State variables
    V          = u[1]
    m          = u[2]
    h          = u[3]
    hs         = u[4]
    l          = u[5]
    n          = u[6]
    p          = u[7]
    q1         = u[8]
    q2         = u[9]
    o          = u[10]
    i          = u[11]
    mH         = u[12]
    mPacemaker = u[13]
    Ca         = u[14]
    
    # Calcium-dependent terms
    SK_inf = 0.0
    ICap = 0.0
    if Ca > 0
        SK_inf = 1 / (1 + (0.00019 / Ca)^4)
        ICap = ICapmax / (1 + (0.0005 / Ca))
    end

    # Membrane potential dynamics
    du[1] = 1/C * (
        - gNa * m^3 * h * (V - VNa)
        - gCaL * l * (V - VCa)
        - gKd * n^3 * (V - VK)
        - gKA * p * (q1/2 + q2/2) * (V - VK)
        - gKERG * o * (V - VK)
        - gKSK * (V - VK) * SK_inf
        - gH * mH^2 * (V - VH)
        - gLCa * (V - VCa)
        - gLNS * (V - VLNS)
        - gPacemaker * mPacemaker * (V - EPacemaker)
        + 100 * Iapp / (pi * d * L)
    )

    # Gating variable dynamics
    du[2]  = (1 / tau_m(V))  * (m_inf_true(V) - m)
    du[3]  = (1 / tau_h(V))  * (h_inf(V)      - h)
    du[4]  = (1 / tau_hs(V)) * (hs_inf(V)     - hs)
    du[5]  = (1 / tau_l(V))  * (l_inf_true(V) - l)
    du[6]  = (1 / tau_n(V))  * (n_inf(V)      - n)
    du[7]  = (1 / tau_p(V))  * (p_inf(V)      - p)
    du[8]  = (1 / tau_q1(V)) * (q1_inf(V)     - q1)
    du[9]  = (1 / tau_q2(V)) * (q2_inf(V)     - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1 / tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = (1 / tau) * (mPacemaker_inf(V) - mPacemaker)
    
    # Calcium dynamics
    du[14] = -2 * fCa * (gLCa * (V - VCa) + ICap + gCaL * l * (V - VCa)) / (F * d * 0.1)
end

# ==============================================================================
# Stimulation Functions
# ==============================================================================

"""
    heaviside(t)

Heaviside step function.

# Returns
- 0 if t < 0
- 0.5 if t = 0
- 1 if t > 0
"""
heaviside(t) = (1 + sign(t)) / 2

"""
    pulse(t, ti, tf)

Rectangular pulse function.

# Arguments
- `t`  : Time
- `ti` : Pulse onset time
- `tf` : Pulse offset time

# Returns
- 1 if ti ≤ t < tf
- 0 otherwise
"""
pulse(t, ti, tf) = heaviside(t - ti) - heaviside(t - tf)
