#=
This file contains differential equations describing the DA Canavier model
=#

include("DA_kinetics.jl") # Include DA model gating functions

## DA model from Canavier 2014 - current-clamp mode
function DA_ODE(du, u, p, t)
    # Parameters
    Iapp  = p[1](t) # Amplitude of constant applied current
    gNa   = p[2] # Sodium current maximal conductance
    gCaL  = p[3] # L-type calcium current maximal conductance
    gKd   = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA   = p[5] # A-type potassium current maximal conductance
    gKERG = p[6] # ERG potassium current maximal conductance
    gKSK  = p[7] # SK current maximal conductance
    gH    = p[8] # SK current maximal conductance
    gLNS  = p[9] # Leak non specific current maximal conductance
    gLCa  = p[10] # Leak calcium current maximal conductance

    # Variables
    V    = u[1] # Membrane potential
    m    = u[2] # Sodium current activation
    h    = u[3] # Sodium current inactivation
    hs   = u[4] # Sodium current second inactivation
    l    = u[5] # L-type calcium current activation
    n    = u[6] # Delayed-rectifier potassium current activation
    p    = u[7] # A-type potassium current activation
    q1   = u[8] # A-type potassium potassium current inactivation 1
    q2   = u[9] # A-type potassium potassium current inactivation 2
    o    = u[10] # ERG potassium current activation
    i    = u[11] # ERG potassium current activation intermediate
    mH   = u[12] # H current activation
    Ca   = u[13] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*hs*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## DA model from Canavier 2014 - current-clamp mode
function DA_ODE_nohs(du, u, p, t)
    # Parameters
    Iapp  = p[1](t) # Amplitude of constant applied current
    gNa   = p[2] # Sodium current maximal conductance
    gCaL  = p[3] # L-type calcium current maximal conductance
    gKd   = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA   = p[5] # A-type potassium current maximal conductance
    gKERG = p[6] # ERG potassium current maximal conductance
    gKSK  = p[7] # SK current maximal conductance
    gH    = p[8] # SK current maximal conductance
    gLNS  = p[9] # Leak non specific current maximal conductance
    gLCa  = p[10] # Leak calcium current maximal conductance

    # Variables
    V    = u[1] # Membrane potential
    m    = u[2] # Sodium current activation
    h    = u[3] # Sodium current inactivation
    hs   = u[4] # Sodium current second inactivation
    l    = u[5] # L-type calcium current activation
    n    = u[6] # Delayed-rectifier potassium current activation
    p    = u[7] # A-type potassium current activation
    q1   = u[8] # A-type potassium potassium current inactivation 1
    q2   = u[9] # A-type potassium potassium current inactivation 2
    o    = u[10] # ERG potassium current activation
    i    = u[11] # ERG potassium current activation intermediate
    mH   = u[12] # H current activation
    Ca   = u[13] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## Corrected DA model from Canavier 2014 - current-clamp mode
function DA_ODE_true_NaLCN(du, u, p, t)
    # Parameters
    Iapp       = p[1](t) # Amplitude of constant applied current
    gNa        = p[2] # Sodium current maximal conductance
    gCaL       = p[3] # L-type calcium current maximal conductance
    gKd        = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA        = p[5] # A-type potassium current maximal conductance
    gKERG      = p[6] # ERG potassium current maximal conductance
    gKSK       = p[7] # SK current maximal conductance
    gH         = p[8] # SK current maximal conductance
    gLNS       = p[9] # Leak non specific current maximal conductance
    gLCa       = p[10] # Leak calcium current maximal conductance
    gNaLCN     = p[11] # Pacemaking current maximal conductance

    # Variables
    V    = u[1] # Membrane potential
    m    = u[2] # Sodium current activation
    h    = u[3] # Sodium current inactivation
    hs   = u[4] # Sodium current second inactivation
    l    = u[5] # L-type calcium current activation
    n    = u[6] # Delayed-rectifier potassium current activation
    p    = u[7] # A-type potassium current activation
    q1   = u[8] # A-type potassium potassium current inactivation 1
    q2   = u[9] # A-type potassium potassium current inactivation 2
    o    = u[10] # ERG potassium current activation
    i    = u[11] # ERG potassium current activation intermediate
    mH   = u[12] # H current activation
    Ca   = u[13] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) -
                  gNaLCN*(V-VNa) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf_true(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf_true(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## Corrected DA model from Canavier 2014 - current-clamp mode
function DA_ODE_true_instant(du, u, p, t)
    # Parameters
    Iapp       = p[1](t) # Amplitude of constant applied current
    gNa        = p[2] # Sodium current maximal conductance
    gCaL       = p[3] # L-type calcium current maximal conductance
    gKd        = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA        = p[5] # A-type potassium current maximal conductance
    gKERG      = p[6] # ERG potassium current maximal conductance
    gKSK       = p[7] # SK current maximal conductance
    gH         = p[8] # SK current maximal conductance
    gLNS       = p[9] # Leak non specific current maximal conductance
    gLCa       = p[10] # Leak calcium current maximal conductance
    gPacemaker = p[11] # Pacemaking current maximal conductance

    # Variables
    V    = u[1] # Membrane potential
    m    = u[2] # Sodium current activation
    h    = u[3] # Sodium current inactivation
    hs   = u[4] # Sodium current second inactivation
    l    = u[5] # L-type calcium current activation
    n    = u[6] # Delayed-rectifier potassium current activation
    p    = u[7] # A-type potassium current activation
    q1   = u[8] # A-type potassium potassium current inactivation 1
    q2   = u[9] # A-type potassium potassium current inactivation 2
    o    = u[10] # ERG potassium current activation
    i    = u[11] # ERG potassium current activation intermediate
    mH   = u[12] # H current activation
    Ca   = u[13] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) -
                  gPacemaker*mPacemaker_inf(V)*(V-EPacemaker) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf_true(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf_true(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## Corrected DA model from Canavier 2014 - current-clamp mode
function DA_ODE_true_notinstant(du, u, p, t)
    # Parameters
    Iapp       = p[1](t) # Amplitude of constant applied current
    gNa        = p[2] # Sodium current maximal conductance
    gCaL       = p[3] # L-type calcium current maximal conductance
    gKd        = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA        = p[5] # A-type potassium current maximal conductance
    gKERG      = p[6] # ERG potassium current maximal conductance
    gKSK       = p[7] # SK current maximal conductance
    gH         = p[8] # SK current maximal conductance
    gLNS       = p[9] # Leak non specific current maximal conductance
    gLCa       = p[10] # Leak calcium current maximal conductance
    gPacemaker = p[11] # Pacemaking current maximal conductance
    tau        = p[12] # Pacemaking time constant multiplier

    # Variables
    V          = u[1] # Membrane potential
    m          = u[2] # Sodium current activation
    h          = u[3] # Sodium current inactivation
    hs         = u[4] # Sodium current second inactivation
    l          = u[5] # L-type calcium current activation
    n          = u[6] # Delayed-rectifier potassium current activation
    p          = u[7] # A-type potassium current activation
    q1         = u[8] # A-type potassium potassium current inactivation 1
    q2         = u[9] # A-type potassium potassium current inactivation 2
    o          = u[10] # ERG potassium current activation
    i          = u[11] # ERG potassium current activation intermediate
    mH         = u[12] # H current activation
    mPacemaker = u[13] # Pacemaking current activation
    Ca         = u[14] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) -
                  gPacemaker*mPacemaker*(V-EPacemaker) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf_true(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf_true(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = (1/(tau*tau_m(V))) * (mPacemaker_inf(V) - mPacemaker)
    du[14] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## Corrected DA model from Canavier 2014 - current-clamp mode
function DA_ODE_true_notinstant_transient(du, u, p, t)
    # Parameters
    Iapp       = p[1](t) # Amplitude of constant applied current
    gNa        = p[2] # Sodium current maximal conductance
    gCaL       = p[3] # L-type calcium current maximal conductance
    gKd        = p[4] # Delayed-rectifier potassium current maximal conductance
    gKA        = p[5] # A-type potassium current maximal conductance
    gKERG      = p[6] # ERG potassium current maximal conductance
    gKSK       = p[7] # SK current maximal conductance
    gH         = p[8] # SK current maximal conductance
    gLNS       = p[9] # Leak non specific current maximal conductance
    gLCa       = p[10] # Leak calcium current maximal conductance
    gPacemaker = p[11](t) # Pacemaking current maximal conductance
    tau        = p[12] # Pacemaking time constant multiplier

    # Variables
    V          = u[1] # Membrane potential
    m          = u[2] # Sodium current activation
    h          = u[3] # Sodium current inactivation
    hs         = u[4] # Sodium current second inactivation
    l          = u[5] # L-type calcium current activation
    n          = u[6] # Delayed-rectifier potassium current activation
    p          = u[7] # A-type potassium current activation
    q1         = u[8] # A-type potassium potassium current inactivation 1
    q2         = u[9] # A-type potassium potassium current inactivation 2
    o          = u[10] # ERG potassium current activation
    i          = u[11] # ERG potassium current activation intermediate
    mH         = u[12] # H current activation
    mPacemaker = u[13] # Pacemaking current activation
    Ca         = u[14] # Intracellular calcium level
    
    SK_inf = 0.
    ICap = 0.
    if Ca > 0
        SK_inf = 1/(1+(0.00019/Ca)^4)
        ICap = ICapmax/(1 + (0.0005/Ca))
    end

    # ODEs
    du[1]= 1/C*(- gNa*m^3*h*(V-VNa) - gCaL*l*(V-VCa) - gKd*n^3*(V-VK) -
                  gKA*p*(q1/2+q2/2)*(V-VK) - gKERG*o*(V-VK) - gKSK*(V-VK)*SK_inf -
                  gH*mH^2*(V-VH) - gLCa*(V-VCa) - gLNS*(V-VLNS) -
                  gPacemaker*mPacemaker*(V-EPacemaker) + 100*Iapp/(pi*d*L))

    du[2] = (1/tau_m(V)) * (m_inf_true(V) - m)
    du[3] = (1/tau_h(V)) * (h_inf(V) - h)
    du[4] = (1/tau_hs(V)) * (hs_inf(V) - hs)
    du[5] = (1/tau_l(V)) * (l_inf_true(V) - l)
    du[6] = (1/tau_n(V)) * (n_inf(V) - n)
    du[7] = (1/tau_p(V)) * (p_inf(V) - p)
    du[8] = (1/tau_q1(V)) * (q1_inf(V) - q1)
    du[9] = (1/tau_q2(V)) * (q2_inf(V) - q2)
    du[10] = alphao(V) * (1 - o - i) + betai(V) * i - o * (alphai(V) + betao(V))
    du[11] = alphai(V) * o - betai(V) * i
    du[12] = (1/tau_mH(V)) * (mH_inf(V) - mH)
    du[13] = (1/tau) * (mPacemaker_inf(V) - mPacemaker)
    du[14] = -2 * fCa * (gLCa*(V-VCa) + ICap + gCaL*l*(V-VCa)) / (F*d*0.1)
end

## Stimulation function
heaviside(t) = (1 + sign(t)) / 2
pulse(t, ti, tf) = heaviside(t-ti) - heaviside(t-tf)
