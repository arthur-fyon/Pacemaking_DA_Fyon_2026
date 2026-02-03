# Dopaminergic Neuron Model - Supplementary Code

This repository contains Julia code for simulating and analyzing dopaminergic (DA) neuron models, based on the Canavier et al. (2014) formulation with extensions for pacemaking currents.

## File Structure

### Core Modules

| File | Description |
|------|-------------|
| `DA_kinetics.jl` | Gating variable kinetics (steady-state functions and time constants) |
| `DA_models.jl` | ODE systems for various model variants |
| `DA_utils.jl` | Utility functions for analysis and visualization |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `01_basic_simulations.ipynb` | Basic simulations of the DA neuron model |
| `02_pacemaker_simulations.ipynb` | Effects of pacemaker currents on dynamics |
| `03_monte_carlo_analysis.ipynb` | Monte Carlo analysis of conductance variability |
| `04_monte_carlo_NaLCN.ipynb` | Monte Carlo with NaLCN pacemaker variant |
| `05_1D_parameter_screening.ipynb` | 1D parameter screening of the pacemaker model |
| `06_1D_screening_currentscape.ipynb` | 1D screening with currentscape visualizations |
| `07_1D_screening_voltage_traces.ipynb` | Voltage traces from 1D screening |
| `08_2D_parameter_screening.ipynb` | 2D parameter screening analysis |
| `09_2D_screening_deterministic.ipynb` | Deterministic 2D screening |

## Model Variants

The code implements several variants of the DA neuron model:

1. **`DA_ODE`**: Original model with shifted activation kinetics
2. **`DA_ODE_nohs`**: Model without slow Na⁺ inactivation
3. **`DA_ODE_true_NaLCN`**: Corrected kinetics with NaLCN pacemaker
4. **`DA_ODE_true`**: Corrected kinetics with fitted pacemaker (instantaneous)
5. **`DA_ODE_true_notinstant`**: Non-instantaneous pacemaker activation
6. **`DA_ODE_true_notinstant_transient`**: Time-varying pacemaker conductance

## Dependencies

```julia
using DifferentialEquations
using Plots, StatsPlots
using Statistics, LinearAlgebra
using LaTeXStrings, Printf
using ColorSchemes, DelimitedFiles
using DataFrames, Random, ProgressMeter
using Polynomials
```

## Usage

1. Include the model files:
```julia
include("DA_kinetics.jl")
include("DA_models.jl")
include("DA_utils.jl")
```

2. Set up parameters and initial conditions, then solve:
```julia
using DifferentialEquations

# Define parameters
p = [t -> 0.0, gNa, gCaL, gKd, gKA, gKERG, gKSK, gH, gLNS, gLCa]

# Initial conditions
u0 = [V0, m0, h0, hs0, l0, n0, p0, q1_0, q2_0, o0, i0, mH0, Ca0]

# Solve
prob = ODEProblem(DA_ODE, u0, tspan, p)
sol = solve(prob, Tsit5())
```

## References

- Canavier et al. (2014) - Dopaminergic neuron model

## Author

Arthur Fyon  
University of Liège
