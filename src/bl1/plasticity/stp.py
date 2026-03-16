"""Tsodyks-Markram short-term plasticity — Phase 2 placeholder.

Facilitation and depression:
    dx/dt = (1-x)/τ_rec - u*x*δ(t-t_spike)
    du/dt = (U-u)/τ_fac + U*(1-u)*δ(t-t_spike)
"""
