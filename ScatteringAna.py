# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:13:34 2017

@author: Dimitris Loukrezis

Analytical calculation of the S11 parameter of a rectangular waveguide with a dielectrical filling.
"""

import numpy as np

def S_ana(z=[5, 30, 3, 7, 5, 2.0, 2.4], dB = False):
    # assign parameter values
    f = z[0]*1e9
    width = z[1]*1e-3
    height = z[2]*1e-3
    fill_l = z[3]*1e-3
    offset = z[4]*1e-3
    epsr = z[5]
    muer = z[6]

    # constants
    c0 = 299792458  # speed of light in vacuum
    mue0 = 4 * np.pi * 1e-7  # vacuum permeability
    # eps0 = 1.0 / (mue0 * c0**2.0) # vacuum permitivity

    # frequency
    w = 2 * np.pi * f  # omega


    # wavenumbers
    k0 = w / c0  # in vacuum
    k = k0 * np.sqrt(muer * epsr)  # in material

    # propagation constants
    ky = np.pi / width  # in y-direction
    kz = np.sqrt(k0 ** 2 - ky ** 2)  # in z-direction for vacuum
    kz1 = np.sqrt(k ** 2 - ky ** 2)  # in z-direction for the debye-material

    # wave impedance for waweguide (TE-wave)
    Z0 = mue0 * w / kz  # for vacuum
    Z1 = mue0 * muer * w / kz1  # for the filling

    # coefficient used to determine the reflection and transmission
    # coefficients from the equation system
    A = (-Z0 / Z1 + 1) * np.exp(-1j * (offset + fill_l) * kz1)
    B = (Z0 / Z1 + 1) * np.exp(1j * (offset + fill_l) * kz1)

    # reflection coefficient at the surface between the 2 materials
    r2 = (2 * Z1 * np.exp(-1j * offset * kz)) / ((Z1 - Z0) * (np.exp(1j * kz1 * offset) - \
                                                              np.exp(1j * (2 * fill_l + offset) * kz1) * (
                                                              (Z1 + Z0) / (Z1 - Z0)) ** 2))
    # transmission coefficient at the surface between the 2 materials
    t2 = -r2 * B / A
    # reflection coefficient at the input port
    r1 = 0.5 * np.exp(-1j * offset * kz) * (t2 * np.exp(-1j * offset * kz1) * (-Z0 / Z1 + 1) + \
                                            r2 * np.exp(1j * offset * kz1) * (Z0 / Z1 + 1))

    #return r1
    if dB:
        return 20*np.log10(abs(r1))
    else: 
        return r1

