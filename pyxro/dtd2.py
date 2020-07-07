# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:41 2019

@author: Henrique
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import lmfit
import xraylib

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

xraylib.XRayInit()

class Sample(object):
    def wavelength_energy_relation(self, t):
        h  = 4.135667662e-15
        c  = 299792458.0
        return h*c/(t*1e-10)
    
    # From QUASES-IMFP-TPP2M
    def imfp(self, Ek=0):
        density = self.symstruct.density
        formula = pt.formula(self.symstruct.formula)
        M_weight = formula.mass
        E_g = 0.0
        v = []
        for i,k in formula.atoms.items():
            v.append(k*ml.element(str(i)).nvalence())
        N_V = np.sum(v)
        E_p = 28.8 * np.sqrt((N_V * density) / M_weight)  
        beta = (-0.10 + (0.944 / (np.sqrt(E_p**2 + E_g**2))) + 
               (0.069 * (density**0.1))) 
        gamma = 0.191 * (1 / np.sqrt(density))
        C_func = 1.97 - (1.096E-3 * E_p**2)
        D_func = 53.4 - (0.025 * E_p**2)
        
        imfp = (E_kinetic / ((E_p**2) * ((beta * np.log(gamma * E_kinetic)) 
               - (C_func / E_kinetic) + (D_func / E_kinetic**2))))
        
        return imfp
    
    def __init__(self, sample = 'Cu', hkl=[1, 1, 1], 
                       mode = 'angular', angle = 0.0, energy = 0.0,
                       polarization = 'sigma', DWF = 1.0):
        
        # Loads crystal from database
        self.crystal = xraylib.Crystal_GetCrystal(sample)
        
        # Some crystal attributes
        self.d_hkl = xraylib.Crystal_dSpacing(self.crystal, *hkl)
        self.stol  = 1/(2*self.d_hkl)
        self.volume = self.crystal['volume']
        self.DWF = DWF
        
        self.h = hkl[0]
        self.k = hkl[1]
        self.l = hkl[2]
        
        # Sets Bragg condition information
        self.mode = mode
        self.Bragg = AttrDict()
        if mode == 'angular':
            if energy == 0.0: # implementar erro
                energy = 3000

            self.Bragg.energy = energy
            self.Bragg.wavelength = self.wavelength_energy_relation(self.Bragg.energy)
            self.Bragg.theta = np.degrees(xraylib.Bragg_angle(self.crystal, self.Bragg.energy / 1000, *hkl))
        elif mode == 'energy':
            if angle == 0.0:
                self.Bragg.energy = energy
                self.Bragg.wavelength = self.wavelength_energy_relation(self.Bragg.energy)
                self.Bragg.theta  = np.degrees(xraylib.Bragg_angle(self.crystal, self.Bragg.energy / 1000, *hkl))
            else:
                self.Bragg.theta = angle
                self.Bragg.wavelength = 2*self.d_hkl*np.sin(np.radians(self.Bragg.theta))
                self.Bragg.energy = self.wavelength_energy_relation(self.Bragg.wavelength)

        self.Bragg.theta_rad = np.radians(self.Bragg.theta)
        
        # Polarization
        self.polarization = polarization
        if self.polarization == 'pi':
            self.P = 2*np.cos(self.Bragg.theta_rad)
        else:
            self.P = 1.0
            
        # Structure factors
        self.F_0  = xraylib.Crystal_F_H_StructureFactor(self.crystal, self.Bragg.energy / 1000, 0, 0, 0, self.DWF, 1.0)
        self.F_H  = xraylib.Crystal_F_H_StructureFactor(self.crystal, self.Bragg.energy / 1000, self.h, self.k, self.l, self.DWF, 1.0)
        self.F_Hb = xraylib.Crystal_F_H_StructureFactor(self.crystal, self.Bragg.energy / 1000, -self.h, -self.k, -self.l, self.DWF, 1.0)
        
        # Gamma factor between Chi and F
        r0 = 2.8179403262e-15
        gamma = 1e10*(r0*self.wavelength_energy_relation(self.Bragg.energy)**2)/(np.pi*self.volume)
        
        # Susceptibilities
        self.Chi_0  = gamma * self.F_0
        self.Chi_H  = gamma * self.F_H
        self.Chi_Hb = gamma * self.F_Hb
        
        # Extra info
        self.info = AttrDict()
        self.info.backscattering_angle = np.degrees(np.pi/2 - 2*np.abs(self.Chi_0))
        self.info.angle_offset   = np.degrees(np.real(self.Chi_0)/np.sin(2*self.Bragg.theta_rad))
        self.info.angle_range    = np.degrees(np.abs(self.P)*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))/np.sin(2*self.Bragg.theta_rad))
        self.info.energy_offset  = np.real(self.Chi_0)*self.Bragg.energy/(2*np.sin(self.Bragg.theta_rad)**2)
        self.info.energy_range   = self.Bragg.energy*np.abs(self.P)*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))/(2*np.sin(self.Bragg.theta_rad)**2)
        self.info.extinct_length = (self.Bragg.wavelength*np.sin(self.Bragg.theta_rad)/(np.pi*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))))


    def calc_reflectivity(self, delta=20, npts=1001, Mono=False, gwidth=False):
        if self.mode == 'angular':
            # a = -(t - tB)*sin(2*tB) - Chi_0
            self.x_range = np.linspace(-delta*self.info.angle_range, delta*self.info.angle_range, npts) # (t - tB)
            x_arg = np.radians(self.x_range)    # (t - tB)
            t_arg = 2*self.Bragg.theta_rad # 2*tB
            a  = - x_arg*np.sin(2*t_arg) + self.Chi_0
        elif self.mode == 'energy':
            # a = -2(E/EB - 1)sin²(tB) - Chi_0
            self.x_range = np.linspace(-delta*self.info.energy_range, delta*self.info.energy_range, npts) # deltaE
            x_arg = self.x_range / self.Bragg.energy # deltaE/E_B
            t_arg = self.Bragg.theta_rad # tB
            a = -2*x_arg*(np.sin(t_arg))**2 + self.Chi_0 # -2(deltaE/E_B)sin²(tB) 
         
        eta = a / np.sqrt(self.Chi_H*self.Chi_Hb)
        
        self.EHE0_plus  = np.sqrt(self.Chi_H/self.Chi_Hb) * (eta + np.sqrt(eta**2 - 1))
        self.EHE0_minus = np.sqrt(self.Chi_H/self.Chi_Hb) * (eta - np.sqrt(eta**2 - 1))
        
        self.EHE0 = self.EHE0_plus.copy()
        self.EHE0[np.real(self.EHE0_plus) < 0] = self.EHE0_plus[np.real(self.EHE0_plus)  < 0]
        self.EHE0[np.real(self.EHE0_plus) > 0] = self.EHE0_minus[np.real(self.EHE0_plus) > 0]
        self.Refl = np.abs(self.EHE0)**2
        
        self.Phase = np.arctan(np.imag(self.EHE0) / np.real(self.EHE0))
        self.Phase[np.real(self.EHE0) > 0] += np.pi
                
        self.Mono = Mono

        if gwidth and gwidth > 0.0:
            gsmear = lmfit.lineshapes.gaussian(self.x_range, center = self.x_range.mean(), sigma = gwidth)
            gsmear = gsmear / sum(gsmear)
            self.Refl = np.convolve(self.Refl, gsmear, mode='same')
            self.Phase = np.convolve(self.Phase, gsmear, mode='same')

    def calc_RC(self, CF, CP, Q = 0.0, Delta = 0.0):
        if self.Mono:
            R = self.Refl_conv_Mono
            P = self.Phase_conv_Mono
        else:
            R = self.Refl
            P = self.Phase

        return 1 + R + 2*np.sqrt(R)*CF*np.cos(P - 2*np.pi*CP)
    
    def Electric_Field(self, z, zmin):
        zpos = z - zmin
        cohpos = np.remainder(zpos / self.d_hkl, 1)
        EF = self.calc_RC(1.0, cohpos)
        if z <= 0:
            EF *= np.exp(-np.abs(z)/self.extinct_length)
        return EF
