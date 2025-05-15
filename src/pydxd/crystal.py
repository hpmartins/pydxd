# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:41 2019

@author: Henrique
"""

import periodictable as pt
from periodictable.xsf import Xray

import mendeleev as ml
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import lmfit
import pandas as pd
import xraydb

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Crystal(object):
    def __init__(self, filename='', hkl=[1, 1, 1], nuc=1, structure=None, shift = [0, 0, 0]):
        # Load crystal structure and gets the symmetrized structure
        if structure:
            self.structure = structure
        else:
            self.structure = Structure.from_file(filename)
        
        # self.structure = self.structure.get_primitive_structure()
        # self.sym_structure = SpacegroupAnalyzer(self.structure).get_symmetrized_structure()
        
        # Calculates interplanar distance
        self.hkl = np.array(hkl)
        self.d_hkl = self.structure.lattice.d_hkl(self.hkl)
        self.stol = 1/(2*self.d_hkl)
        self.H = self.structure.lattice.reciprocal_lattice_crystallographic.matrix.dot(self.hkl)

        #
        self.hkl_norm = self.hkl/np.sqrt(np.dot(self.hkl, self.hkl))
        self.z_dist = np.abs(self.hkl_norm.dot(self.structure.lattice.abc))
    
        # Vector perpendicular to the (hkl) direction
        tmpa = np.random.randn(3)
        tmpa -= tmpa.dot(self.hkl_norm) * self.hkl_norm
        tmpa /= np.linalg.norm(tmpa)
        tmpb = np.cross(tmpa, self.hkl_norm)
        self.hkl_perp = (tmpa + tmpb)/np.linalg.norm(tmpa + tmpb)
    
        # Sets sites
        self.sites = pd.DataFrame([{
            'Z': site.specie.number,
            'name': site.specie.name,
            'label': '{}{}'.format(site.specie.name, idx),
            'zcoord': self.hkl_norm.dot(site.coords + shift),
            'Hdotr': self.H.dot(site.coords + shift),
        } for idx, site in enumerate(self.structure.sites)])
        
        self.sites = self.sites.sort_values(by = 'zcoord', ignore_index = True, ascending = True)

    def volume(self):
        a = self.structure.lattice.a
        b = self.structure.lattice.b
        c = self.structure.lattice.c
    
        cosA = np.cos(np.radians(self.structure.lattice.alpha))
        cosB = np.cos(np.radians(self.structure.lattice.beta))
        cosC = np.cos(np.radians(self.structure.lattice.gamma))
    
        return a*b*c*np.sqrt(1 - cosA**2 - cosB**2 - cosC**2 + 2*cosA*cosB*cosC)

    def wavelength_energy_relation(self, t):
        h  = 4.135667662e-15
        c  = 299792458.0
        return h*c/(t*1e-10)

    def imfp(self, Ek=0):
        formula = pt.formula(self.structure.formula)
        density = formula.molecular_mass / (1e-24*self.volume())
        
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
        
        imfp = (Ek / ((E_p**2) * ((beta * np.log(gamma * Ek)) 
               - (C_func / Ek) + (D_func / Ek**2))))
        
        return imfp
    
    # Based on:
    # 1. J. Zegenhagen, Surface structure determination with X-ray standing waves. Surf. Sci. Rep. 18, 202–271 (1993).
    # 2. B. W. Batterman, H. Cole, Dynamical Diffraction of X Rays by Perfect Crystals. Rev. Mod. Phys. 36, 681–717 (1964).
    #
    # The structure factor is used to describe the coherent, elastic scattering
    # from a periodic structure. The full F_Q structure factor is the sum of each
    # F_Q_i*exp(-M_i) for each atom i inside the unit cell, where exp(-M_i) is the
    # Debye-Waller factor that takes into account the thermal vibrations, and the
    # F_Q_i is the structure factor of the element. Each element has one F_Q_i,
    # written as the sum of the contribution of each atom j of this element in the
    # unit cell, or:
    #
    # F_Q_i = f_i(Q, lambda) * sum(j=1, N) exp(i*2*pi*Q.dot(r_i_j))
    #
    # and the scattering amplitude of the individual atom i is:
    #
    # f_i(Q, lambda) = f_i + f'_i + i*f''_i 
    #
    # or 
    #
    # f_i(Q, lambda) = f0(Q) + f1(lambda) + if2(lambda)
    #
    # where f_i is the atomic form factor, f'_i and f''_i account for anomalous
    # dispersion and absorption.
    #
    #
    # Please note that f0 depends only on the Q vector, while f1 and f2
    # depend only on the photon energy.
    #
    # These parameters are fetched from xraydb. Another possibility is to use
    # the periodictable library, but it has some differences. In xraydb the f0
    # is given in terms of sin(theta)/lambda = 1/(2d_hkl), while in periodictable
    # it is given interms of 4pi/(2d_hkl). Another possibility is also to calculate
    # it, since it follows a simple rule (to be referenced here). The f1 and f2
    # parameters also present differences. The periodictable database uses the .nff
    # files from CXRO, but in these the f1 parameter has Z electrons already
    # included in the value.
    #
    # To be done:
    # 1. transform everything into dataframes. Ongoing.
    # 2. Add Debye-Waller factor
    def set_structure_factor(self, energy):
        self.xray_scattering_factors = pd.DataFrame([{
            'f0': xraydb.f0(site.Z, self.stol)[0] - site.Z,
            # 'f1': xraydb.f1_chantler(site.Z, energy) + site.Z,
            # 'f2': xraydb.f2_chantler(site.Z, energy),
            # 'f0_pt_Q': Xray(pt.elements[site.Z]).f0(4*np.pi*self.stol) - site.Z,
            'f1': Xray(pt.elements[site.Z]).scattering_factors(energy=energy/1000)[0],
            'f2': Xray(pt.elements[site.Z]).scattering_factors(energy=energy/1000)[1],
        } for idx, site in self.sites.iterrows()])
        
        self.xray_structure_factors = pd.DataFrame([{
            'F_0':   xsf.f1 + 1j*xsf.f2,
            'F_H':  (xsf.f0 + xsf.f1 + 1j*xsf.f2)*np.exp(-self.DWF)*np.exp( 2j*np.pi*self.sites.loc[idx, 'Hdotr']),
            'F_Hb': (xsf.f0 + xsf.f1 + 1j*xsf.f2)*np.exp(-self.DWF)*np.exp(-2j*np.pi*self.sites.loc[idx, 'Hdotr']),
        } for idx, xsf in self.xray_scattering_factors.iterrows()])
        
        self.F_0  = self.xray_structure_factors['F_0'].sum()
        self.F_H  = self.xray_structure_factors['F_H'].sum()
        self.F_Hb = self.xray_structure_factors['F_Hb'].sum()
        

    # Based on:
    # 1. J. Zegenhagen, Surface structure determination with X-ray standing waves. Surf. Sci. Rep. 18, 202–271 (1993).
    #
    #
    # E_H = sqrt(R)*E_0*exp(iP),
    # where R is the Reflectivity: R = |E_H|²/|E_0|²,
    # and P is the Phase P = v for Re(E_H/E_0) > 0, P = v + pi for Re(E_H/E_0) < 0
    # where v = arctan(Im(E_H/E_H)/Re(E_H/E_0))
    #
    # in the end, I = 1 + R + 2sqrt(R)cos(P - 2*pi*H.r)
    #
    # The important quantity here is E_H/E_0.
    #
    # E_H/E_0 = sqrt(Chi_H/Chi_Hb) [eta +- sqrt(eta² - 1)]
    #
    # Chi_0 and Chi_H are the 0th- and Hth-order Fourier components of the complex
    # lattice periodic susceptibility Chi(r), which can be written in terms of the
    # also complex structure factor for a reflection characterized by H:
    #
    # Chi_H = -Γ*F_H,
    # 
    # where Γ = (r_0*lambda²)/(pi*V),  r_0 is the classical electron radius, 
    # lambda is the wavelength and V is the unit cell volume. The same unit 
    # cell used to calculate the structure factor.
    #
    # Finally, the complex variable eta is:
    #
    # eta = a/sqrt(Chi_H*Chi_Hb)
    def calc_reflectivity(self, delta=20, npts=1001, Mono=False, gwidth=False,
                          mode = 'angular', angle=0.0, energy=0.0, polarization='sigma', DWF = 0.0):
        
        self.mode = mode
        self.DWF = DWF
        self.b = -1.0
        
        self.Bragg = AttrDict()
        if mode == 'angular':
            if energy == 0.0: # implementar erro
                energy = 3000

            self.Bragg.energy = energy

            self.Bragg.wavelength = self.wavelength_energy_relation(self.Bragg.energy)
            self.Bragg.angle  = np.degrees(np.arcsin(self.stol*self.Bragg.wavelength))
        elif mode == 'energy':
            if angle == 0.0:
                self.Bragg.energy = energy
                self.Bragg.wavelength = self.wavelength_energy_relation(self.Bragg.energy)
                self.Bragg.angle  = np.degrees(np.arcsin(self.stol*self.Bragg.wavelength))
            else:
                self.Bragg.angle = angle
                self.Bragg.wavelength = 2*self.d_hkl*np.sin(np.radians(self.Bragg.angle))
                self.Bragg.energy = self.wavelength_energy_relation(self.Bragg.wavelength)

        self.Bragg.angle_rad = np.radians(self.Bragg.angle)
        # these relations are correct        
        self.polarization = polarization 
        if self.polarization == 'pi':
            self.P = np.cos(2*self.Bragg.angle_rad)
        elif self.polarization == 'sigma':
            self.P = 1.0
            
        self.set_structure_factor(self.Bragg.energy)
        
        r0 = 2.8179403262e-15
        gamma = 1e10*(r0*self.wavelength_energy_relation(self.Bragg.energy)**2)/(np.pi*self.volume())
        self.Chi_0  = -gamma*self.F_0
        self.Chi_H  = -gamma*self.F_H
        self.Chi_Hb = -gamma*self.F_Hb
    
        # Extra info
        self.info = AttrDict()
        self.info.backscattering_angle = np.degrees(np.pi/2 - 2*np.abs(self.Chi_0))
        self.info.angle_range    = np.degrees(np.abs(self.P)*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))/(np.sqrt(np.abs(self.b))*np.sin(2*self.Bragg.angle_rad)))
        self.info.energy_range   = self.Bragg.energy*np.abs(self.P)*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))/(2*np.sin(self.Bragg.angle_rad)**2)
        self.info.extinct_length = (self.Bragg.wavelength*np.sin(self.Bragg.angle_rad)/(np.pi*np.sqrt(np.abs(self.Chi_H*self.Chi_Hb))))

        self.info.angle_width = 2*self.info.angle_range
        self.info.energy_width = 2*self.info.energy_range
        
        offset = -self.Chi_0*((1 - self.b)/2) 
        denominator = np.abs(self.P)*np.sqrt(np.abs(self.b)*self.Chi_H*self.Chi_Hb)
        
        self.info.angle_offset = np.degrees(offset.real / np.sin(2*self.Bragg.angle_rad))
        self.info.energy_offset = offset.real * self.Bragg.energy/(2*np.sin(self.Bragg.angle_rad)**2)

        if self.mode == 'angular':
            x_range = np.linspace(-delta*self.info.angle_range, delta*self.info.angle_range, npts) + self.info.angle_offset
            xarg = self.b*np.radians(x_range)*np.sin(2*self.Bragg.angle_rad)
        elif self.mode == 'energy':
            x_range = np.linspace(-delta*self.info.energy_range, delta*self.info.energy_range, npts) + self.info.energy_offset
            xarg = 2*self.b*(x_range/self.Bragg.energy)*(np.sin(self.Bragg.angle_rad))**2
            
        self.eta = (xarg + offset) / denominator
        
        # prefactor = np.sign(self.P)*(np.sign(self.b)/np.sqrt(np.abs(self.b)))*(np.sqrt(self.Chi_H*self.Chi_Hb)/self.Chi_Hb)
        prefactor = -np.sign(self.P)*np.sqrt(np.abs(self.b))*np.sqrt(self.F_H/self.F_Hb)
        # prefactor = np.sign(self.P)*(np.sign(self.b)/np.sqrt(np.abs(self.b)))*(np.sqrt(self.Chi_H/self.Chi_Hb))

        self.EH_E0 = prefactor*np.piecewise(self.eta, self.eta.real <= 0, 
                                [
                                    lambda x: x + np.sqrt(x**2 + np.sign(self.b)), 
                                    lambda x: x - np.sqrt(x**2 + np.sign(self.b))
                                ])
        
        self.x_range = x_range
        
        Refl = np.abs(self.EH_E0)**2
        Phase = np.angle(self.EH_E0)
        
        shift_mask = (Phase > (np.pi + np.angle(self.F_H))) | (Phase < np.angle(self.F_H))
        Phase[shift_mask] = np.remainder(Phase[shift_mask] + np.angle(self.F_H), 2*np.pi) - np.angle(self.F_H)
        
        self.Refl_unbroadened = Refl
        self.Phase_unbroadened = Phase

        self.Refl = Refl
        self.Phase = Phase
        self.Mono = Mono
        if (gwidth and gwidth > 0.0) and not self.Mono:
            gsmear = lmfit.lineshapes.gaussian(self.x_range, center = self.x_range.mean(), sigma = gwidth)
            gsmear = gsmear / sum(gsmear)
            
            self.Refl = np.convolve(Refl, gsmear, mode='same')
            self.Phase = np.convolve(Phase, gsmear, mode='same')
            
        # if Mono:
            
        #     Mono.calc_reflectivity(delta=delta, npts=npts)
            # Mono.Refl_shifted = np.interp(Mono.x_range, Mono.x_range, Mono.Refl_unbroadened)
            
            # if gwidth and gwidth > 0.0:
            #     gsmear = lmfit.lineshapes.gaussian(self.Mono.x_range, center = self.Mono.x_range.mean(), sigma = gwidth)
            #     gsmear = gsmear / sum(gsmear)
            #     self.Mono.Refl_shifted = np.convolve(self.Mono.Refl_shifted, gsmear, mode='same')

            # self.Mono.Squared_Refl = self.Mono.Refl_shifted**2/np.sum(self.Mono.Refl_shifted**2)

            # self.Refl_conv_Mono = np.convolve(self.Refl, self.Mono.Squared_Refl, mode='same')
            # self.Phase_conv_Mono = np.convolve(self.Phase, self.Mono.Squared_Refl, mode='same')
        # else:
        #     if gwidth and gwidth > 0.0:
                # gsmear = lmfit.lineshapes.gaussian(self.x_range, center = self.x_range.mean(), sigma = gwidth)
                # gsmear = gsmear / sum(gsmear)
                # self.Refl = np.convolve(self.Refl, gsmear, mode='same')
                # self.Phase = np.convolve(self.Phase, gsmear, mode='same')
    
    
        self.data = pd.DataFrame({'x': self.x_range, 'Refl': self.Refl, 'Phase': self.Phase})
        
    def calc_RC(self, CF, CP, Q = 0.0, Delta = 0.0):
        if self.Mono:
            R = self.Refl_conv_Mono
            P = self.Phase_conv_Mono
        else:
            R = self.Refl
            P = self.Phase

        return 1 + R + 2*np.sqrt(R)*CF*np.cos(P - 2*np.pi*CP)
    
    def calc_part_RC(self, CF, CP):
        if self.Mono:
            R = self.Refl_conv_Mono
            P = self.Phase_conv_Mono
        else:
            R = self.Refl
            P = self.Phase
        
        return 2*self.P*np.sqrt(R)*CF*np.cos(P - 2*np.pi*CP)
    
    def Electric_Field(self, z, zmin):
        zpos = z - zmin
        cohpos = np.remainder(zpos / self.d_hkl, 1)
        EF = self.calc_RC(1.0, cohpos)
        if z <= 0:
            EF *= np.exp(-np.abs(z)/self.extinct_length)
        return EF