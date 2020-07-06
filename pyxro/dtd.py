# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:41 2019

@author: Henrique
"""


import periodictable as pt
import mendeleev as ml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen import Structure
import lmfit
import pandas as pd
import xraydb

class Sample(object):
    def __init__(self, filename='', hkl=[1, 1, 1], nuc=1, structure=None):
        # Load crystal structure and gets the symmetrized structure
        if structure:
            self.structure = structure
        else:
            self.structure = Structure.from_file(filename)
            
        # self.structure = self.structure.get_primitive_structure()
        # self.structure = SpacegroupAnalyzer(self.structure).get_symmetrized_structure()
        
        
        # Calculates interplanar distance
        self.hkl = np.array(hkl)
        self.d_hkl = self.structure.lattice.d_hkl(self.hkl)
        self.stol = 1/(2*self.d_hkl)
        self.H = self.structure.lattice.reciprocal_lattice_crystallographic.matrix.dot(self.hkl)

        #
        self.hkl_norm = self.hkl/np.sqrt(np.dot(self.hkl, self.hkl))
        self.z_dist = self.hkl_norm.dot(self.structure.lattice.abc)
    
        # Vector perpendicular to the (hkl) direction
        tmpa = np.random.randn(3)
        tmpa -= tmpa.dot(self.hkl_norm) * self.hkl_norm
        tmpa /= np.linalg.norm(tmpa)
        tmpb = np.cross(tmpa, self.hkl_norm)
        self.hkl_perp = (tmpa + tmpb)/np.linalg.norm(tmpa + tmpb)
    
        # Sets sites
        self.sites = pd.DataFrame(index = range(len(self.structure.sites)), columns = ['name', 'coords', 'geom_factor', 'cohpos'])
        for idx, site in enumerate(self.structure.sites):
            name = site.specie.name + str(idx)
            coords = self.H.dot(site.coords)
            cohpos = np.remainder(self.H.dot(site.coords), 1)
            geom_factor = np.exp(2j*np.pi*self.H.dot(site.coords))
            
            self.sites.iloc[idx, :] = [name, coords, geom_factor, cohpos]

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
        
    def set_mode(self, mode = 'angular', angle=0.0, energy=0.0, polarization='sigma'):
        self.mode = mode
        self.polarization = polarization
        
        if mode == 'angular':
            if energy == 0.0:
                # implementar erro
                self.energy_Bragg = 3000
            else:
                self.energy_Bragg = energy

            self.lambda_Bragg = self.wavelength_energy_relation(self.energy_Bragg)
            self.theta_Bragg  = np.degrees(np.arcsin(self.stol*self.lambda_Bragg))
        elif mode == 'energy':
            if angle == 0.0:
                self.energy_Bragg = energy
                self.lambda_Bragg = self.wavelength_energy_relation(self.energy_Bragg)
                self.theta_Bragg  = np.degrees(np.arcsin(self.stol*self.lambda_Bragg))
            else:
                self.theta_Bragg = angle
                self.lambda_Bragg = 2*self.d_hkl*np.sin(np.radians(self.theta_Bragg))
                self.energy_Bragg = self.wavelength_energy_relation(self.lambda_Bragg)

        self.theta_Bragg_rad = np.radians(self.theta_Bragg)
        self.set_structure_factor(self.energy_Bragg)
    
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
        F_0  = 0
        F_H  = 0
        F_Hb = 0

        self.SF = pd.DataFrame(index = range(len(self.structure.sites)), columns=[(0,0,0), tuple(self.hkl), tuple(-self.hkl)])
        self.scattering_factor = pd.DataFrame(index = range(len(self.structure.sites)), columns=['f0', 'f1', 'f2'])
        self.test = pd.DataFrame(index = range(len(self.structure.sites)), columns=['name', 'coords', 'frac_coords', 'gf', 'cp'])
        for idx, site in enumerate(self.structure.sites):
            f0 = xraydb.f0(site.specie.number, [0, self.stol])
            f1 = xraydb.f1_chantler(site.specie.number, energy)
            f2 = xraydb.f2_chantler(site.specie.number, energy)
            
            self.scattering_factor.iloc[idx, :] = [f0, f1, f2]
            
            F_0  += (f0[0] + f1 + 1j*f2)
            F_H  += (f0[1] + f1 + 1j*f2)*np.exp( 2j*np.pi*self.H.dot(site.coords))
            F_Hb += (f0[1] + f1 + 1j*f2)*np.exp(-2j*np.pi*self.H.dot(site.coords))
            
            gf = np.exp( 2j*np.pi*self.H.dot(site.coords))
            cp = np.remainder(self.H.dot(site.coords), 1)
            self.test.iloc[idx, :] = [site.species.formula, site.coords, site.frac_coords, gf, cp]

        self.F_0  = F_0
        self.F_H  = F_H
        self.F_Hb = F_Hb
        
        for idx, row in self.scattering_factor.iterrows():
            tmp0 = row['f0'][0] + row['f1'] + 1j*row['f2']
            tmpH = row['f0'][1] + row['f1'] + 1j*row['f2']
            Hdr = self.H.dot(self.structure.sites[idx].coords)
            self.SF.iloc[idx, :] = [tmp0, tmpH*np.exp(2j*np.pi*Hdr), tmpH*np.exp(-2j*np.pi*Hdr)]


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
    # E_H/E_0 = -sqrt(F_H/F_Hb) [eta +- sqrt(eta² - 1)]
    #
    # Chi_0 and Chi_H are the 0th- and Hth-order Fourier components of the complex,
    # lattice periodic susceptibility Chi(r). These Fourier components are related
    # to the structure factor by a prefactor Gamma = (r_0*lambda²)/(pi*V), where r_0
    # is the classical electron radius, lambda is the wavelength and V is the unit cell
    # volume. The same unit cell used to calculate the structure factor.
    #
    # Finally, the complex variable eta is:
    #
    # eta = a/sqrt(Chi_H*Chi_Hb)
    def calc_reflectivity(self, delta=20, npts=1001, Mono=False, gwidth=False):
        if self.polarization == 'pi':
            P = 2*np.cos(np.radians(self.theta_Bragg))
        else:
            P = 1.0
            
        r0 = 2.8179403262e-15
        gamma = 1e10*(r0*self.wavelength_energy_relation(self.energy_Bragg)**2)/(np.pi*self.volume())
        Chi_0  = gamma*self.F_0
        Chi_H  = gamma*self.F_H
        Chi_Hb = gamma*self.F_Hb

        self.backscattering_angle = np.degrees(np.pi/2 - 2*np.abs(Chi_0))
        self.angle_offset   = np.degrees(np.real(Chi_0)/np.sin(2*self.theta_Bragg_rad))
        self.angle_range    = np.degrees(np.abs(P)*np.sqrt(np.abs(Chi_H*Chi_Hb))/np.sin(2*self.theta_Bragg_rad))
        self.energy_offset  = np.real(Chi_0)*self.energy_Bragg/(2*np.sin(self.theta_Bragg_rad)**2)
        self.energy_range   = self.energy_Bragg*np.abs(P)*np.sqrt(np.abs(Chi_H*Chi_Hb))/(2*np.sin(self.theta_Bragg_rad)**2)
        self.extinct_length = (self.lambda_Bragg*np.sin(self.theta_Bragg)/(np.pi*np.sqrt(np.abs(Chi_H*Chi_Hb))))
        
        if self.mode == 'angular':
            # a = (tB - t)*sin(2tB) - Chi_0
            x_range = np.linspace(-delta*self.angle_range, delta*self.angle_range, npts) # t - tB
            xarg = np.radians(x_range)  # (t - tB)
            targ = np.radians(self.theta_Bragg) # 2tB
            a  = xarg*np.sin(2*targ) - Chi_0
        elif self.mode == 'energy':
            x_range = np.linspace(-delta*self.energy_range, delta*self.energy_range, npts) # deltaE
            xarg = x_range/self.energy_Bragg # deltaE/E_B
            targ = np.radians(self.theta_Bragg) # tB
            a =  2*xarg*(np.sin(targ))**2 - Chi_0 # -2(deltaE/E_B)sin²(tB) 
             
        eta = a / np.sqrt(Chi_H*Chi_Hb)
            
        EHE0 = np.sqrt(Chi_H/Chi_Hb)*np.piecewise(eta, np.real(eta) > 0,
                 [lambda x: x - np.sqrt(x**2 - 1), lambda x: x + np.sqrt(x**2 - 1)])
        
        Refl = np.abs(EHE0)**2
        Phase = np.remainder(np.angle(EHE0) + np.pi/2, 2*np.pi) - np.pi/2

        self.x_range = x_range
        self.Refl = Refl
        self.Phase = Phase
        self.Mono = Mono
        
        # EH_E0_pos = np.sqrt(Chi_H/Chi_Hb)*(eta + np.sqrt(eta**2 - 1))
        # EH_E0_neg = np.sqrt(Chi_H/Chi_Hb)*(eta - np.sqrt(eta**2 - 1))
        
        # Refl_pos = np.abs(EH_E0_pos)**2
        # Refl_neg = np.abs(EH_E0_neg)**2
        
        # phi_pos = np.arctan(np.imag(EH_E0_pos) / np.real(EH_E0_pos))
        # phi_neg = np.arctan(np.imag(EH_E0_neg) / np.real(EH_E0_neg))
        
        # Refl = np.zeros_like(Refl_pos)
        # Phase = np.zeros_like(Refl_pos)
        
        # Refl[ (Refl_pos > 0) & (Refl_pos < 1)] = Refl_pos[(Refl_pos > 0) & (Refl_pos < 1)]
        # Refl[ (Refl_neg > 0) & (Refl_neg < 1)] = Refl_neg[(Refl_neg > 0) & (Refl_neg < 1)]
        # Phase[(Refl_pos > 0) & (Refl_pos < 1)] = phi_pos[ (Refl_pos > 0) & (Refl_pos < 1)]
        # Phase[(Refl_neg > 0) & (Refl_neg < 1)] = phi_neg[ (Refl_neg > 0) & (Refl_neg < 1)]
        
        # if self.polarization == 'sigma':
        #     Phase[np.real(EH_E0_neg) < 0] += np.pi
        # else:
        #     Phase[np.real(EH_E0_neg) > 0] += np.pi
        
        # Convolutes with monochromator R
        if self.Mono:
            self.Mono.set_mode(self.mode, energy=self.energy_Bragg)
            self.Mono.calc_reflectivity(delta=delta, npts=npts)
            
            shift_mono = (self.Mono.angle_offset if self.mode == 'angular' else self.Mono.energy_offset)
            self.Mono.Refl_shifted = np.interp(self.Mono.x_range, self.Mono.x_range - shift_mono, self.Mono.Refl)
            
            if gwidth and gwidth > 0.0:
                gsmear = lmfit.lineshapes.gaussian(self.Mono.x_range, center = self.Mono.x_range.mean(), sigma = gwidth)
                gsmear = gsmear / sum(gsmear)
                self.Mono.Refl_shifted = np.convolve(self.Mono.Refl_shifted, gsmear, mode='same')

            self.Mono.Squared_Refl = self.Mono.Refl_shifted**2/np.sum(self.Mono.Refl_shifted**2)

            self.Refl_conv_Mono = np.convolve(self.Refl, self.Mono.Squared_Refl, mode='same')
            self.Phase_conv_Mono = np.convolve(self.Phase, self.Mono.Squared_Refl, mode='same')
        else:
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
            
        S_R = (1 + Q)/(1 - Q)
        Psi = np.arctan(Q*np.tan(np.radians(Delta)))
        S_I = ((S_R + 1)/2)*np.sqrt(1 + (np.tan(Psi))**2)

        return 1 + R + 2*np.sqrt(R)*CF*np.cos(P - 2*np.pi*CP)
    
    
    def Electric_Field(self, z, zmin):
        zpos = z - zmin
        cohpos = np.remainder(zpos / self.d_hkl, 1)
        EF = self.calc_RC(1.0, cohpos)
        if z <= 0:
            EF *= np.exp(-np.abs(z)/self.extinct_length)
        return EF
