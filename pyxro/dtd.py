# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:41 2019

@author: Henrique
"""

import periodictable as pt
import numpy as np
import mendeleev as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen import Structure
import lmfit

class Sample(object):
    def gaussian(sigma, x):
        g = lmfit.lineshapes.gaussian()
        g = scipy_norm.pdf(array, 0, sigma)
        return g/sum(g)

    def __init__(self, filename='', hkl=[1, 1, 1], nuc=1, structure=None):
        # Number of unit cells to consider. This is only useful when using the calc_yield() function
        self.number_of_ucs = nuc
        
        # Load crystal structure and gets the symmetrized structure
        if structure:
            self.structure = structure
        else:
            self.structure = Structure.from_file(filename)

        # Gets symmetrized structure: equivalent sites are separated
        self.symstruct = SpacegroupAnalyzer(self.structure).get_symmetrized_structure()

        # Debye-Waller factor
        self.DWF = 1.0
        
        # Calculates interplanar distance
        self.hkl = np.array(hkl)
        self.d_hkl = self.structure.lattice.d_hkl(self.hkl)
        self.stol = 1/(2*self.d_hkl)
        self.hkl_norm = self.hkl/np.sqrt(np.dot(self.hkl, self.hkl))
        self.z_dist = np.dot(self.hkl_norm, self.structure.lattice.abc)
    
        tmpa = np.random.randn(3)
        tmpa -= tmpa.dot(self.hkl_norm) * self.hkl_norm
        tmpa /= np.linalg.norm(tmpa)
        tmpb = np.cross(tmpa, self.hkl_norm)
        self.hkl_perp = (tmpa + tmpb)/np.linalg.norm(tmpa + tmpb)
    
        # Sets sites
        self.sites = {}
        self.zpos = {}
        self.geom_factor = {}
        self.cohpos = {}
        self.cohfra = {}
        for idx, site in enumerate(self.structure.sites):
            self.sites[idx] = site.specie.name + str(idx)
            self.zpos[idx] = np.dot(self.hkl_norm, site.coords)
            self.cohpos[idx] = np.remainder(self.zpos[idx] / self.d_hkl, 1)
            
        self.toplayer = None

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
        
    def calc_susceptibility(self, energy):
        F_0  = 0
        F_H  = 0
        F_Hb = 0

        r0 = 2.8179403262e-15
        gamma = 1e10*(r0*self.wavelength_energy_relation(energy)**2)/(np.pi*self.volume())
    
        # Debye-Waller correction to F_H and FH_Hb: exp(-Bs²), with B = 8*pi*<u**2>, s = sin(th)/lambda = 1/(2*d_hkl)
        # DW_correction = self.DWF/(2*self.d_hkl)**2
        
        for idx, site in enumerate(self.structure.sites):
            ptab = pt.elements[site.specie.number]
            
            f0 = ptab.xray.f0(4*np.pi/(2*self.d_hkl))
            f1, f2 = ptab.xray.scattering_factors(energy=energy/1000)
            prefac = (f0 - site.specie.number + f1 + 1j*f2)
            
            F_0  += (f1 + 1j*f2)
            F_H  += prefac*self.DWF*np.exp(2j*np.pi*self.cohpos[idx])
            F_Hb += prefac*self.DWF*np.exp(2j*np.pi*self.cohpos[idx])
        
        return gamma*F_0, gamma*F_H, gamma*F_Hb


    def calc_reflectivity(self, delta=20, npts=1001, Mono=False, gwidth=False):
        Chi_0, Chi_H, Chi_Hb = self.calc_susceptibility(self.energy_Bragg)
        
        if self.polarization == 'pi':
            P = 2*np.cos(np.radians(self.theta_Bragg))
        else:
            P = 1.0
            
        b = -1
        
        self.backscattering_angle = np.degrees(np.pi/2 - 2*np.abs(Chi_0))
        
        self.angle_offset = np.degrees(np.real(Chi_0)/np.sin(2*self.theta_Bragg_rad))
        self.angle_range  = np.degrees(np.abs(P)*np.sqrt(np.abs(Chi_H*Chi_Hb))/np.sin(2*self.theta_Bragg_rad))
        self.energy_offset = np.real(Chi_0)*self.energy_Bragg/(2*np.sin(self.theta_Bragg_rad)**2)
        self.energy_range  = self.energy_Bragg*np.abs(P)*np.sqrt(np.abs(Chi_H*Chi_Hb))/(2*np.sin(self.theta_Bragg_rad)**2)
        self.extinct_length = (self.lambda_Bragg*np.sin(self.theta_Bragg)/(np.pi*np.sqrt(np.abs(Chi_H*Chi_Hb))))
        
        if self.mode == 'angular':
            x_range = np.linspace(-delta*self.angle_range, delta*self.angle_range, npts)
            xarg = np.radians(x_range)
            targ = np.radians(self.theta_Bragg)
            if self.theta_Bragg < self.backscattering_angle:
                first_term = b*xarg*np.sin(2*targ)
            else:
                first_term = np.sqrt(1 + 2*xarg**2 * (np.sin(2*targ))**2 + 2*b*xarg*np.sin(2*targ)) - 1
        elif self.mode == 'energy':
            x_range = np.linspace(-delta*self.energy_range, delta*self.energy_range, npts)
            xarg = x_range/(x_range + self.energy_Bragg)
            targ = np.radians(self.theta_Bragg)
            first_term = 2*b*xarg*(np.sin(targ))**2
             
        eta = (first_term + ((1-b)/2)*Chi_0) / (np.absolute(P)*np.sqrt(np.absolute(b)*Chi_H*Chi_Hb))
    
        prefactor = np.sqrt(np.abs(b)) * (np.abs(P)/P) * np.sqrt(Chi_H/Chi_Hb)
        EHE0 = -prefactor*np.piecewise(eta, np.real(eta) > 0,
                 [lambda x: x-(x**2-1)**.5, lambda x: x+(x**2-1)**.5])
        
        Refl = np.abs(EHE0)**2
        Phase = np.remainder(np.angle(EHE0) + np.pi/2, 2*np.pi) - np.pi/2

        self.x_range = x_range
        self.Refl = Refl
        self.Phase = Phase
        self.Mono = Mono
        
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

        return 1 + S_R*R + 2*np.abs(S_I)*np.sqrt(R)*CF*np.cos(P - 2*np.pi*CP + Psi)
        
    def calc_sw_cohpos(self, CF, CP, Q = 0.0, Delta = 0.0):
        return self.calc_RC(CF, CP, Q, Delta)
    
    # def calc_yield(self, KE):
    #     lambda_IMFP = self.imfp() * np.sin(np.radians(self.theta_Bragg))

    #     Y = {}
    #     for idx, sites in enumerate(self.symstruct.equivalent_sites):
    #         tmpY = self.calc_sw_cohpos(self.cohfra[idx], self.cohpos[idx])
    #         Y[idx] = np.zeros((len(tmpY), self.number_of_ucs))
    #         for uc in np.arange(self.number_of_ucs):
    #             Y[idx][:, uc] = tmpY * np.exp(-uc*self.z_dist/lambda_IMFP)
    #         Y[idx] = Y[idx].sum(axis=1)/self.number_of_ucs
    #     self.Y = Y
        
    #     if self.toplayer != None:
    #         self.toplayer.Y = {}
    #         uniq_cohpos, uniq_idx = np.unique(list(self.toplayer.cohpos.values()), return_index=True)
    #         for idx in uniq_idx:
    #             name = self.toplayer.structure.sites[idx].specie.name + str(idx)
    #             self.toplayer.Y[name] = self.calc_sw_cohpos(self.toplayer.cohfra[idx], self.toplayer.cohpos[idx])

    def Electric_Field(self, z, zmin):
        zpos = z - zmin
        cohpos = np.remainder(zpos / self.d_hkl, 1)
        EF = self.calc_RC(1.0, cohpos)
        if z <= 0:
            EF *= np.exp(-np.abs(z)/self.extinct_length)
        return EF

    def set_toplayer(self, toplayer, distance):
        self.toplayer = toplayer
        self.toplayer.distance = distance
        self.toplayer.sites = {}
        self.toplayer.zpos = {}
        self.toplayer.geom_factor = {}
        self.toplayer.cohpos = {}
        self.toplayer.cohfra = {}
        
        all_pos = []
        for idx, site in enumerate(self.toplayer.structure.sites):
            all_pos = np.append(all_pos, np.dot(self.toplayer.hkl_norm, site.coords))
            
        for idx, site in enumerate(self.toplayer.structure.sites):
            self.toplayer.sites[idx] = site.specie.name + str(idx)
            self.toplayer.zpos[idx] = np.dot(self.toplayer.hkl_norm, site.coords) 
            self.toplayer.zpos[idx] -= all_pos.min()
            self.toplayer.zpos[idx] += self.toplayer.distance
            self.toplayer.geom_factor[idx] = np.exp(2j*np.pi*self.toplayer.zpos[idx]/self.d_hkl)
            self.toplayer.cohfra[idx] = np.abs(self.toplayer.geom_factor[idx])
            self.toplayer.cohpos[idx] = np.remainder(np.angle(self.toplayer.geom_factor[idx]) / (2*np.pi), 1)
    
    def print(self):
        if self.toplayer:
            print('Top layer:')
            for idx, site in self.toplayer.sites.items():
                print('{:>3}: {:>6.3f} {:>4.2}'.format(site, self.toplayer.zpos[idx], self.toplayer.cohpos[idx]))
        
        print('Sample/substrate:')
        for idx, site in self.sites.items():
            print('{:>2}: {:>6.3f} {:>4.2}'.format(site, self.zpos[idx], self.cohpos[idx]))
        

def plot_Electric_Field(Sample, colors=None, atoms=False, layers=False, shift=0.0):
    f, axl = plt.subplots(1, 2, sharey=True, figsize=(5,6), gridspec_kw=dict(width_ratios=[3, 1]))
    
    ax = axl[0]
    ax.minorticks_on()
    ax.tick_params(which='both', axis='y', direction='in', labelleft=True, left=True)

    if Sample.mode == 'energy':
        ax.set_xlabel(r'Relative photon energy (eV)')
        ax.set_title('E$_B$ = {:.2f} eV @ {:.2f}$\degree$'.format(Sample.energy_Bragg, Sample.theta_Bragg))
        offset = Sample.energy_offset
        trim   = 10*Sample.energy_range
    elif Sample.mode == 'angular':
        ax.set_xlabel(r'Relative incident angle ($\degree$)')
        ax.set_title(r'$\theta_B$ = {:.2f}$\degree$ @ {:.2f} eV'.format(Sample.theta_Bragg, Sample.energy_Bragg))
        offset = Sample.angle_offset
        trim   = 10*Sample.angle_range

    # Plots electric field
    z_range = np.linspace(Sample.d_hkl, -Sample.number_of_ucs*Sample.z_dist, 2001)
    ElectricField = np.zeros((len(z_range), len(Sample.x_range)))
    
    for idx, z in enumerate(z_range):
        ElectricField[idx, :] = Sample.Electric_Field(z, z_range.min())
    
    ttx, tty = np.meshgrid(Sample.x_range, z_range + shift)
    efplot = ax.pcolormesh(ttx, tty, ElectricField, cmap=cm.seismic)
    plt.colorbar(efplot)
    # ax.plot(Sample.x_range, Sample.Refl_conv_Mono, c='k')

    if layers:
        for i in np.arange(Sample.number_of_ucs*np.int(Sample.z_dist/Sample.d_hkl) + 1):
            ax.axhline(y = -i*Sample.d_hkl, c='k', lw=1.5, ls='--', zorder=1)    
        ax.axhline(y=0.0, color='k', lw=2.0, zorder=1)
    
    for i, site in enumerate(Sample.structure.sites):
        for k in np.arange(Sample.number_of_ucs):
            zpos = -(k+1)*Sample.z_dist + Sample.zpos[i]
            if colors:
                clr = colors[site.specie.name]
            else:
                clr = 'r'
                
            if zpos+shift > 0:
                zpos -= Sample.z_dist
                
            if atoms:
                axl[0].scatter(offset, zpos + shift, c=clr, edgecolor='w', s=100, zorder=2)

            CP = np.remainder(zpos / Sample.d_hkl, 1)
            SW = Sample.calc_RC(1.0, CP)
            axl[1].plot(Sample.x_range, SW - 1 + zpos + shift, c=clr)
            
    ax.set_ylabel('Depth ($\AA$)')
    axl[0].set_xlim(- trim + offset, trim + offset)
    axl[1].set_xlim(- trim + offset, trim + offset)
    ax.set_ylim(z_range.min() + shift, z_range.max() + Sample.d_hkl + shift)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.show()