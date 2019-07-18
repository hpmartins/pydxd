import pyrxo.main as pm
import numpy as np

calc = pm.Main()

param = {
    'Experiment'         : 3,
    'Polarization'       : 2,
    'CalcOrder'          : np.array([3, 2, 1, 5, 4]),
    'IncidenceAngleScan' : np.array([0, 0.03, 8, 0, 45]),
    'PhotonEnergyScan'   : np.array([500, 5, 1000, 1, 4000]),
    'TakeOffAngleScan'   : np.array([10, 1, 90, 0, 45]),
    'InBetween'          : np.array([1, 60]),
    'DepthScan'          : np.array([-100, 0.5, 500, 0, 50]),
    'WedgeScan'          : np.array([10, 1, 100, 0, 50]),
    'IntegrationMesh'    : np.array([1, 40]),
    
    'Layers' : {
        0: {
            'Thickness': 100.0,
            'Diffusion': np.array([1, 0]),
        },
        
        1: {
            'Name': 'LaTiO3',
            'OpticalData': 'LaTiO3_3000-4500eV.txt',
            'Thickness': 11.0,
            'Repetition': np.array([10, 1]),
            'Diffusion': np.array([1, 0]),
            'RepeatDiffusion': np.array([1, 0]),
            'Orbitals': np.array(['La4d', 'Ti2p', 'Ti3p', 'O1s'])
        },

        2: {
            'Name': 'LaNiO3',
            'OpticalData': 'LaNiO3_3000-4500eV.txt',
            'Thickness': 10.0,
            'Repetition': np.array([10, 0]),
            'Diffusion': np.array([1, 0]),
            'RepeatDiffusion': np.array([1, 0]),
            'Orbitals': np.array(['La4d', 'Ni2p', 'Ni3p', 'O1s'])
        },
        
        99: {
            'Name': 'SrTiO3',
            'OpticalData': 'SrTiO3_3000-4500eV.txt',
            'Thickness': np.inf,
            'Diffusion': np.array([1, 0]),
            'Orbitals': np.array(['O1s'])
        }

    }
    
}

calc.load_param(param)

print(calc.params)