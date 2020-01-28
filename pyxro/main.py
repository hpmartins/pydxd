import numpy as np
import pandas as pd
import json
import re
import copy
from collections import OrderedDict

class ParameterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.reset_index().to_json(orient='records'), object_pairs_hook=OrderedDict)
        return json.JSONEncoder.default(self, obj)

class MultilayerSample(object):

    _defaults = {
    }

    column_names = ['Name', 'OptConstant', 'RepetitionVal', 'RepetitionCheck',
                    'Thickness', 'DiffusionType', 'DiffusionVal', 'OrbitalName',
                    'OrbitalFile', 'BindingEnergy', 'IMFP', 'MolWeight', 'AtomZ',
                    'NumberOfAtoms', 'Density', 'NValence', 'Gap', 'Flag',
                    'RepDiffusionType', 'RepDiffusionVal']

    column_types = {'Name':'str', 'OptConstant':'str', 'RepetitionVal':'int', 'RepetitionCheck':'int',
                    'Thickness':'float', 'DiffusionType':'int', 'DiffusionVal':'float', 'OrbitalName':'str',
                    'OrbitalFile':'str', 'BindingEnergy':'float', 'IMFP':'float', 'MolWeight':'float',
                    'AtomZ':'int', 'NumberOfAtoms':'int', 'Density':'float', 'NValence':'int', 'Gap':'float',
                    'Flag':'int', 'RepDiffusionType':'int', 'RepDiffusionVal':'float'}

    def __init__(self):
        self.__dict__.update(self._defaults)

    def from_json(self, data):
        self.parameters = data
        Layers = self.parameters['sample']['layers']
        Layers = pd.DataFrame.from_dict(data=Layers)
        Layers = Layers[self.column_names]
        Layers = Layers.astype(self.column_types)
        self.parameters['sample']['layers'] = Layers

    def to_json(self):
        tmp = copy.deepcopy(self.parameters)
        return json.dumps(tmp, cls=ParameterJSONEncoder, indent=4)

    def to_parfile(self):
        C = copy.deepcopy(self.parameters['calculation'])
        V = copy.deepcopy(self.parameters['sample']['vacuum'])
        S = copy.deepcopy(self.parameters['sample']['substrate'])
        L = copy.deepcopy(self.parameters['sample']['layers'])

        empty_layer = {
                "Name": "-",
                "OptConstant": "-",
                "RepetitionVal": 1,
                "RepetitionCheck": 0,
                "Thickness": 0.0,
                "DiffusionType": 1,
                "DiffusionVal": 0.0,
                "OrbitalName": "-",
                "OrbitalFile": "-",
                "BindingEnergy": 0.0,
                "IMFP": -1.0,
                "MolWeight": 0.0,
                "AtomZ": 0,
                "NumberOfAtoms": 0,
                "Density": 0.0,
                "NValence": 0,
                "Gap": 0.0,
                "Flag": 0,
                "RepDiffusionType": 1,
                "RepDiffusionVal": 0.0
        }

        if np.size(L, axis=0) < 13:
            for i in np.arange(13 - np.size(L, axis=0)):
                L = L.append(empty_layer, ignore_index=True)

        parfile  = ''
        parfile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['IncAngle'], C['CalcOrder'][0])
        parfile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['PhEnergy'], C['CalcOrder'][1])
        parfile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['TakeOff'],  C['CalcOrder'][2])
        parfile += '{}\t{}\n'.format(*C['InBetween'])
        parfile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['Depth'],    C['CalcOrder'][3])
        parfile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['Wedge'],    C['CalcOrder'][4])
        parfile += '{}\t{}\n'.format(*C['IntMesh'])

        tmp = {}
        for i in L.columns.to_list():
            if i == 'RepetitionCheck':
                tmp[i] = '\t'.join([str(a) for a in L[:-1].loc[:, i]])
            else:
                tmp[i] = '\t'.join([str(a) for a in L.loc[:, i]])

        tmp['Name'] += '\t{}'.format(S['Name'])
        tmp['OptConstant'] += '\t{}'.format(S['OptConstant'])
        tmp['Thickness'] = '{}\t{}\t{}'.format(V['Thickness'], tmp['Thickness'], S['Thickness'])
        tmp['DiffusionType'] = '{}\t{}'.format(V['DiffusionType'], tmp['DiffusionType'])
        tmp['DiffusionVal'] = '{}\t{}'.format(V['DiffusionVal'], tmp['DiffusionVal'])
        tmp['OrbitalName'] += '\t{}'.format(S['OrbitalName'])
        tmp['OrbitalFile'] += '\t{}'.format(S['OrbitalFile'])
        tmp['BindingEnergy'] += '\t{}'.format(S['BindingEnergy'])
        tmp['IMFP'] += '\t{}'.format(S['IMFP'])
        tmp['MolWeight'] += '\t{}'.format(S['MolWeight'])
        tmp['AtomZ'] += '\t{}'.format(S['AtomZ'])
        tmp['NumberOfAtoms'] += '\t{}'.format(S['NumberOfAtoms'])
        tmp['Density'] += '\t{}'.format(S['Density'])
        tmp['NValence'] += '\t{}'.format(S['NValence'])
        tmp['Gap'] += '\t{}'.format(S['Gap'])
        tmp['Flag'] += '\t{}'.format(S['Flag'])

        for i in L.columns.to_list():
            if not i.startswith('RepDiffusion'):
                parfile += tmp[i] + '\n'

        parfile += '{}\t{}\t{}\n'.format(C['Mode'], C['Polarization'], C['WIT'])
        parfile += '{}\t{}\t{}\t{}\n'.format(*C['MCD'])
        parfile += '{}\n'.format(tmp['RepDiffusionType'])
        parfile += '{}\n'.format(tmp['RepDiffusionVal'])
        parfile += '{}\n'.format(C['ACS'])
        parfile += '{}\n'.format(C['OPC'])
        parfile += '{}\n'.format(C['SpotSizeArea'])

        return parfile

    def from_parfile(self, par_filename):
        with open(par_filename, 'r') as f:
            data = f.read()
        self.from_par(data)

    def from_par(self, origdata):
        data = re.sub("[\t]{2,}", "\t", origdata)
        data = re.sub(r"\r", "", data)
        data = re.sub(r"\t\n", "\n", data)
        data = np.array(data.split('\n'))

        CalcOrder = np.zeros(5)

        IncAngle_start,  IncAngle_step, IncAngle_end, IncAngle_fixed, IncAngle_val, CalcOrder[0] = data[0].split('\t')
        PhEnergy_start,  PhEnergy_step, PhEnergy_end, PhEnergy_fixed, PhEnergy_val, CalcOrder[1] = data[1].split('\t')
        TakeOff_start,   TakeOff_step,  TakeOff_end,  TakeOff_fixed,  TakeOff_val,  CalcOrder[2] = data[2].split('\t')
        InBetween_fixed, InBetween_val                                                           = data[3].split('\t')
        Depth_start,     Depth_step,    Depth_end,    Depth_fixed,    Depth_val,    CalcOrder[3] = data[4].split('\t')
        Wedge_start,     Wedge_step,    Wedge_end,    Wedge_fixed,    Wedge_val,    CalcOrder[4] = data[5].split('\t')
        IntMesh_checked, IntMesh_val                                                             = data[6].split('\t')
        L_Name             = np.array(data[ 7].split('\t'))
        L_OptConstant      = np.array(data[ 8].split('\t'))
        L_RepetitionVal    = np.array(data[ 9].split('\t'))
        L_RepetitionCheck  = np.append(data[10].split('\t'), [0])
        L_Thickness        = np.array(data[11].split('\t'))
        L_DiffusionType    = np.array(data[12].split('\t'))
        L_DiffusionVal     = np.array(data[13].split('\t'))
        L_OrbitalName      = np.array(data[14].split('\t'))
        L_OrbitalFile      = np.array(data[15].split('\t'))
        L_BindingEnergy    = np.array(data[16].split('\t'))
        L_IMFP             = np.array(data[17].split('\t'))
        L_MolWeight        = np.array(data[18].split('\t'))
        L_AtomZ            = np.array(data[19].split('\t'))
        L_NumberOfAtoms    = np.array(data[20].split('\t'))
        L_Density          = np.array(data[21].split('\t'))
        L_NValence         = np.array(data[22].split('\t'))
        L_Gap              = np.array(data[23].split('\t'))
        L_Flag             = np.array(data[24].split('\t'))
        Mode, Polar, WIT   = np.array(data[25].split('\t'))
        MCD                = np.array(data[26].split('\t'))
        L_RepDiffusionType = np.array(data[27].split('\t'))
        L_RepDiffusionVal  = np.array(data[28].split('\t'))
        ACS_Dir            = np.array(data[29])
        OPC_Dir            = np.array(data[30])
        SpotSizeArea       = np.array(data[31])

        IncAngle  = np.array([IncAngle_start,  IncAngle_step, IncAngle_end, IncAngle_fixed, IncAngle_val])
        PhEnergy  = np.array([PhEnergy_start,  PhEnergy_step, PhEnergy_end, PhEnergy_fixed, PhEnergy_val])
        TakeOff   = np.array([TakeOff_start,   TakeOff_step,  TakeOff_end,  TakeOff_fixed,  TakeOff_val])
        Depth     = np.array([Depth_start,     Depth_step,    Depth_end,    Depth_fixed,    Depth_val])
        Wedge     = np.array([Wedge_start,     Wedge_step,    Wedge_end,    Wedge_fixed,    Wedge_val])
        InBetween = np.array([InBetween_fixed, InBetween_val])
        IntMesh   = np.array([IntMesh_checked, IntMesh_val])

        # Separates Vacuum, Layers, and Substrate information
        L_Name,          S_Name                   = L_Name[:-1],          L_Name[-1]
        L_OptConstant,   S_OptConstant            = L_OptConstant[:-1],   L_OptConstant[-1]
        V_Thickness,     L_Thickness, S_Thickness = L_Thickness[0],       L_Thickness[1:-1], L_Thickness[-1]
        V_DiffusionType, L_DiffusionType          = L_DiffusionType[0],   L_DiffusionType[1:]
        V_DiffusionVal,  L_DiffusionVal           = L_DiffusionVal[0],    L_DiffusionVal[1:]
        L_OrbitalName,   S_OrbitalName            = L_OrbitalName[:-1],   L_OrbitalName[-1]
        L_OrbitalFile,   S_OrbitalFile            = L_OrbitalFile[:-1],   L_OrbitalFile[-1]
        L_BindingEnergy, S_BindingEnergy          = L_BindingEnergy[:-1], L_BindingEnergy[-1]
        L_IMFP,          S_IMFP                   = L_IMFP[:-1],          L_IMFP[-1]
        L_MolWeight,     S_MolWeight              = L_MolWeight[:-1],     L_MolWeight[-1]
        L_AtomZ,         S_AtomZ                  = L_AtomZ[:-1],         L_AtomZ[-1]
        L_NumberOfAtoms, S_NumberOfAtoms          = L_NumberOfAtoms[:-1], L_NumberOfAtoms[-1]
        L_Density,       S_Density                = L_Density[:-1],       L_Density[-1]
        L_NValence,      S_NValence               = L_NValence[:-1],      L_NValence[-1]
        L_Gap,           S_Gap                    = L_Gap[:-1],           L_Gap[-1]
        L_Flag,          S_Flag                   = L_Flag[:-1],          L_Flag[-1]

        L_RepetitionVal = L_RepetitionVal[:L_Name.size]
        L_RepetitionCheck = L_RepetitionCheck[:L_Name.size]

        # Layers
        column_data = {'Name': L_Name,
                       'OptConstant': L_OptConstant,
                       'RepetitionVal': L_RepetitionVal,
                       'RepetitionCheck': L_RepetitionCheck,
                       'Thickness': L_Thickness,
                       'DiffusionType': L_DiffusionType,
                       'DiffusionVal': L_DiffusionVal,
                       'RepDiffusionType': L_RepDiffusionType,
                       'RepDiffusionVal': L_RepDiffusionVal,
                       'OrbitalName': L_OrbitalName,
                       'OrbitalFile': L_OrbitalFile,
                       'BindingEnergy': L_BindingEnergy,
                       'IMFP': L_IMFP,
                       'MolWeight': L_MolWeight,
                       'AtomZ': L_AtomZ,
                       'NumberOfAtoms': L_NumberOfAtoms,
                       'Density': L_Density,
                       'NValence': L_NValence,
                       'Gap': L_Gap,
                       'Flag': L_Flag,
        }

        for i in column_data.keys():
            if np.size(column_data[i]) != np.size(column_data['Name']):
                print('{} -> {}'.format(i, np.size(column_data[i])))
                print('{}'.format(column_data[i]))

        Layers = pd.DataFrame.from_dict(data=column_data)
        Layers = Layers[self.column_names]
        Layers.replace([np.inf, -np.inf, 'Inf', 'Infinity'], 0)
        Layers = Layers.astype(self.column_types)

        Layers = Layers[Layers['Name'] != 'NaN']
        Layers = Layers[Layers['Thickness'] != 0]


        # Vacuum
        Vacuum = {
            'Thickness'    : float(V_Thickness),
            'DiffusionType': int(V_DiffusionType),
            'DiffusionVal' : float(V_DiffusionVal)
        }

        if np.isinf(Vacuum['Thickness']):
            Vacuum['Thickness'] = 100.0;

        # Substrate
        Substrate = {
            'Name'         : S_Name,
            'OptConstant'  : S_OptConstant,
            'Thickness'    : float(S_Thickness),
            'OrbitalName'  : S_OrbitalName,
            'OrbitalFile'  : S_OrbitalFile,
            'BindingEnergy': float(S_BindingEnergy),
            'IMFP'         : float(S_IMFP),
            'MolWeight'    : float(S_MolWeight),
            'AtomZ'        : float(S_AtomZ),
            'NumberOfAtoms': float(S_NumberOfAtoms),
            'Density'      : float(S_Density),
            'NValence'     : float(S_NValence),
            'Gap'          : float(S_Gap),
            'Flag'         : int(S_Flag)
        }

        if np.isinf(Substrate['Thickness']):
            Substrate['Thickness'] = 100.0;

        self.parameters = {
            'calculation': {
                'Mode'        : int(Mode),
                'Polarization': int(Polar),
                'CalcOrder'   : np.array([int(a) for a in CalcOrder]),
                'ACS'         : ACS_Dir,
                'OPC'         : OPC_Dir,
                'SpotSizeArea': int(SpotSizeArea),
                'WIT'         : int(WIT),
                'MCD'         : np.array([int(a) for a in MCD]),
                'IncAngle'    : np.array([float(a) for a in IncAngle]),
                'PhEnergy'    : np.array([float(a) for a in PhEnergy]),
                'TakeOff'     : np.array([float(a) for a in TakeOff]),
                'Depth'       : np.array([float(a) for a in Depth]),
                'Wedge'       : np.array([float(a) for a in Wedge]),
                'InBetween'   : np.array([float(a) for a in InBetween]),
                'IntMesh'     : np.array([float(a) for a in IntMesh]),
            },
            'sample': {
                'vacuum'   : Vacuum,
                'layers'   : Layers,
                'substrate': Substrate,
            },
            'optimizer': {},
        }

        return True
