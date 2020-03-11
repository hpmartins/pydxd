import numpy as np
import pandas as pd
import json
import re 
import copy
import periodictable as pt
import mendeleev as ml

class ParameterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.reset_index().to_json(orient='records'), object_pairs_hook=OrderedDict)
        return json.JSONEncoder.default(self, obj)

class MultilayerSample(object):
    column_names = ['Name', 'Formula', 'OPC', 'RepVal', 'RepCheck',
                    'Thickness', 'DiffType', 'DiffVal', 'OrbName',
                    'OrbACS', 'BE', 'IMFP', 'MolWeight', 'AtomZ',
                    'AtomN', 'Density', 'NVal', 'Gap', 'Flag',
                    'RepDiffType', 'RepDiffVal']

    column_types = {'Name':'str', 'Formula': 'str', 'OPC':'str', 'RepVal':'int', 'RepCheck':'int',
                    'Thickness':'float', 'DiffType':'int', 'DiffVal':'float', 'OrbName':'str',
                    'OrbACS':'str', 'BE':'float', 'IMFP':'float', 'MolWeight':'float',
                    'AtomZ':'int', 'AtomN':'float', 'Density':'float', 'NVal':'float', 'Gap':'float',
                    'Flag':'int', 'RepDiffType':'int', 'RepDiffVal':'float'}

    empty_layer = {
            "Name": "-",
            "Formula": "-",
            "OPC": "-",
            "RepVal": 1,
            "RepCheck": 0,
            "Thickness": 0.0,
            "DiffType": 1,
            "DiffVal": 0.0,
            "OrbName": "-",
            "OrbACS": "-",
            "BE": 0.0,
            "IMFP": -1.0,
            "MolWeight": 0.0,
            "AtomZ": 0,
            "AtomN": 0,
            "Density": 0.0,
            "NVal": 0,
            "Gap": 0.0,
            "Flag": 0,
            "RepDiffType": 1,
            "RepDiffVal": 0.0
    }

    def_vacuum = {
        'Thickness': 100.0,
        'DiffType': 0,
        'DiffVal':  0,
    }

    def __init__(self, name='ML Sample'):
        self.name = name
        self.calculation = {
            'Mode'        : 3,
            'Polarization': 2,
            'Order'       : np.array([3, 2, 1, 5, 4]),
            'ACSDir'      : 'ACS_Dir',
            'OPCDir'      : 'OPC_Dir',
            'SpotSizeArea': 0,
            'WIT'         : 0,
            'MCD'         : np.zeros(4),
            'IncAngle'    : np.zeros(5),
            'PhEnergy'    : np.zeros(5),
            'TakeOff'     : np.zeros(5),
            'Depth'       : np.zeros(5),
            'Wedge'       : np.zeros(5),
            'InBetween'   : np.array([1, 90]),
            'IntMesh'     : np.array([1, 60]),
        }
        self.vacuum    = {}
        self.layers    = {}
        self.substrate = {}
        self.optimizer = {}

    def set_calculation(self, mode = 'photoemission', angle = None, energy = None):
        if mode == 'photoemission':
            self.calculation['Mode'] = 3
            self.calculation['Order'] = np.array([3, 2, 1, 5, 4])
        
        if isinstance(angle, (int, float)):
            self.calculation['IncAngle'] = np.array([0, 0, 0, 1, angle])
        else:
            self.calculation['IncAngle'] = np.hstack((np.array(angle), [0, 0]))
        
        if isinstance(energy, (int, float)):
            self.calculation['PhEnergy'] = np.array([0, 0, 0, 1, energy])
        else:
            self.calculation['PhEnergy'] = np.hstack((np.array(energy), [0, 0]))
        
    def set_substrate(self, substrate_data = empty_layer):
        self.substrate = copy.deepcopy(self.empty_layer)
        
        for k,v in substrate_data.items():
            self.substrate[k] = v
        
        if self.substrate['Thickness'] == 0:
            self.substrate['Thickness'] = 100.0

    def set_vacuum(self, vacuum_data = def_vacuum):
        self.vacuum = copy.deepcopy(self.def_vacuum)
        
        for k,v in vacuum_data.items():
            self.vacuum[k] = v
        
        if self.vacuum['Thickness'] == 0:
            self.vacuum['Thickness'] = 100.0

    def add_layer(self, layer_data = empty_layer):
        new_layer = copy.deepcopy(self.empty_layer)
        for k,v in layer_data.items():
            new_layer[k] = v
            
        if len(self.layers) == 0:
            Layers = pd.DataFrame(new_layer, index=[0])
        else:
            pd_new_layer = pd.DataFrame(new_layer, index=[len(self.layers)])
            Layers = pd.concat([self.layers, pd_new_layer])

        self.layers = Layers[self.column_names].astype(self.column_types)

    def set_layers_info(self):
        for i, layer in self.layers.iterrows():
            formula = pt.formula(layer['Formula'])
            self.layers.loc[i, 'MolWeight'] = formula.mass
            self.layers.loc[i, 'NVal'] = np.sum([v*ml.element(str(k)).nvalence() for k,v in formula.atoms.items()])

    def from_json(self, data):
        tmp = json.loads(data)
        
        self.calculation = tmp['calculation']
        self.vacuum = tmp['vacuum']
        self.substrate = tmp['substrate']
        self.optimizer = tmp['optimizer']
        
        Layers = pd.DataFrame.from_dict(data=tmp['layers'])
        Layers = Layers[self.column_names]
        self.layers = Layers.astype(self.column_types)

    def to_json(self):
        tmp = {
            'calculation': self.calculation,
            'vacuum': self.vacuum,
            'layers': self.layers,
            'substrate': self.substrate,
            'optimizer': self.optimizer
        }
        return json.dumps(tmp, cls=ParameterJSONEncoder, indent=4)

    def to_yxrofile(self):
        C = copy.deepcopy(self.calculation)
        V = copy.deepcopy(self.vacuum)
        S = copy.deepcopy(self.substrate)
        L = copy.deepcopy(self.layers)

        if np.size(L, axis=0) < 13:
            for i in np.arange(13 - np.size(L, axis=0)):
                L = L.append(self.empty_layer, ignore_index=True)

        yxrofile  = ''
        yxrofile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['IncAngle'], C['Order'][0])
        yxrofile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['PhEnergy'], C['Order'][1])
        yxrofile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['TakeOff'],  C['Order'][2])
        yxrofile += '{}\t{}\n'.format(*C['InBetween'])
        yxrofile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['Depth'],    C['Order'][3])
        yxrofile += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(*C['Wedge'],    C['Order'][4])
        yxrofile += '{}\t{}\n'.format(*C['IntMesh'])

        tmp = {}
        for i in L.columns.to_list():
            if i == 'RepCheck':
                tmp[i] = '\t'.join([str(a) for a in L[:-1].loc[:, i]])
            else:
                tmp[i] = '\t'.join([str(a) for a in L.loc[:, i]])

        tmp['Name']      += '\t{}'.format(S['Name'])
        tmp['OPC']       += '\t{}'.format(S['OPC'])
        tmp['Thickness']  = '{}\t{}\t{}'.format(V['Thickness'], tmp['Thickness'], S['Thickness'])
        tmp['DiffType']   = '{}\t{}'.format(V['DiffType'], tmp['DiffType'])
        tmp['DiffVal']    = '{}\t{}'.format(V['DiffVal'], tmp['DiffVal'])
        tmp['OrbName']   += '\t{}'.format(S['OrbName'])
        tmp['OrbACS']    += '\t{}'.format(S['OrbACS'])
        tmp['BE']        += '\t{}'.format(S['BE'])
        tmp['IMFP']      += '\t{}'.format(S['IMFP'])
        tmp['MolWeight'] += '\t{}'.format(S['MolWeight'])
        tmp['AtomZ']     += '\t{}'.format(S['AtomZ'])
        tmp['AtomN']     += '\t{}'.format(S['AtomN'])
        tmp['Density']   += '\t{}'.format(S['Density'])
        tmp['NVal']      += '\t{}'.format(S['NVal'])
        tmp['Gap']       += '\t{}'.format(S['Gap'])
        tmp['Flag']      += '\t{}'.format(S['Flag'])

        for i in L.columns.to_list():
            if not i.startswith('RepDiff'):
                yxrofile += tmp[i] + '\n'

        yxrofile += '{}\t{}\t{}\n'.format(C['Mode'], C['Polarization'], C['WIT'])
        yxrofile += '{}\t{}\t{}\t{}\n'.format(*C['MCD'])
        yxrofile += '{}\n'.format(tmp['RepDiffType'])
        yxrofile += '{}\n'.format(tmp['RepDiffVal'])
        yxrofile += '{}\n'.format(C['ACSDir'])
        yxrofile += '{}\n'.format(C['OPCDir'])
        yxrofile += '{}\n'.format(C['SpotSizeArea'])

        return yxrofile

    def from_yxrofile(self, par_filename):
        with open(par_filename, 'r') as f:
            origdata = f.read()
        
        self.origdata = origdata
        
        data = re.sub("[\t]{2,}", "\t", origdata)
        data = re.sub(r"\r", "", data)
        data = re.sub(r"\t\n", "\n", data)
        data = np.array(data.split('\n'))

        Order = np.zeros(5)

        IncAngle_start,  IncAngle_step, IncAngle_end, IncAngle_fixed, IncAngle_val, Order[0] = data[0].split('\t')
        PhEnergy_start,  PhEnergy_step, PhEnergy_end, PhEnergy_fixed, PhEnergy_val, Order[1] = data[1].split('\t')
        TakeOff_start,   TakeOff_step,  TakeOff_end,  TakeOff_fixed,  TakeOff_val,  Order[2] = data[2].split('\t')
        InBetween_fixed, InBetween_val                                                       = data[3].split('\t')
        Depth_start,     Depth_step,    Depth_end,    Depth_fixed,    Depth_val,    Order[3] = data[4].split('\t')
        Wedge_start,     Wedge_step,    Wedge_end,    Wedge_fixed,    Wedge_val,    Order[4] = data[5].split('\t')
        IntMesh_checked, IntMesh_val                                                         = data[6].split('\t')
        L_Name           = np.array(data[ 7].split('\t'))
        L_OPC            = np.array(data[ 8].split('\t'))
        L_RepVal         = np.array(data[ 9].split('\t'))
        L_RepCheck       = np.append(data[10].split('\t'), [0])
        L_Thickness      = np.array(data[11].split('\t'))
        L_DiffType       = np.array(data[12].split('\t'))
        L_DiffVal        = np.array(data[13].split('\t'))
        L_OrbName        = np.array(data[14].split('\t'))
        L_OrbACS         = np.array(data[15].split('\t'))
        L_BE             = np.array(data[16].split('\t'))
        L_IMFP           = np.array(data[17].split('\t'))
        L_MolWeight      = np.array(data[18].split('\t'))
        L_AtomZ          = np.array(data[19].split('\t'))
        L_AtomN          = np.array(data[20].split('\t'))
        L_Density        = np.array(data[21].split('\t'))
        L_NVal           = np.array(data[22].split('\t'))
        L_Gap            = np.array(data[23].split('\t'))
        L_Flag           = np.array(data[24].split('\t'))
        Mode, Polar, WIT = np.array(data[25].split('\t'))
        MCD              = np.array(data[26].split('\t'))
        L_RepDiffType    = np.array(data[27].split('\t'))
        L_RepDiffVal     = np.array(data[28].split('\t'))
        ACS_Dir          = np.array(data[29])
        OPC_Dir          = np.array(data[30])
        SpotSizeArea     = np.array(data[31])

        IncAngle  = np.array([IncAngle_start,  IncAngle_step, IncAngle_end, IncAngle_fixed, IncAngle_val])
        PhEnergy  = np.array([PhEnergy_start,  PhEnergy_step, PhEnergy_end, PhEnergy_fixed, PhEnergy_val])
        TakeOff   = np.array([TakeOff_start,   TakeOff_step,  TakeOff_end,  TakeOff_fixed,  TakeOff_val])
        Depth     = np.array([Depth_start,     Depth_step,    Depth_end,    Depth_fixed,    Depth_val])
        Wedge     = np.array([Wedge_start,     Wedge_step,    Wedge_end,    Wedge_fixed,    Wedge_val])
        InBetween = np.array([InBetween_fixed, InBetween_val])
        IntMesh   = np.array([IntMesh_checked, IntMesh_val])

        # Separates Vacuum, Layers, and Substrate information
        L_Name,      S_Name                   = L_Name[:-1],      L_Name[-1]
        L_OPC,       S_OPC                    = L_OPC[:-1],       L_OPC[-1]
        V_Thickness, L_Thickness, S_Thickness = L_Thickness[0],   L_Thickness[1:-1],   L_Thickness[-1]
        V_DiffType,  L_DiffType               = L_DiffType[0],    L_DiffType[1:]
        V_DiffVal,   L_DiffVal                = L_DiffVal[0],     L_DiffVal[1:]
        L_OrbName,   S_OrbName                = L_OrbName[:-1],   L_OrbName[-1]
        L_OrbACS,    S_OrbACS                 = L_OrbACS[:-1],    L_OrbACS[-1]
        L_BE,        S_BE                     = L_BE[:-1],        L_BE[-1]
        L_IMFP,      S_IMFP                   = L_IMFP[:-1],      L_IMFP[-1]
        L_MolWeight, S_MolWeight              = L_MolWeight[:-1], L_MolWeight[-1]
        L_AtomZ,     S_AtomZ                  = L_AtomZ[:-1],     L_AtomZ[-1]
        L_AtomN,     S_AtomN                  = L_AtomN[:-1],     L_AtomN[-1]
        L_Density,   S_Density                = L_Density[:-1],   L_Density[-1]
        L_NVal,      S_NVal                   = L_NVal[:-1],      L_NVal[-1]
        L_Gap,       S_Gap                    = L_Gap[:-1],       L_Gap[-1]
        L_Flag,      S_Flag                   = L_Flag[:-1],      L_Flag[-1]

        L_RepVal = L_RepVal[:L_Name.size]
        L_RepCheck = L_RepCheck[:L_Name.size]

        # Layers
        column_data = {'Name': L_Name,
                       'OPC': L_OPC,
                       'RepVal': L_RepVal,
                       'RepCheck': L_RepCheck,
                       'Thickness': L_Thickness,
                       'DiffType': L_DiffType,
                       'DiffVal': L_DiffVal,
                       'RepDiffType': L_RepDiffType,
                       'RepDiffVal': L_RepDiffVal,
                       'OrbName': L_OrbName,
                       'OrbACS': L_OrbACS,
                       'BE': L_BE,
                       'IMFP': L_IMFP,
                       'MolWeight': L_MolWeight,
                       'AtomZ': L_AtomZ,
                       'AtomN': L_AtomN,
                       'Density': L_Density,
                       'NVal': L_NVal,
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
            'Thickness': float(V_Thickness),
            'DiffType' : int(V_DiffType),
            'DiffVal'  : float(V_DiffVal)
        }

        if np.isinf(Vacuum['Thickness']):
            Vacuum['Thickness'] = 100.0;

        # Substrate
        Substrate = {
            'Name'     : S_Name,
            'OPC'      : S_OPC,
            'Thickness': float(S_Thickness),
            'OrbName'  : S_OrbName,
            'OrbACS'   : S_OrbACS,
            'BE'       : float(S_BE),
            'IMFP'     : float(S_IMFP),
            'MolWeight': float(S_MolWeight),
            'AtomZ'    : float(S_AtomZ),
            'AtomN'    : float(S_AtomN),
            'Density'  : float(S_Density),
            'NVal'     : float(S_NVal),
            'Gap'      : float(S_Gap),
            'Flag'     : int(S_Flag)
        }

        if np.isinf(Substrate['Thickness']):
            Substrate['Thickness'] = 100.0;

        self.calculation: {
            'Mode'        : int(Mode),
            'Polarization': int(Polar),
            'Order'       : np.array([int(a) for a in Order]),
            'ACSDir'      : ACS_Dir,
            'OPCDir'      : OPC_Dir,
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
        }
        
        self.vacuum = Vacuum
        self.layers = Layers
        self.substrate = Substrate
        self.optimizer = {}

        return True
