import sys
import os
import re
import numpy as np
import math
import pdb
import json
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import constants
import numpy as np
from ase.units import Hartree
import BandProjections as bp
#import MatFileio as mio

ha2ryd = 2.0
ryd2ev = constants.physical_constants['Rydberg constant times hc in eV'][0]
ha2ev = ha2ryd * ryd2ev
bohr2ang = 1 / (constants.physical_constants['Bohr radius'][0] * 10**10)

lat_re = re.compile(r'\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+.*')

atom_energies = "JSON/Atom_RPA_Energy-nospin.json"
def A_to_B(num):
    return num*1.88973

def B_to_A(num):
    return num/1.88973

def line(x,m,b):
    return x*m + b

def Fit_Line(x,y):
    par, cov = opt.curve_fit(line,x,y)
    xd = np.arange(-0.001,np.max(x)+.001,0.001)
    yd = line(xd,*par)
    return xd,yd,par

def Get_Data_From_JSON(file):
    with open(file,mode='r',encoding='utf-8') as f:
        return json.load(f)

def Write_JSON(data, file):
    with open(file,'w') as f:
        json.dump(data,f,indent=4)

def Check_Append(dic_lst,add_key,add_dat):
    for key in dic_lst:
        if(key == add_key):
            print(f"This Dict already contains {add_key}!")
            return dic_lst
    print(f"Added {add_key}\n")
    dic_lst[add_key] = add_dat
    return dic_lst


def Append_Data_to_JSON_Field(field_key,dat_key,dat, json_file):
    Data = Get_Data_From_JSON(json_file)
    local_dict = Data[field_key]
    Data[field_key] = Check_Append(local_dict, dat_key, dat)

def Append_Bands_to_JSON(bulk_mat,adsorbate,dat, json_file):
    Data = Get_Data_From_JSON(json_file)
    mat_dict = Data[bulk_mat]
    for d in mat_dict:
        if(d["Adsorbate"] == adsorbate):
            d["Band_Projections"] = dat
            break
    Data[bulk_mat] = mat_dict
    Write_JSON(Data,json_file)



def Comp_Molecules(dic, file):
    rpa_dict = {}
    Data = Get_Data_From_JSON(file)
    F, Exc, Exx, Ec, deltaE, Erpa = 0.,0.,0.,0.,0.,0.
    FDD, ExcDD, ExxDD, EcDD, deltaEDD, ErpaDD = 0.,0.,0.,0.,0.,0.
    for k,v in dic.items():
        F+=Data[k]['Egga']*v
        Exc+=Data[k]['Exc']*v
        Exx+=Data[k]['Exx']*v
        Ec+=Data[k]['Ecorr']*v
        deltaE+=Data[k]['DeltaE']*v
        Erpa+=Data[k]['Erpa']*v
        FDD+=Data[k]['EggaDD']*v
        ExcDD+=Data[k]['ExcDD']*v
        ExxDD+=Data[k]['ExxDD']*v
        EcDD+=Data[k]['EcorrDD']*v
        deltaEDD+=Data[k]['DeltaEDD']*v
        ErpaDD+=Data[k]['ErpaDD']*v
    rpa_dict['Egga'] = F
    rpa_dict['Exc'] = Exc
    rpa_dict['Exx'] = Exx
    rpa_dict['Ecorr'] = Ec
    rpa_dict['DeltaE'] = deltaE
    rpa_dict['Erpa'] = Erpa
    rpa_dict['EggaDD'] = FDD
    rpa_dict['ExcDD'] = ExcDD
    rpa_dict['ExxDD'] = ExxDD
    rpa_dict['EcorrDD'] = EcDD
    rpa_dict['DeltaEDD'] = deltaEDD
    rpa_dict['ErpaDD'] = ErpaDD
    return rpa_dict



def molecs():
    molecs = {"CO":{"co":1.0} , "CO2":{"co2":1.0} , "COOH":{"co2":1.0, "h2":0.5} , "H":{"h2":0.5} , "H2O":{"h2o":1.0} , "N2":{"n2":1.0} , "O":{"o2":0.5} , "O2":{"o2":1.0} , "OH":{"h2o":1.0, "h2":-0.5} , "OOH":{"o2":1.0, "h2":0.5}}
    in_file = f"base_molecs.json"
    out_file = f"Molecules-RPA.json"
    out = open(out_file, 'w')
    out.write("{}")
    out.close()
    for k,v in molecs.items():
        Data = Get_Data_From_JSON(out_file)
        Data[k] = Comp_Molecules(v,in_file)
        Write_JSON(Data,out_file)

def Molecules():
    molecs = ["co", "co2", "h2", "h2o", "n2", "o2"]
    Dir = "out-Molecules-RPA"
    out_file = f"base_molecs.json"
    out = open(out_file, 'w')
    out.write("{}")
    out.close()
    for l in molecs:
        En_file = Dir + f"/{l}/{l}-evals.dat" 
        rpa_dict = Comp_RPA_Energy(En_file)
        Data = Get_Data_From_JSON(out_file)
        Data[l] = rpa_dict
        Write_JSON(Data,out_file)




def Get_Vec_Brackets(line):
    ret = []
    get = False
    sub_str = ""
    for c in line:
        if(get):
            if(re.match(']',c)):
                break
            else:
                sub_str+= c
        else:
            if(re.match("[",c)):
                get = True
    vals = sub_str.split()
    for v in vals:
        ret.append(v)
    return ret



def Get_Structure(metal,MI,ads):
    ion_file = f"out-metals-111-RPA/{metal}-{MI}/{metal}-{MI}-{ads}.ionpos" 
    latt_file = f"out-metals-111-RPA/{metal}-{MI}/{metal}-{MI}-{ads}.lattice" 
    latt = mio.Get_JDFTX_Lattice_Mat3_Ang(latt_file)
    atoms = mio.Get_JDFTX_Ionpos(ion_file)
    return latt ,atoms

def Get_EF(file):
    for line in open(file).readlines():
        if(re.search(r"LUMO:",line)):
            vals = line.split()
            LUMO = ha2ev*float(vals[1])
        elif(re.search(r'HOMO:',line)):
            vals = line.split()
            HOMO = ha2ev*float(vals[1])
    EF = (HOMO + LUMO)/2.
    return EF


def Get_Eigenvalues(EF_file,file, kpts, bands):
    EF = Get_EF(EF_file)
    print(EF)
    mat=np.fromfile(file)
    matrix=mat.reshape(kpts,bands)
    matrix*=ha2ev
    print(np.min(mat))
    print(np.max(mat))
    matrix-=EF
    print(np.min(mat))
    print(np.max(mat))
    mat_lst = []
    for r in matrix:
        tmp = []
        for c in r:
            tmp.append(c)
        mat_lst.append(tmp)
    return mat_lst


def Comp_RPA_Energy(file):
    rpa_dict = {'Egga':0., 'Exc':0., 'Exx':0., 'Ecorr':0., 'DeltaE':0., 'Erpa':0.}
    Erpa = []
    epi_cut = []
    Evdw = 0.0
    SP = False
    for line in open(file).readlines():
        if(SP):
            if(re.search(r"RPA energy",line)):
                vals = line.split()
                Erpa.append(ha2ev*float(vals[-1]))
                epi_cut.append(float(vals[0]))
            elif(re.search(r'EXX\(RPA\) =',line)):
                vals = line.split()
                Exx=ha2ev*float(vals[-1])
            elif(re.search(r'F =',line) or re.search(r'Etot =',line)):
                vals = line.split()
                F=ha2ev*float(vals[-1])
            elif(re.search(r'Exc =',line)):
                vals = line.split()
                Exc=ha2ev*float(vals[-1])
            elif(re.search(r'EvdW =',line)):
                vals = line.split()
                Evdw=ha2ev*float(vals[-1])
        elif(re.search(r'#+SP#+',line)):
            SP=True
            continue
    if(not Evdw == 0):
        print(f"EVDW = {Evdw}\n")
        F-=Evdw
    if(len(Erpa) < 6):
        print(file)
        print(len(Erpa))
    epi_cut = np.power(epi_cut,-1.5)
    epi_inf, Erpa_inf, par = Fit_Line(epi_cut,np.asarray(Erpa))
    Ec = par[1]
    dE = (Exx + Ec) - Exc
    Erpa = F + dE
    rpa_dict['F'] = F/ha2ev
    rpa_dict['Egga'] = F
    rpa_dict['Exc'] = Exc
    rpa_dict['Exx'] = Exx
    rpa_dict['Ecorr'] = par[1]
    rpa_dict['DeltaE'] = dE
    rpa_dict['Erpa'] = Erpa
    return rpa_dict

def Get_Ion_Counts(file):
    #typs, pos=mio.Get_JDFTX_Ionpos(file)
    counts = {}
    #pdb.set_trace()
    for t in typs:
        if(t in counts):
            counts[t] +=1
        else:
            counts[t] = 1
    return counts


def Write_RPA_Data_JSON(out_file):
    _dict = {}
    rpa_line = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n"
    Dir = "OUT/out-RPA-OER-1/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    for m in mats:
        for a in molecs:
            counts = {
                    "Ag":{"count": 0, "occupation": 0.0},
                    "Cu":{"count": 0, "occupation": 0.0},
                    "Fe":{"count": 0, "occupation": 0.0},
                    "Co":{"count": 0, "occupation": 0.0},
                    "Mn":{"count": 0, "occupation": 0.0},
                    "Pt":{"count": 0, "occupation": 0.0},
                    "Ru":{"count": 0, "occupation": 0.0},
                    "H":{"count": 0, "occupation": 0.0},
                    "C":{"count": 0, "occupation": 0.0},
                    "N":{"count": 0, "occupation": 0.0},
                    "O":{"count": 0, "occupation": 0.0}
                      }
            material = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}/evals-full.dat"
            prefix = f"RPA-BandProj/{m}/{m}-{a}/"
            en_dict = Comp_RPA_Energy(En_file)
            atom_occupations = bp.Get_Full_Occupations(prefix)
            for k,v in atom_occupations.items():
                counts[k] = v
            _dict[material] = {"Energies": en_dict, "Counts": counts}
    Write_JSON(_dict, out_file)


def F_check(out_file):
    Dir = "RPA-BandProj/"
    mats = ["Ag-111", "Cu-111", "Pt-111", "Ru-001","AgNC", "CuNC", "FeNC", "CoNC", "MnNC" ]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    out = open(out_file,'w')
    for m in mats:
        for a in molecs:
            material = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}/sp.Ecomponents"
            for line in open(En_file).readlines():
                if(re.match(r'^\s+F =.*$',line)):
                   vals = line.split()
                   F = float(vals[-1])
            out.write(f"{m}-{a} {F}\n")
    out.close()

if __name__ == "__main__":
    #Write_RPA_Data_JSON("rpa_delta_learning.json")
    F_check("dftU_f_check.txt")




