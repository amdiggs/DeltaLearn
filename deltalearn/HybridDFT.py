import sys
import os
import re
import linecache as lc
import numpy as np
import math
import pdb
import json
import shutil
from scipy import constants
import h5py




ha2ryd = 2.0
ryd2ev = constants.physical_constants['Rydberg constant times hc in eV'][0]
ha2ev = ha2ryd * ryd2ev
bohr2ang = 1 / (constants.physical_constants['Bohr radius'][0] * 10**10)



def Mat3_V_Prod(M,vecs):
    new_vecs = []
    for v in vecs:
        nv = np.matmul(M,v)
        new_vecs.append(nv)
    return new_vecs

def Get_Data_From_JSON(file):
    if os.path.exists(file):
        print(f"The file '{file}' exists.")
        with open(file,mode='r',encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"The file '{file}' does not exist.")
        touch =open(file,'w')
        touch.close()
        return {}


def Write_JSON(data, file):
    with open(file,'w') as f:
        json.dump(data,f,indent=4)


def Check_Append(dic_lst,add_key,add_dat):
    for key in dic_lst:
        if(key == add_key):
            print(f"This Dict already contains {add_key}!")
            return dic_lst
        else:
            dic_lst[add_key] = add_dat
    return dic_lst

def Append_Data_to_JSON(path,dat_key,dat, file):
    Data = Get_Data_From_JSON(file)
    local_dict = Data[path]
    Data[path] = Check_Append(local_dict, dat_key, dat)

def Get_JDFTX_Ionpos(file):
    atoms = []
    for line in open(file).readlines():
        if(re.match(r'^\s*#',line)):
            continue
        vals = line.split()
        if(len(vals) == 0):
            continue
        at = {}
        at["type"] = vals[1]
        x = float(vals[2])
        y = float(vals[3])
        z = float(vals[4])
        at["Position"] = [x,y,z]
        atoms.append(at)
    return atoms


def Get_JDFTX_Lattice(file):
    lattice = []
    for line in open(file).readlines():
        if(re.match(r'^\s*lattice .*',line)):
            continue
        vals = line.split()
        if(len(vals) == 0):
            continue
        x = float(vals[0])
        y = float(vals[1])
        z = float(vals[2])
        lattice.append([x,y,z])
    return np.asarray(lattice)

#FillingsUpdate:  mu: -0.199368166  nElectrons: 266.000000  magneticMoment: [ Abs: 0.00038  Tot: -0.00027 ]

# magnetic-moments Co +3.001
def Get_JDFTX_Mag(file):
    mm = 0
    for line in open(file).readlines():
        if(re.match(r'^\s*FillingsUpdate:',line)):
            vals = line.split()
            mm = float(vals[-2])
            #print(line)
    return mm

# magnetic-moments Co +3.001
def Get_Atom_Mag(file):
    mm = 0
    for line in open(file).readlines():
        if(re.match(r'^\#\s*magnetic-moments',line)):
            vals = line.split()
            mm = float(vals[-1])
            #print(line)
    return mm

def Get_Energies(file,calc):
    Evdw = 0.0
    lines = open(file).readlines()
    if(calc == "pbe"):
        ret_dict = {"F": None, "Exc":None}
    elif(calc == "hse"):
        ret_dict = {"F": None, "Exc":None,"EXX": None}
    else:
        ValueError("clac type did not match pbe or hse")
    #start at the bottom of the output
    for line in reversed(lines):
        if(None in ret_dict.values()):
            if(re.search(r'F =',line)):
                vals = line.split()
                ret_dict["F"]=ha2ev*float(vals[-1])
            elif(re.search(r'Exc =',line)):
                vals = line.split()
                ret_dict["Exc"]=ha2ev*float(vals[-1])
            elif(re.search(r'EXX =',line)):
                vals = line.split()
                ret_dict["EXX"]=ha2ev*float(vals[-1])
        else:
            break
    return ret_dict

def Get_Energy_Dict(round, metal, adsorbate):
    calc_type = ["pbe","hse"]
    pbe_file = f"../Data/Prev_MNC/MN{round}/{metal}/{adsorbate}/No_bias/01/pbe.out"
    hse_file = f"../Data/Prev_MNC/MN{round}/{metal}/{adsorbate}/No_bias/01/hse.out"
    pbe_dict = Get_Energies(pbe_file, "pbe")
    hse_dict = Get_Energies(hse_file, "hse")
    ret_dict = {}
    ret_dict["Fpbe"] = pbe_dict["F"]
    ret_dict["Exc_pbe"] = pbe_dict["Exc"]
    ret_dict["Fhse"] = hse_dict["F"]
    ret_dict["Exc_hse"] = hse_dict["Exc"]
    ret_dict["Exx_hse"] = hse_dict["EXX"]
    ret_dict["DeltaE"] = hse_dict["F"] - pbe_dict["F"]
    ret_dict["DeltaExc"] = hse_dict["Exc"] + hse_dict["EXX"] - pbe_dict["Exc"]
    return ret_dict

def Get_Counts_Dict(poscar):
    ret_dict = {"Co": {"count":0}, "Cu":{"count":0}, "Fe":{"count":0}, "Mo":{"count":0}, "Ni":{"count":0}, "Pt":{"count":0}, "C":{"count":0}, "O":{"count":0}, "N":{"count":0}, "H":{"count":0}}
    lines = open(poscar).readlines()
    # poscar have line 6 looks like C N Fe N O
    types = lines[5].split()
    counts = lines[6].split()
    for t,c in zip(types,counts):
        ret_dict[t]['count'] = int(c)
    return ret_dict

def Write_Hybrid_JSON(json_file):
    _dict = {}
    rounds = ["2C", "3C", "4C"]
    metals = ["Co", "Cu", "Fe", "Mo", "Ni", "Pt"]
    adsorbates = ["CO", "CO2", "COOH", "H", "N2","N2H","NH3"]
    for round in rounds:
        for m in metals:
            for a in adsorbates:
                system = f"{m}-{round}-{a}"
                _dict[system] = {}
                _dict[system]["Energies"] = Get_Energy_Dict(round, m, a)
                poscar_file = f"../Data/Prev_MNC/MN{round}/{m}/{a}/No_bias/01/POSCAR"
                _dict[system]["Counts"] = Get_Counts_Dict(poscar_file)
    Write_JSON(_dict,json_file)


if __name__ == "__main__":
    file = "../Data/PBE-HSE-MNC.json"
    Write_Hybrid_JSON(file)











