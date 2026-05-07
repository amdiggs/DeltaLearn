from os.path import exists, isdir, isfile
import sys
import os
import re
import linecache as lc
import numpy as np
import math
import pdb
import json
import scipy.optimize as opt
from scipy import constants
import random
from BandProjections import Get_Average_Occupations
from RhoMatrix import Get_Rho_Trace
from RPA import Comp_RPA_Energy

EMPTY= re.compile(r"\s*\n")
INT=r"-?\d+"
FLOAT=r"-?\d+\.\d+"


Opt_File = ""

Data_Dir = "../Data/"

ha2ryd = 2.0
ryd2ev = constants.physical_constants['Rydberg constant times hc in eV'][0]
ha2ev = ha2ryd * ryd2ev
bohr2ang = 1 / (constants.physical_constants['Bohr radius'][0] * 10**10)


at_line = "{0:.0f}   {1}   {2:.8f}   {3:.8f}   {4:.8f}\n"

lmp_line = "{0:.0f}   {1:.0f}   {2:.8f}   {3:.8f}   {4:.8f}\n"

qe_line = "{0}   {1:.10f}   {2:.10f}   {3:.10f}\n"

neb_line = "{0:.0f}   {1:.10f}   {2:.10f}   {3:.10f}\n"

def line(x,m,b):
    return x*m + b

def Fit_Line(x,y):
    par, cov = opt.curve_fit(line,x,y)
    xd = np.arange(-0.001,np.max(x)+.001,0.001)
    yd = line(xd,*par)
    return xd,yd,par

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

def Write_Latex_Params(file,out_name):
    out = open(out_name,'w')
    txt = "\\hline\n{0} & {1} & {2}\\\\\n"
    lines = open(file).readlines()
    for i in range(0,len(lines),2):
        v1 = lines[i].split()
        v2 = lines[i+1].split()
        out.write(txt.format(v1[0],v1[1],v2[1]))
    out.close()

def Write_Latex_Error(file,out_name):
    out = open(out_name,'w')
    txt = "\\hline\n{0} & {1} & {2} & {3}\\\\\n"
    for line in open(file).readlines()[1:]:
        vals = line.split()
        out.write(txt.format(vals[0],vals[1],vals[3],vals[4]))
    out.close()

#rsync -av --max-size=50M src/ dest/
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


def Write_Optimized_Atomic_RPA_JSON(opt_file, json_file):
    Data = {}
    for line in open(opt_file).readlines():
        rpa_dict = {'Egga':0., 'Exc':0., 'Exx':0., 'Ecorr':0., 'DeltaE':0., 'Erpa':0.}
        vals = line.split()
        bulk_name = vals[0]
        rpa_dict['DeltaE'] = float(vals[1])
        Data[bulk_name] = rpa_dict
    Write_JSON(Data,json_file)


def Add_Atomic_RPA_JSON(at,mol_file, json_file):
    if(len(at) == 1):
        bulk_name = at.upper()
    else:
        bulk_name = f'{at}'
    epi_cut, Erpa, Exx, F, Exc = Read_RPA(mol_file)
    Data = Get_Data_From_JSON(json_file)
    Data[bulk_name]['Exc'] = Exc
    Write_JSON(Data,json_file)

def Get_DeltaE(prefix):
    file = prefix + "evals.day"
    F, Exc, Exx, Ec = 0.0, 0.0, 0.0, 0.0
    for line in open(file).readlines():
        if(re.match(r"^\s*F\s*.*$",line)):
           vals = line.split()
           F = float(vals[2])
        if(re.match(r"^\s*Exc\s*.*$",line)):
           vals = line.split()
           Exc = float(vals[2])
        if(re.match(r"^\s*Exx\s*.*$",line)):
           vals = line.split()
           Exx = float(vals[2])
        if(re.match(r"^\s*Ecorr\s*.*$",line)):
           vals = line.split()
           Ec = float(vals[2])
    DeltaE = (Exx + Ec) - Exc
    return DeltaE

def Write_Atomic_RPA_JSON(at,mol_file, json_file):
    rpa_dict = {'Egga':0., 'Exc':0., 'Exx':0., 'Ecorr':0., 'Erpa':0.}
    bulk_name = f'{at}'
    epi_cut, Erpa, Exx, F, Exc = Read_RPA(mol_file)
    rpa_dict['Egga'] = F
    rpa_dict['Exc'] = Exc
    rpa_dict['Exx'] = Exx
    epi_cut = np.power(epi_cut,-1.5)
    epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
    deltaE = Exc - (Exx + par[1])
    rpa_dict['Ecorr'] = par[1]
    Erpa = F - deltaE
    rpa_dict['Erpa'] = Erpa
    Data = Get_Data_From_JSON(json_file)
    Data[bulk_name]['Exc'] = Exc
    Write_JSON(Data,json_file)

def Write_Params(x,format):
    if(len(x) != len(format)):
        ValueError("len x != len format")
    out = open(Opt_File,'w')
    for a,v in zip(format,x):
        out.write(f"{a} {v}\n")
    out.close()
    #Print_Chi_sq(M,x,y)


def Get_Ion_Counts(file):
    typs, pos=mio.Get_JDFTX_Ionpos(file)
    counts = {}
    #pdb.set_trace()
    for t in typs:
        if(t in counts):
            counts[t] +=1
        else:
            counts[t] = 1
    return counts


def File_to_Matrix(file):
    mat = []
    lines = open(file).readlines()
    for line in lines:
        row = []
        vals = line.split()
        if(len(vals) == 0 ):
            continue
        for v in vals:
            row.append(float(v))
        mat.append(row)
    return np.asarray(mat)


def File_to_Vec(file, col):
    vec = []
    lines = open(file).readlines()
    for line in lines:
        vals = line.split()
        if(len(vals) == 0 ):
            continue
        vec.append(float(vals[col]))
    return np.asarray(vec)


def JSON_to_Matrix_Occupations(file, mat_list):
    counts_mat = []
    energy_vec = []
    ats = []
    data = Get_Data_From_JSON(file)
    out = open("atomic-matrix",'w')
    for m in mat_list:
        vals = data[m]
        energy_vec.append(vals["Energies"]["DeltaE"])
        counts = vals["Counts"]
        tmp = []
        line = ""
        for k,v in counts.items():
            tmp.append(v["count"])
            tmp.append(v["occupation"])
            ats.append(k)
            ats.append(k + "-occ")
        for val in tmp:
            line += f"{val} "
        line += "\n"
        out.write(line)
        counts_mat.append(tmp)
    out.close()
    return np.asarray(counts_mat), np.asarray(energy_vec), ats

def JSON_to_Matrix_RhoMat(file, mat_list):
    counts_mat = []
    energy_vec = []
    ats = []
    data = Get_Data_From_JSON(file)
    out = open("atomic-matrix",'w')
    for m in mat_list:
        vals = data[m]
        energy_vec.append(vals["Energies"]["DeltaE"])
        counts = vals["Counts"]
        tmp = []
        line = ""
        for k,v in counts.items():
            tmp.append(v["count"])
            tmp.append(v["tr_rho"])
            tmp.append(v["tr_diff"])
            ats.append(k)
            ats.append(k + "-tr_rho")
            ats.append(k + "-tr_diff")
        for val in tmp:
            line += f"{val} "
        line += "\n"
        out.write(line)
        counts_mat.append(tmp)
    out.close()
    return np.asarray(counts_mat), np.asarray(energy_vec), ats

def Leave_Ten_Out(working_dir):
    training = []
    leave = []
    slabs = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = [ "clean","CO", "CO2", "COOH", "H", "H2O", "N2", "O", "O2", "OH", "OOH"]
    for s in slabs:
        for a in molecs:
            training.append(f"{s}-{a}")
    for i in range(10):
        idx = random.randint(0, 98 - i)
        leave.append(training[idx])
        training.pop(idx)
    out = open(working_dir + "/training.dat",'w')
    for a in training:
        out.write(f"{a}\n")
    out.close()
    out = open(working_dir + "/leave.dat",'w')
    for a in leave:
        out.write(f"{a}\n")
    out.close()
    return training, leave

def Get_Training_Set(working_dir):
    mats = []
    leave = []
    if not (os.path.isfile(working_dir+"/training.dat")):
        Leave_Ten_Out(working_dir)
    for l in open(working_dir+"/training.dat").readlines():
        mats.append(l.split()[0])
    for l in open(working_dir+"/leave.dat").readlines():
        leave.append(l.split()[0])
    return mats, leave


def Write_RPA_Projections_JSON(out_file):
    _dict = {}
    rpa_line = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n"
    Dir = Data_Dir  + "Energies/"
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

def Write_RPA_RhoMat_JSON(out_file):
    #ret_dict[k] = {'count':count, 'tr_rho': tr, 'tr_diff': tr2}
    _dict = {}
    rpa_line = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n"
    Dir = Data_Dir  + "Energies/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    for m in mats:
        for a in molecs:
            counts = {
                    "Ag":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Cu":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Fe":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Co":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Mn":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Pt":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Ru":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "Ru":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "H":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "C":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "N":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                    "O":{"count": 0, "tr_rho": 0.0, "tr_diff": 0.0},
                      }
            material = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}/evals-full.dat"
            en_dict = Comp_RPA_Energy(En_file)
            traces = Get_Rho_Trace(m,a)
            for k,v in traces.items():
                counts[k] = v
            _dict[material] = {"Energies": en_dict, "Counts": counts}
    Write_JSON(_dict, out_file)


def Init(working_dir, params="rhomat"):
    Ats = ["Ag", "Cu", "Fe", "Co", "Mn", "Pt", "Ru", "H", "C", "N", "O"]
    if(params == "rhomat"):
        json_file = Data_Dir + "Energy_RhoMat.json"
    elif(params == "proj"):
        json_file = Data_Dir + "Energy_Proj.json"
    else:
        ValueError("params type did not match rhomat or proj")
    opt_file = working_dir + "/optimized_atomic_energies.dat"
    if not (os.path.isdir(working_dir)):
        os.mkdir(working_dir)
    if not (os.path.isfile(opt_file)):
        out = open(opt_file,'w')
        for a in Ats:
            if(len(a) == 2):
                out.write(f"{a} 10.000\n")
            else:
                out.write(f"{a} -1.000\n")
            out.write(f"{a}-tr_rho 1.000\n")
            out.write(f"{a}-tr_diff 1.000\n")
        out.close()
    mats, leave = Get_Training_Set(working_dir)
    if not(os.path.isfile(json_file)):
        if(params == "rhomat"):
            Write_RPA_RhoMat_JSON(json_file)
        elif(params == "proj"):
            Write_RPA_Projections_JSON(json_file)
    if(params == "rhomat"):
        M,Y,fmt = JSON_to_Matrix_RhoMat(json_file, mats)
    elif(params == "proj"):
        M,Y,fmt = JSON_to_Matrix_Occupations(json_file, mats)
    global Opt_File
    Opt_File = opt_file
    X = File_to_Vec(Opt_File,1)
    return M,X,Y,fmt




if __name__ == "__main__":
    opt_file = "Ridge_Regression/rand10_ridge.dat"
    err_file = "Ridge_Regression/leave-rand-error.txt"
    #main(opt_file)
    #Write_Latex_Error(err_file,"ridge_err_latex")
    #Write_Latex_Params(opt_file,"ridge_params")




