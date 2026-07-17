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
import pickle
from . import BandProjections
from . import RhoMatrix
from . import RPA

EMPTY= re.compile(r"\s*\n")
INT=r"-?\d+"
FLOAT=r"-?\d+\.\d+"


Opt_File = ""

Data_Dir = "./Data/"

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



class LearningData:
    def __init__(self, json_file, opt_file) -> None:
        self.row_list = []
        self.col_list = None
        self.matrix = None
        self.E_vec = None
        self.params_vec = None
        self.opt_file = opt_file
        self.Init_JSON(json_file)
        self.Read_Opt_File(opt_file)

    def Init_JSON(self,file):
        counts_mat = []
        energy_vec = []
        ats = []
        get_ats = True
        data = Get_Data_From_JSON(file)
        #loop through materials
        for material, dat in data.items():
            self.row_list.append(material)
            energy_vec.append(dat["Energies"]["DeltaE"])
            counts = dat["Counts"]
            tmp = []
            #loop through atoms for a given mat
            ats = []
            for at,vals in counts.items():
                for k, v in vals.items():
                    tmp.append(v)
                    if(get_ats):
                        ats.append(f"{at}:{k}")
            counts_mat.append(tmp)
        self.matrix = np.asarray(counts_mat)
        self.E_vec = np.asarray(energy_vec)
        self.col_list = ats

    def Read_Opt_File(self,file):
        vec = []
        if(os.path.isfile(file)):
            lines = open(file).readlines()
            for line in lines:
                vals = line.split()
                if(len(vals) == 0 ):
                    continue
                vec.append(float(vals[1]))
        else:
            for lab in self.col_list:
                vec.append(0.0)
        self.params_vec = np.asarray(vec)

    def Write_Params(self):
        out = open(self.opt_file,'w')
        for a,v in zip(self.col_list, self.params_vec):
            out.write(f"{a} {v}\n")
        out.close()
        #Print_Chi_sq(M,x,y)

    def Set_Params(self, x):
        if not (len(x) == len(self.params_vec)):
            ValueError("Len x != len params vec")
        for i in range(len(x)):
            self.params_vec[i] = x[i]

    def Row_to_Material(self,row):
        loop_len = np.shape(self.matrix)[1]
        match = False
        ret = None
        for idx,m in enumerate(self.matrix):
            for i in range(0,loop_len,5):
                match = True
                inp = int(row[i])
                mat = int(m[i])
                if not inp == mat:
                    match = False
                    break
            if(match):
                ret = self.row_list[idx]
                break
        return ret

    def Test_to_Material_List(self, sub_mat):
        ret = []
        for row in sub_mat:
            tmp = self.Row_to_Material(row)
            if(tmp == None):
                print(row)
                ValueError("row in sub matrix did not match any row in fulll matrix")
            else:
                ret.append(tmp)
        return ret


def Write_RPA_RhoMat_JSON(out_file):
    #ret_dict[k] = {'count':count, 'tr_rho': tr, 'tr_diff': tr2}
    _dict = {}
    Dir = Data_Dir  + "Energies/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    for m in mats:
        for a in molecs:
            material = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}/evals-full.dat"
            en_dict = RPA.Comp_RPA_Energy(En_file)
            traces = RhoMatrix.Get_Rho_Trace(m,a)
            _dict[material] = {"Energies": en_dict, "Counts": traces}
    Write_JSON(_dict, out_file)


def Write_Tests_JSON(out_file):
    #ret_dict[k] = {'count':count, 'tr_rho': tr, 'tr_diff': tr2}
    _dict = {}
    Dir = Data_Dir  + "FeN3/"
    molecs = [ "clean", "CO", "OH"]
    for a in molecs:
        material = f"FeN3-{a}"
        En_file=Dir + f"{material}-evals.dat"
        en_dict = RPA.Comp_RPA_Energy(En_file)
        traces = RhoMatrix.Get_Rho_Tests("FeN3",a)
        _dict[material] = {"Energies": en_dict, "Counts": traces}
    Write_JSON(_dict, out_file)

def Get_Data(working_dir,params='rhomat'):
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
    if not(os.path.isfile(json_file)):
        if(params == "rhomat"):
            Write_RPA_RhoMat_JSON(json_file)
        elif(params == "proj"):
            Write_RPA_Projections_JSON(json_file)
    global Opt_File
    Opt_File = opt_file
    return LearningData(json_file, opt_file)


if __name__ == "__main__":
    opt_file = "Ridge_Regression/rand10_ridge.dat"
    err_file = "Ridge_Regression/leave-rand-error.txt"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    mats2 = ["RuO2", "RuMoO2", "RuHfO2", "RuSnO2", "Au"]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    Write_Tests_JSON("FeN3.json")
    #main(opt_file)
    #Write_Latex_Error(err_file,"ridge_err_latex")
    #Write_Latex_Params(opt_file,"ridge_params")




