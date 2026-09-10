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

Data_Dir = "./Data/"

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

class LearningData:
    def __init__(self, json_file,params_list) -> None:
        self.row_list = []
        self.col_list = None
        self.matrix = None
        self.E_vec = None
        self.params_vec = None
        self.Init_JSON(json_file,params_list)
        self.num_rows = 0

    def Init_JSON(self,file,params_list):
        atoms = ["Ag", "Cu", "Fe", "Co", "Mn", "Pt", "Ru"]
        atoms2 = ["H", "C", "N", "O"]
        #atoms = ["Co", "Cu", "Fe", "Mo", "Ni", "Pt", "H", "C", "N", "O"]
        counts_mat = []
        energy_vec = []
        ats = []
        get_ats = True
        data = Get_Data_From_JSON(file)
        #loop through materials
        row_count = 0
        for material, dat in data.items():
            row_count += 1
            self.row_list.append(material)
            energy_vec.append(dat["Energies"]["DeltaE"])
            counts = dat["Counts"]
            tmp = []
            #loop through atoms for a given mat
            ats = []
            for at,vals in counts.items():
                for k, v in vals.items():
                    if(at in atoms):
                        if(k in params_list):
                            tmp.append(v)
                            if(get_ats):
                                ats.append(f"{at}:{k}")
                    else:
                        if(k == 'count'):
                            tmp.append(v)
                            if(get_ats):
                                ats.append(f"{at}:{k}")
            counts_mat.append(tmp)
        self.matrix = np.asarray(counts_mat)
        self.E_vec = np.asarray(energy_vec)
        self.col_list = ats
        self.num_rows = row_count

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


def Write_RPA_Data_JSON(out_file):
    _dict = {}
    Dir = Data_Dir  + "Energies/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC", "Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = [ "clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    for m in mats:
        for a in molecs:
            material = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}/evals-full.dat"
            en_dict = RPA.Comp_RPA_Energy(En_file)
            trace_dict = RhoMatrix.Get_Rho_Trace(m,a)
            proj_dict = BandProjections.Get_Occupations(m,a)
            for k in proj_dict.keys():
                trace_dict[k]["occupation"] = proj_dict[k]["occupation"]
            _dict[material] = {"Energies": en_dict, "Counts": trace_dict}
    Write_JSON(_dict, out_file)


def Write_Tests_JSON(out_file):
    _dict = {}
    Dir = Data_Dir  + "Tests/"
    mats1 = [ "FeN3-clean", "FeN3-CO", "FeN3-OH"]
    mats2 = [ "Cu-H2", "Cu-defect", "Cu-bulk"]
    mats3 = [ "Cu-111-33-2", "Cu-111-33-3"]
    materials = mats1 + mats2 + mats3
    for m in materials:
        En_file=Dir + f"{m}-evals.dat"
        en_dict = RPA.Comp_RPA_Energy(En_file)
        trace_dict = RhoMatrix.Get_Rho_Tests(m)
        proj_dict = BandProjections.Get_Occupations_Tests(m)
        for k in proj_dict.keys():
            trace_dict[k]["occupation"] = proj_dict[k]["occupation"]
        _dict[m] = {"Energies": en_dict, "Counts": trace_dict}
    Write_JSON(_dict, out_file)

def Update_Data(typ='training'):
    if(typ == 'training'):
        json_file = Data_Dir + "RPA.json"
        Write_RPA_Data_JSON(json_file)
    elif(typ == 'test'):
        json_file = Data_Dir + "Tests.json"
        Write_Tests_JSON(json_file)
    else:
        ValueError("type did not match training or test")
    return


def Get_Data(params_list,typ):
    if(typ == 'training'):
        json_file = Data_Dir + "RPA.json"
    elif(typ == 'test'):
        json_file = Data_Dir + "Tests.json"
    return LearningData(json_file,params_list)

if __name__ == "__main__":
    print("Get_Data.py")



