from os.path import exists
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
from SetUp import Initalize

EMPTY= re.compile(r"\s*\n")
INT=r"-?\d+"
FLOAT=r"-?\d+\.\d+"


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

def Get_Ion_Counts(dict):
    ret = []
    counts = {"Ag":0, "Pt":0,"Cu":0,"Ru":0,"Fe":0,"Co":0,"Mn":0,"C":0,"H":0,"O":0,"N":0,}
    #pdb.set_trace()
    for k,v in dict.items():
        ret.append(v)
    return ret


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



def Vec_Line(M,X,y):
    vals = []
    for i in range(len(y)):
        tmp = 0.
        r = M[i]
        for c,x in zip(r,X):
            tmp += c*x
        vals.append(tmp-y[i])
    return vals


def mat_vec_mul(M,X,y):
    vals = []
    counts = []
    for i in range(len(y)):
        tmp = 0.
        count = 0
        r = M[i]
        for c,x in zip(r,X):
            tmp += c*x
            count += c
        vals.append(tmp)
        counts.append(count)
    return np.asarray(vals), np.asarray(counts)

def Chi_sq(M,x,y):
    V = Vec_Line(M,x,y)
    tot = 0.0
    for i in range(len(V)):
        v = V[i]
        tot += v*v
    return tot

def Ridge(M,x,y):
    lam = 1.0
    V = Vec_Line(M,x,y)
    tot = 0.0
    for i in range(len(V)):
        v = V[i]
        tot += v*v
    for xi in x:
        tot += lam*xi*xi
    return tot

def Print_Chi_sq(M,x,y):
    V = Vec_Line(M,x,y)
    count = 1
    for v in V:
        print(f"{count}: {v}\n")
        count += 1


def Comp_Grad(M,x,y,func):
    grad = []
    max_mag = -1.0
    for i in range(len(x)):
        x[i] += 0.001
        fp = func(M,x,y)
        x[i] -= 0.002
        fm = func(M,x,y)
        tmp = fp - fm
        grad.append(tmp)
        if(abs(tmp) > max_mag):
            max_mag = abs(tmp)
        x[i]+=0.001
    if(max_mag < 1.0):
        N = 1.0
    else:
        N = max_mag
    ret = np.zeros(len(grad))
    for i,v in enumerate(grad):
        ret[i] = v/N
    return ret, max_mag


def Comp_Line_Search(M,x,y,func):
    grad = []
    for i in range(len(x)):
        step = 0.1
        x[i] += 0.001
        fp = func(M,x,y)
        x[i] -= 0.002
        fm = func(M,x,y)
        x[i]+=0.001
        tmp = 0.5*(fp - fm)
        grad.append(tmp)
        prev = func(M,x,y)
        for j in range(15):
            dx = -1.0*tmp*step
            tmp_vec = np.copy(x)
            tmp_vec[i] += dx
            tmp_val = func(M,tmp_vec,y)
            if(tmp_val < prev):
                x[i] = tmp_vec[i]
                break
            else:
                step /= 2.0
    val = func(M,x,y)
    return val


def JSON_to_Matrix(file, mat_list):
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

def test_serial_line(json_file, mat_list, Opt_file):
    #Opt M*x - y
    #M and y are known
    func = Chi_sq
    M, y, Ats = JSON_to_Matrix(json_file, mat_list)
    x = File_to_Vec(Opt_file,1)
    step = -0.01
    lowest = 10000000.
    prev = 1.0e+18
    val = 0.0
    for i in range(10000):
        val = Comp_Line_Search(M,x,y,func)
        if(val < lowest):
            lowest = val
        if not (i%100):
            print(lowest)
            print(val)
    print(x)
    out = open(Opt_file,'w')
    for a,v in zip(Ats,x):
        out.write(f"{a} {v}\n")
    out.close()
    Print_Chi_sq(M,x,y)
    return x

def Bad_CG(json_file, mat_list, Opt_file):
    #Opt M*x - y
    #M and y are known
    func = Ridge
    M, y, Ats = JSON_to_Matrix(json_file, mat_list)
    x = File_to_Vec(Opt_file,1)
    step = -0.01
    lowest = 10000000.
    prev = 1.0e+18
    val = 0.0
    for i in range(10000):
        step = 0.1
        nab, tot = Comp_Grad(M,x,y,func)
        for j in range(15):
            dx = -1.0*nab*step
            tmp = np.copy(x)
            tmp += dx
            tmp_val = func(M,tmp,y)
            if(tmp_val < prev):
                x += dx
                val = func(M,x,y)
                prev = tmp_val
                break
            else:
                step /= 2.0
        if(val < lowest):
            lowest = val
        if not (i%100):
            print(lowest)
            print(val,tot, step)
    print(x)
    out = open(Opt_file,'w')
    for a,v in zip(Ats,x):
        out.write(f"{a} {v}\n")
    out.close()
    Print_Chi_sq(M,x,y)
    return x

def Plot_Lin_Model(json_file,leave_list, opt_file):
    M, dft, ats = JSON_to_Matrix(json_file, leave_list)
    x = File_to_Vec(opt_file,1)
    lin, c = mat_vec_mul(M, x, dft)
    lin2 = lin/c
    dft2 = dft/c
    xy = np.arange(np.min(dft2)-1.0,np.max(dft2)+1.0)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dft2, lin2)
    ax.plot(xy, xy, linestyle='--')
    #ax.text(0.1,0.85, "$E_{RPA}(\infty)$ = " + "{0:.5f} (eV)".format(par[1]), fontsize = 18,color = 'k', transform = ax.transAxes)
    ax.set_ylabel("$Liner Regression$ (eV)")
    ax.set_xlabel("$RPA$")
    fig.savefig("Random_with_Occupations.pdf", dpi = 300, format = 'pdf', bbox_inches = 'tight')
    plt.show()
    err = np.abs(dft-lin)
    err2 = np.abs(dft2-lin2)
    diff = dft-lin
    rel_diff = diff/dft
    rel_err = np.abs(rel_diff)
    diff2 = dft2-lin2
    rel_diff2 = diff2/dft2
    rel_err2 = np.abs(rel_diff2)
    out = open("leave-rand-error.txt", 'w')
    txt = "Surface:  Total Error: Absolute (eV) Relative % Per-Atom Error: Absolute (eV) Relative %\n"
    out.write(txt)
    txt2 = "{0}          {1:.3f}                 {2:.3f}          {3:.3f}                 {4:.3f}\n"
    print(txt)
    for i in range(len(err)):
        pt = txt2.format(leave_list[i], err[i], rel_err[i]*100, err2[i], rel_err2[i]*100)
        out.write(pt)
    out.close()




def CompDeltaE(working_dir):
    json_file, opt_file, mats, leave = Initalize(working_dir)
    Bad_CG(json_file, mats, opt_file)
    Plot_Lin_Model(json_file,leave, opt_file)



if __name__ == "__main__":
    opt_file = "Ridge_Regression/rand10_ridge.dat"
    err_file = "Ridge_Regression/leave-rand-error.txt"
    #main(opt_file)
    #Write_Latex_Error(err_file,"ridge_err_latex")
    #Write_Latex_Params(opt_file,"ridge_params")
    dist = [0.22289, 0.212899,  0.2029, 0.1829]
    for d in dist:
        dm = 6.35*d
        print(dm)




