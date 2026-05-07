import sys
import os
import re
import linecache as lc
import numpy as np
import math
import pdb
import json
import shutil
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import constants
import h5py
import Plot


ITEMS=re.compile('ITEM:')
TIMESTEP=re.compile('ITEM: TIMESTEP')
NUM=re.compile('ITEM: NUMBER OF ATOMS')
BB=re.compile('ITEM: BOX BOUNDS')
ATOMS=re.compile('ITEM: ATOMS \w')
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

doc = """ITEM: TIMESTEP
{0:.0f}
ITEM: NUMBER OF ATOMS
{1:.0f}
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 {2:.10e}
0.0000000000000000e+00 {3:.10e}
0.0000000000000000e+00 {4:.10e}
ITEM: ATOMS id type x y z\n"""

cell_doc = """CELL PARAMETERS
  {0:.10f}   {1:.10f}   {2:.10f}
  {3:.10f}   {4:.10f}   {5:.10f}
  {6:.10f}   {7:.10f}   {8:.10f}

Atomic Coordinates
"""

def line(x,m,b):
    return x*m + b


def Mat3_V_Prod(M,vecs):
    new_vecs = []
    for v in vecs:
        nv = np.matmul(M,v)
        new_vecs.append(nv)
    return new_vecs

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



def Read_Converge(file):
    E, P = [],[]
    for line in open(file).readlines():
        vals = line.split()
        if(len(vals) == 0):
            continue
        else:
            P.append(float(vals[0]))
            E.append(float(vals[1]))
    E = np.asarray(E)
    E*=ha2ev
    Delta = []
    for i in range(len(P)-1):
        print("delta Ecut wfc: {0:.0f} - {1:.0f}".format(P[i+1],P[i]))
        val = E[i+1] - E[i]
        Delta.append(val)
        print("delta F = {0:.4e} eV/atom".format(val/8.0))
    E= E - E[0]
    return np.asarray(P), E, np.asarray(Delta)

def Plot_Converge(file):
    fig = plt.figure()
    ax = fig.add_subplot()
    par, E, delta = Read_Converge(file)
    #print(delta/8.0)
    ax.scatter(par[1:],delta/8.0)
    ax.set_xticks(par[1:])
    ax.set_ylabel("$\Delta$E (eV)")
    #ax.set_ylabel("E (eV)")
    ax.set_xlabel("KP")
    #fig.savefig("IrO2_KP_Delta_converge.pdf", dpi = 300, format = 'pdf', bbox_inches = 'tight')
    plt.show()

def Read_RPA(file):
    Erpa = []
    epi_cut = []
    Evdw = 0.0
    SP = True
    for line in open(file).readlines():
        if(SP):
            if(re.search(r"RPA energy",line)):
                vals = line.split()
                Erpa.append(ha2ev*float(vals[-1]))
                epi_cut.append(float(vals[0]))
            elif(re.search(r'EXX\(RPA\) =',line)):
                vals = line.split()
                Exx=ha2ev*float(vals[-1])
            elif(re.search(r'F =',line)):
                vals = line.split()
                F=ha2ev*float(vals[-1])
            elif(re.search(r'Etot =',line)):
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
        #print(f"EVDW = {Evdw}\n")
        F-=Evdw
    return np.asarray(epi_cut),np.asarray(Erpa), Exx, F, Exc


def Check_RPA(metal,ads,file):
    Evdw = 100
    SP = False
    for line in open(file).readlines():
        if(SP):
            if(re.search(r'F =',line)):
                vals = line.split()
                F=ha2ev*float(vals[-1])
            elif(re.search(r'EvdW =',line)):
                vals = line.split()
                Evdw=ha2ev*float(vals[-1])
        elif(re.search(r'#+SP#+',line)):
            SP=True
            continue
        else:
            if(re.search(r'F =',line)):
                vals = line.split()
                Fi=ha2ev*float(vals[-1])
            elif(re.search(r'EvdW =',line)):
                vals = line.split()
                Evdwi=ha2ev*float(vals[-1])
    if(Evdw == 100):
        delta = Fi - Evdwi - F
    else:
        delta = Fi - F
    print(f"{metal} + {ads}: {delta}  {Evdwi}\n")
    return


def Compute_Ead(metal):
    molecs = Get_Data_From_JSON("Materials/Adsorbates/Adsorbates.json")
    out = open(f"{metal}-RPA-Adsorption-Energies.txt", 'w')
    out.write("# Adsorption Energy\n")
    Eads_lst = []
    for line in open(f"{metal}-RPA-Energy-Data.dat").readlines():
        vals = line.split()
        if(len(vals) == 0 or re.match(r'^#',line)):
            continue
        else:
            dic = {}
            if(vals[1] == 'clean'):
                Ecg = float(vals[2])
                Ecr = float(vals[3])
                continue
            else:
                dic["Molecule"] = vals[1]
                dic["Egga"] = float(vals[2])
                dic["Erpa"] = float(vals[3])
                Eads_lst.append(dic)
    for m in molecs:
        ads = m["label"]
        # print("###########", ads, "###################")
        Emgga = m["Egga"]
        Emrpa = m["Erpa"]
        for l in Eads_lst:
            # print(l["Molecule"])
            rpa_line = "{0}\\ce{1}{2}{3}&{4:.4f} & {5:.4f} \\\\ \n\\hline\n"
            m_face = f"{metal} + "
            if(l["Molecule"] == ads):
                Egga = l["Egga"] - Ecg - Emgga
                Erpa = l["Erpa"] - Ecr - Emrpa
                latex_vals = [m_face,"{",ads, "}", Egga, Erpa]
                out.write(rpa_line.format(*latex_vals))
    return



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

def Write_RPA_JSON(metal,MI,molec,mol_file, json_file):
    rpa_dict = {'Adsorbate': molec,'Egga':0., 'Exc':0., 'Exx':0., 'Ecorr':0., 'Erpa':0.}
    bulk_name = f'{metal}-{MI}'
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
    if not (bulk_name in Data):
        Data[bulk_name] = []
    for ad in Data[bulk_name]:
        if(ad['Adsorbate'] == molec):
            print(f"Data all ready contains an entry for {molec}")
            return
    Data[bulk_name].append(rpa_dict)
    Write_JSON(Data,json_file)



def Write_DD_RPA_JSON(metal,MI,molec,mol_file, json_file):
    rpa_dict = {'Adsorbate': molec,'Egga':0., 'Exc':0., 'Exx':0., 'Ecorr':0., 'Erpa':0.}
    bulk_name = f'{metal}-{MI}'
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
    if not (bulk_name in Data):
        Data[bulk_name] = []
    for ad in Data[bulk_name]:
        if(ad['Adsorbate'] == molec):
            print(f"Data all ready contains an entry for {molec}")
            return
    Data[bulk_name].append(rpa_dict)
    Write_JSON(Data,json_file)

def Write_RPA(metal,MI,molec,mol_file, out_file):
    rpa_line = "{0}\\ce{1}{2}{3}&{4} & {5:.4f} & {6:.4f} & {7:.4f} & {8:.4f} & {9:.4f} & {10:.4f} \\\\ \n\\hline\n"
    epi_cut, Erpa, Exx, F, Exc = Read_RPA(mol_file)
    epi_cut = np.power(epi_cut,-1.5)
    epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
    delE = Exc - (Exx + par[1])
    print(delE) 
    Etot = F - delE
    print(Etot)
    m_face = f"{metal} +"
    latex_vals = [m_face,"{",molec, "}", MI, F, Exc, Exx, par[1], delE, Etot]
    out = open(out_file, 'a')
    out.write(rpa_line.format(*latex_vals))
    out.close()

def Write_RPA_Data(mat,mm,mol_file, out_file):
    rpa_line = "{0} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f} {7:.3f}\n"
    epi_cut, Erpa, Exx, F, Exc = Read_RPA(mol_file)
    epi_cut = np.power(epi_cut,-1.5)
    epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
    Ec=par[1]
    DeltaE = Exc - (Exx + Ec)
    Etot = F - DeltaE
    latex_vals = [mat, F,Exc,Exx,Ec,DeltaE,Etot,mm]
    out = open(out_file, 'a')
    out.write(rpa_line.format(*latex_vals))
    out.close()



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

def Get_Ion_Counts(file):
    atoms=Get_JDFTX_Ionpos(file)
    counts = {"Ag":0, "Pt":0,"Cu":0,"Ru":0,"Fe":0,"Co":0,"Mn":0,"C":0,"H":0,"O":0,"N":0,}
    #pdb.set_trace()
    for at in atoms:
        counts[at["type"]] +=1
    return counts


def Comp_Ion_Matrix(out_file):
    dict = {}
    rpa_line = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n"
    Dir = "OUT/out-MNC-RPA/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    slabs = ["Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = ["clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    counts = {"Ag":0, "Pt":0,"Cu":0,"Ru":0,"Fe":0,"Co":0,"Mn":0,"C":0,"H":0,"O":0,"N":0,}
    out = open(out_file,'w')
    out2 = open("rpa-energies-vec.dat", 'w')
    for m in mats:
        for a in molecs:
            En_file=Dir + f"{m}/{m}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec=par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{m}/{m}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            for k,v in vals.items():
                out.write(f"{v} ")
            out.write("\n")
            out2.write(rpa_line.format(F,Exc,Exx,Ec,DeltaE))
    Dir = "OUT/out-metals-111-RPA/"
    for s in slabs:
        for a in molecs:
            En_file=Dir + f"{s}/{s}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec = par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{s}/{s}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            for k,v in vals.items():
                out.write(f"{v} ")
            out.write("\n")
            out2.write(rpa_line.format(F,Exc,Exx,Ec,DeltaE))
    out.close()

def Comp_Ion_Matrix_Adsorbate(ion_file, en_file):
    rpa_line = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n"
    Dir = "OUT/out-MNC-RPA/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    slabs = ["Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = ["N2"]
    out = open(ion_file,'w')
    out2 = open(en_file, 'w')
    for m in mats:
        for a in molecs:
            En_file=Dir + f"{m}/{m}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec=par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{m}/{m}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            for k,v in vals.items():
                out.write(f"{v} ")
            out.write("\n")
            out2.write(rpa_line.format(F,Exc,Exx,Ec,DeltaE))
    Dir = "OUT/out-metals-111-RPA/"
    for s in slabs:
        for a in molecs:
            En_file=Dir + f"{s}/{s}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec = par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{s}/{s}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            for k,v in vals.items():
                out.write(f"{v} ")
            out.write("\n")
            out2.write(rpa_line.format(F,Exc,Exx,Ec,DeltaE))
    out.close()
    out2.close()




def Ion_Matrix_to_JSON(out_file):
    dict = {}
    Dir = "OUT/out-MNC-RPA/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    slabs = ["Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = ["clean", "CO", "CO2", "COOH", "H", "H2O", "N2","O", "O2", "OH", "OOH"]
    counts = {"Ag":0, "Pt":0,"Cu":0,"Ru":0,"Fe":0,"Co":0,"Mn":0,"C":0,"H":0,"O":0,"N":0,}
    for m in mats:
        for a in molecs:
            sys = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec=par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{m}/{m}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            dict[sys] = {"DeltaE": DeltaE, "Counts": vals}
    Dir = "OUT/out-metals-111-RPA/"
    for m in slabs:
        for a in molecs:
            sys = f"{m}-{a}"
            En_file=Dir + f"{m}/{m}-{a}-evals.dat"
            epi_cut, Erpa, Exx, F, Exc = Read_RPA(En_file)
            epi_cut = np.power(epi_cut,-1.5)
            epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
            Ec=par[1]
            DeltaE =(Exx + Ec) - Exc
            file=Dir + f"{m}/{m}-{a}.ionpos"
            vals = Get_Ion_Counts(file)
            dict[sys] = {"DeltaE": DeltaE, "Counts": vals}
    Write_JSON(dict, out_file)

def Plot_RPA(file,molec):
    fig = plt.figure()
    ax = fig.add_subplot()
    epi_cut, Erpa, Exx, F, Exc = Read_RPA(file)
    epi_cut = np.power(epi_cut,-1.5)
    epi_inf, Erpa_inf, par = Fit_Line(epi_cut,Erpa)
    delE = Exc - (Exx + par[1])
    print(delE)
    Etot = F - delE
    print(Etot)
    latex_vals = [molec, F, Exc, Exx, par[1], delE, Etot]
    out = open("CuNC3.out", 'a')
    out.write("{0}&{1:.4f} & {2:.4f} & {3:.4f} & {4:.4f} & {5:.4f} & {6:.4f} \\\\ \n\\hline\n".format(*latex_vals))
    out.close()
    ax.scatter(epi_cut, Erpa,label="BGW")
    ax.plot(epi_inf, Erpa_inf, label="fit")
    ax.text(0.1,0.85, "$E_{RPA}(\infty)$ = " + "{0:.5f} (eV)".format(par[1]), fontsize = 18,color = 'k', transform = ax.transAxes)
    ax.set_ylabel("$E_{RPA}$ (eV)")
    ax.set_xlabel("$1/\epsilon_{cut} (Ry)^{-3/2}$")
    fig.savefig("Figures/{0}-RPA.pdf".format(molec), dpi = 300, format = 'pdf', bbox_inches = 'tight')
    #plt.show()


def Comp_RPA_Energy():
    Dir = "out-MNC-RPA/"
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    molecs = ["clean", "CO", "CO2", "COOH", "H", "H2O", "N2", "O", "O2", "OH", "OOH"]
    for m in mats:
        Compute_Ead(m)

def Test_matmul(M,X,y):
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

def Comp_DD(M,x,y):
    V = Test_matmul(M,x,y)
    tot = 0.0
    for i in range(len(V)):
        v = V[i]
        tot += v*v
    return tot

def Print_DD(M,x,y):
    V = Test_matmul(M,x,y)
    count = 1
    for v in V:
        print(f"{count}: {v}\n")
        count += 1


def Comp_Grad(M,x,y):
    grad = []
    tot = 0.
    for i in range(len(x)):
        x[i] += 0.001
        fp = Comp_DD(M,x,y)
        x[i] -= 0.002
        fm = Comp_DD(M,x,y)
        tmp = fp - fm
        grad.append(tmp)
        tot+=tmp*tmp
        x[i]+=0.001
    N=math.sqrt(tot)
    if(N<11.0):
        N=N
    ret = np.zeros([len(grad)])
    for i in range(len(grad)):
        ret[i] = grad[i]/N
    return ret, N



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


def Bad_CG(mat_file, E_file, Opt_file):
    #Opt M*x - y
    #M and y are known
    Ats = ["Ag", "Pt", "Cu", "Ru", "Fe", "Co", "Mn", "C", "H", "O", "N"]
    M = File_to_Matrix(mat_file)
    y = File_to_Vec(E_file, 4)
    x = File_to_Vec(Opt_file,1)
    for i in range(100000):
        nab, tot = Comp_Grad(M,x,y)
        if(tot > 11.0):
            step = 0.1
        elif(tot > 1.0):
            step = 0.01
        else:
            step = 0.001
        x -= nab*step
        val = Comp_DD(M,x,y)
        if not (i%100):
            print(val, step)
    print(x)
    out = open(Opt_file,'w')
    for a,v in zip(Ats,x):
        out.write(f"{a}: {v}\n")
    out.close()
    Print_DD(M,x,y)
    return x

def Plot_Lin_Model():
    P = "leave-N2-out/"
    mat_file = "N2-IonCounts.dat"
    E_file = "N2-Energies.dat"
    Opt_file = P + "N2-Out-Optimized-Delta-Vals.txt"
    Comp_Ion_Matrix_Adsorbate(mat_file, E_file)
    M = File_to_Matrix(mat_file)
    dft = File_to_Vec(E_file, 4)
    x = File_to_Vec(Opt_file,1)
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
    fig.savefig("Figures/N2-Out-parity-perAtom.pdf", dpi = 300, format = 'pdf', bbox_inches = 'tight')
    plt.show()
    mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC","Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    err = np.abs(dft-lin)
    err2 = np.abs(dft2-lin2)
    diff = dft-lin
    rel_diff = diff/dft
    rel_err = np.abs(rel_diff)
    diff2 = dft2-lin2
    rel_diff2 = diff2/dft2
    rel_err2 = np.abs(rel_diff2)
    out = open(P+"error.dat", 'w')
    txt = "Surface:  Total Error: Absolute (eV) Relative % Per-Atom Error: Absolute (eV) Relative %\n"
    out.write(txt)
    txt2 = "{0}          {1:.3f}                 {2:.3f}          {3:.3f}                 {4:.3f}\n"
    print(txt)
    for i in range(len(err)):
        pt = txt2.format(mats[i], err[i], rel_err[i]*100, err2[i], rel_err2[i]*100)
        out.write(pt)
    out.close()

def main():
    Dir = "OUT/out-atoms-RPA/"
    #MNC,out-MNC-RPA,out-Molecules-RPA,out-metals-111-RPA,
    #mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    mats = ["FeNC"]
    #mats = ["Ag-111", "Cu-111", "Pt-111", "Ru-001"]
    molecs = ["clean", "CO", "CO2", "COOH", "H", "H2O", "N2", "O", "O2", "OH", "OOH"]
    nums = [0,2, 3, 4, 1, 3, 2, 1, 2, 2, 3]
    #mats = ["Ag", "Cu", "Fe", "Co", "Mn", "Pt", "Ru", "c", "n", "o", "h"]
    json_file = "Atom-Optimized-Energies.json"
    dic = {}
    for m in mats:
        latt_file = f"OUT/out-MNC-RPA/{m}/{m}-clean.lattice"
        latt = Get_JDFTX_Lattice(latt_file)
        M = np.transpose(latt)
        for a,n in zip(molecs,nums):
            file = f"OUT/out-MNC-RPA/{m}/{m}-{a}.ionpos"
            lines = open(file).readlines()
            lst = []
            tmp = []
            ag = np.zeros([3])
            typs = []
            for line in lines[1:]:
                vals = line.split()
                if(len(vals) == 0):
                    continue
                x = float(vals[2])
                y = float(vals[3])
                z = float(vals[4])
                if(z > 0.5):
                    typs.append(vals[1])
                    tmp.append(np.array([x,y,z]))
                elif(vals[1] == 'Fe'):
                    ag = Mat3_V_Prod(M, [np.array([x,y,z])])
            print(ag)
            cart = Mat3_V_Prod(M, tmp)
            fe = ag[0]
            for t,p in zip(typs,cart):
                print(p)
                at_dict = {}
                dx = p[0] - fe[0]
                dy = p[1] - fe[1]
                dz = p[2] - fe[2]
                at_dict[t] = [dx,dy,dz]
                lst.append(at_dict)
            dic[a] = lst
    Write_JSON(dic, "adsorbate.json")

def Gen_Submit(dst, sub_name, job_name, in_name, out_name):
    in_file = f"INPUT=\"{in_name}\"\n"
    out_file = f"OUTPUT=\"{out_name}\"\n"
    last = "srun -n 4 -c 16 jdftx_gpu -i $INPUT &> $OUTPUT"
    lines = open("RUN-DFT.sh").readlines()
    out = open(dst + sub_name, 'w')
    for line in lines:
        if(re.match(r"^\s*#SBATCH -J.*$",line)):
            out.write(f"#SBATCH -J {job_name}\n")
        else:
            out.write(line)
    out.write(in_file)
    out.write(out_file)
    out.write(last)
    out.close()

def Get_KP(file):
    nbnd = 0
    kp = 0
    for line in open(file).readlines():
        if(re.match(r"^\s*elec-n-bands .*$",line)):
            nbnd = line
        elif(re.match(r"^\s*kpoint-folding .*$",line)):
            kp = line
        else:
            continue
    return nbnd, kp


def Gen_SP(dst,out_file,mat):
    nbnd, kp = Get_KP(out_file)
    lines = open("in-sp").readlines()
    out = open(dst + "in-sp", 'w')
    for line in lines:
        if(re.match(r"^\s*elec-n-bands .*$",line)):
            out.write(str(nbnd))
        elif(re.match(r"^\s*kpoint-folding .*$",line)):
            out.write(str(kp))
        else:
            out.write(line)
    out.close()
    Gen_Submit(dst,"submit_sp.sh",mat+"-sp","in-sp","out-sp")


def Gen_Band(dst,out_file,mat):
    nbnd, kp = Get_KP(out_file)
    lines = open("in-bands").readlines()
    out = open(dst + "in-bands", 'w')
    for line in lines:
        if(re.match(r"^\s*elec-n-bands .*$",line)):
            out.write(str(nbnd))
        elif(re.match(r"^\s*kpoint-folding .*$",line)):
            out.write(str(kp))
        else:
            out.write(line)
    out.close()
    Gen_Submit(dst,"submit_bnds.sh",mat+"-bnd","in-bands","out-bands")

def move_files():
    Dir = "OUT/out-RPA-OER-1/"
    #MNC,out-MNC-RPA,out-Molecules-RPA,out-metals-111-RPA,
    #mats = ["AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    #out_file = f"OUT/out-RPA-OER-1/{m}/{m}-{a}/sp.out"
    out_dir = "RPA-BandProj"
    mats = ["Ag-111", "Cu-111", "Pt-111", "Ru-001","AgNC", "CuNC", "FeNC", "CoNC", "MnNC"]
    molecs = ["clean", "CO", "CO2", "COOH", "H", "H2O", "N2", "O", "O2", "OH", "OOH"]
    if(not os.path.isdir(out_dir)):
       os.mkdir(out_dir)
    for m in mats:
        sub1 = out_dir + f"/{m}"
        if(not os.path.isdir(sub1)):
            os.mkdir(sub1)
        for a in molecs:
            sub2 = sub1 + f"/{m}-{a}"
            if(not os.path.isdir(sub2)):
                os.mkdir(sub2)
            ion_src = f"OUT/out-RPA-OER-1/{m}/{m}-{a}/sp.ionpos"
            ion_dst = out_dir + f"/{m}/{m}-{a}/in.ionpos"
            shutil.copyfile(ion_src,ion_dst)
            latt_src = f"OUT/out-RPA-OER-1/{m}/{m}-{a}/sp.lattice"
            latt_dst = out_dir + f"/{m}/{m}-{a}/in.lattice"
            shutil.copyfile(latt_src,latt_dst)
            dst = out_dir + f"/{m}/{m}-{a}/"
            sp_out = f"OUT/out-RPA-OER-1/{m}/{m}-{a}/sp.out"
            Gen_SP(dst,sp_out,f"{m}-{a}")
            #Gen_Band(dst,sp_out,f"{m}-{a}")


if __name__ == "__main__":
    move_files()
    #Plot_Lin_Model()
    #Ion_Matrix_to_JSON("Energy-IonCounts.json")











