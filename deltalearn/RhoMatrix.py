import re
import numpy as np
import math
import pdb
import json
import matplotlib.pyplot as plt
import struct



Orbs = {"Ag":[1,1,5],"Cu":[1,1,5],"Co":[1,1,5],"Mn":[1,1,5],
        "Pt":[1,5],"Fe":[1,1,5],"Ru":[1,1,5],"C":[1,3],"N":[1,3], "O":[1,3],"H":[1]}

orb_typs = ['s', 'p', 'd','f']


Data_Dir = '/Users/diggs/Desktop/RPA-OER-1/Delta-Learning/Data/'

class RhoMat:
    def __init__(self, orb_num, arr, complex=True):
        if(complex):
            mat = np.zeros([orb_num,orb_num])
            count = 0
            for i in range(orb_num):
                for j in range(orb_num):
                    mat[i][j] = arr[count]*arr[count] + arr[count+1]*arr[count+1]
                    count += 2
            self.shape = np.shape(mat)
            self.mat = mat
        else:
            self.shape = np.shape(arr)
            self.mat = arr

    def trace(self):
        tot = 0.0
        for i in range(self.shape[0]):
            tot += self.mat[i][i]
        #print(f"Rho mat trace: {tot}")
        return tot

    def OneMinus(self):
        one = np.ones(self.shape)
        mat = one - self.mat
        return RhoMat(0,mat, complex=False)

    def RhoMinusRhoSquared(self):
        sq = np.matmul(self.mat,self.mat)
        mat = self.mat - sq
        return RhoMat(0,mat, complex=False)

def Get_Data_From_JSON(file):
    with open(file,mode='r',encoding='utf-8') as f:
        return json.load(f)

def Write_JSON(data, file):
    with open(file,'w') as f:
        json.dump(data,f,indent=4)

def Read_Binary(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype='<f8')
    return data

def Get_Fillings(file, n_states, n_bnds):
    fil = Read_Binary(file)
    return fil.reshape(n_states,n_bnds)

def Get_Num_Elec(file):
#FillingsUpdate:  mu: -0.234743451  nElectrons: 204.000000  magneticMoment: [ Abs: 0.08841  Tot: +0.02249 ]
    lines = open(file).readlines()
    num = 0.0
    for line in lines:
        if(re.match(r"^\s*FillingsUpdate:\s.*$",line)):
            vals = line.split()
            num = float(vals[4])
            break
    return num


def Comment_or_Blank(line):
    if(re.match(r'^\s*$',line)):
        return True
    elif(re.match(r'^\s*#.*$',line)):
        return True
    else:
        return False

def Get_Ion_Counts(file):
    ret = {}
    for line in open(file).readlines()[1:]:
        vals = line.split()
        if(Comment_or_Blank(line)):
            continue
        typ = vals[1]
        if(typ in ret):
            ret[typ] += 1
        else:
            ret[typ] = 1
    return ret



def Get_RhoMats(rho_file, ion_file):
    # for species{
            #for orb {
                #for spin{
                    #for atom}}}
    counts = Get_Ion_Counts(ion_file)
    rho = Read_Binary(rho_file)
    print(np.shape(rho))
    ats = {}
    # for each species, species count
    start = 0
    for k,c in counts.items():
        at_dict = {}
        at_dict['type'] = k
        at_dict['count'] = c
        at_dict['nspin'] = 2
        n_spin = 2
        n_U = len(Orbs[k])
        if(n_U == 3):
            at_dict['n_U'] = 2
            at_dict['orbs'] = Orbs[k][1:]
        else:
            at_dict['n_U'] = n_U
            at_dict['orbs'] = Orbs[k]
        n_ats = c
        num_rhoMats = n_U*n_spin*n_ats*2.0
        num_mats = int(num_rhoMats)
        mats = []
        print(f"{k}: {c}")
        # for U
        test_count = 0
        for i,l in enumerate(Orbs[k]):
            orb_count = int(2.0*l*l)
            for s in range(n_spin):
                for at in range(n_ats):
                    end = start + orb_count
                    #print(f"start = {start} end = {end}")
                    vals = rho[start:end]
                    start = end
                    # skips the lower s orbital.
                    if(n_U == 3 and i == 0):
                        continue
                    else:
                        test_count += 1
                        mats.append(RhoMat(l,vals))
        at_dict['rhoMat'] = mats
        ats[k] = at_dict
    return ats

def Get_Zeros_Dict():
    ret_dict = {}
    for k,v in Orbs.items():
        ret_dict[k] = {}
        ret_dict[k]['count'] = 0
        count = 0
        for q,n in enumerate(v):
            if(len(v) == 3 and q == 0):
                continue
            l = int((n-1)/2)
            typ = orb_typs[l]
            lab = f"mu_{typ}"
            lab_diff = f"U_{typ}"
            ret_dict[k][lab] = 0.0
            ret_dict[k][lab_diff] = 0.0
    return ret_dict



def Get_Rho_Trace(mat, ad):
    rho_file = Data_Dir + f"rhoMatricies/{mat}-{ad}.rhoAtom"
    ion_file = Data_Dir + f"Energies/{mat}/{mat}-{ad}/sp.ionpos"
    at_dict = Get_RhoMats(rho_file, ion_file)
    ret_dict = Get_Zeros_Dict()
    for k,v in at_dict.items():
        mats = v['rhoMat']
        at_count = v['count']
        ret_dict[k]['count'] = at_count
        orbs = v['orbs']
        #orbs = Orbs[k]
        n_U = v['n_U']
        count = 0
        tmp_stuff = [5,6,5]
        for p,n in enumerate(orbs):
            num_mat = 2*at_count
            tr = 0.0
            tr_diff = 0.0
            l = int((n-1)/2)
            typ = orb_typs[l]
            lab = f"mu_{typ}"
            lab_diff = f"U_{typ}"
            for i in range(num_mat):
                mat = mats[count]
                val = mat.trace()
                tr += val
                diff_mat = mat.RhoMinusRhoSquared()
                v2 = diff_mat.trace()
                tr_diff += v2
                count += 1
            ret_dict[k][lab] = tr
            ret_dict[k][lab_diff] = tr_diff
    return ret_dict

def Get_Rho_Tests(mat, ad):
    rho_file = Data_Dir + f"FeN3/{mat}-{ad}.rhoAtom"
    ion_file = Data_Dir + f"FeN3/{mat}-{ad}.ionpos"
    at_dict = Get_RhoMats(rho_file, ion_file)
    ret_dict = Get_Zeros_Dict()
    for k,v in at_dict.items():
        mats = v['rhoMat']
        at_count = v['count']
        ret_dict[k]['count'] = at_count
        orbs = v['orbs']
        #orbs = Orbs[k]
        n_U = v['n_U']
        count = 0
        tmp_stuff = [5,6,5]
        for p,n in enumerate(orbs):
            num_mat = 2*at_count
            tr = 0.0
            tr_diff = 0.0
            l = int((n-1)/2)
            typ = orb_typs[l]
            lab = f"mu_{typ}"
            lab_diff = f"U_{typ}"
            for i in range(num_mat):
                mat = mats[count]
                val = mat.trace()
                tr += val
                diff_mat = mat.RhoMinusRhoSquared()
                v2 = diff_mat.trace()
                tr_diff += v2
                count += 1
            ret_dict[k][lab] = tr
            ret_dict[k][lab_diff] = tr_diff
    return ret_dict



def check():
    mat = "Ag-111"
    molecs = [ "clean","CO", "CO2", "COOH", "H", "H2O", "N2", "O", "O2", "OH", "OOH"]
    molecs = [ "clean","CO"]
    ad = "CO2"
    ad = "clean"
    for ad in molecs:
        rho_file = f"../Data/rhoMatricies/{mat}-{ad}.rhoAtom"
        ion_file = f"../Data/Energies/{mat}/{mat}-{ad}/sp.ionpos"
        ret_dic = Get_Rho_Trace(mat,ad)
        for k,v in ret_dic.items():
            print(f"#### {k} #######")
            for id, value in v.items():
                print(f"{id} = {value}")




if __name__ == '__main__':
    check()











