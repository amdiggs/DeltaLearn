import re
import numpy as np
import math
import pdb
import json
import matplotlib.pyplot as plt

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



def Get_Proj_Info(line):
    vals = line.split()
    if(len(vals) == 0):
        print("Projections Line is empty/n")
        exit(4)
    ret = {}
    ret['Num_States'] = int(vals[0])
    ret['Num_Bands'] = int(vals[2])
    ret['Num_Proj'] = int(vals[4])
    ret['Num_Species'] = int(vals[6])
    return ret

def Get_Type_Info(lines):
    dat = []
    num_ats = 0
    for line in lines:
        vals = line.split()
        if(len(vals) == 0):
            print("Projections state line is empty/n")
            exit(4)
        ret = {}
        ret["Type"] = vals[0]
        ret['Num_Atoms'] = int(vals[1])
        ret['Num_Orbitals'] = int(vals[2])
        ret['l_Max'] = int(vals[3])
        shells = []
        for i in range(4,len(vals)):
            shells.append(int(vals[i]))
        ret["shells"] = shells
        dat.append(ret)
        num_ats += ret["Num_Atoms"]
    return dat, num_ats


def Get_State_Info(line):
    vals = line.split()
    if(len(vals) == 0):
        print("Projections state line is empty/n")
        exit(4)
    ret = {}
    ret['State_Num'] = vals[1]
    ret['K_Point'] = [float(vals[3]),float(vals[4]),float(vals[5])]
    ret['Weight'] = float(vals[7])
    ret['Spin'] = int(vals[9][0:2])
    return ret

def Get_Projections(file):
    lines = open(file).readlines()
    # n states, n bands, n projections, n species
    # Symbol nAtoms nOrbitalsPerAtom lMax nShells(l=0) ... nShells(l=lMax)
    # State num, k vec, weight, spin
    # tot_dict = {"Info":proj_info, "atom": {orbital_info: ###, 
    # "states":[{"state_info":###,"projections":[state_matrix]}],
    dat = {}
    dat["Genral_info"] = Get_Proj_Info(lines[0])
    num_specs = dat["Genral_info"]["Num_Species"]
    num_bands = dat["Genral_info"]["Num_Bands"]
    num_states = dat["Genral_info"]["Num_States"]
    dat["Species_info"], num_ats = Get_Type_Info(lines[2:2+num_specs])
    states = []
    get_state = False
    i = 2+num_specs
    while (i < len(lines)):
        #iterate over lines until match the line that I want
        # then grab line and store in a python dictonary 
        # then iter over next n lines
        # 
        line = lines[i]
        if(re.match(r'#\s+\d',line)):
            if(get_state):
                state_dict["Bands"] = projections
                states.append(state_dict)
            state_dict = {}
            state_dict["State_info"] = Get_State_Info(line)
            projections = []
            i+=1
            get_state = True
            #put projections into state_dict
            continue
        else:
            vals = line.split()
            cont = []
            for v in vals:
                cont.append(float(v))
            projections.append(cont)
            i+=1
    state_dict["Bands"] = projections
    states.append(state_dict)
    dat["States"] = states
    return dat, num_states, num_bands, num_ats




def Get_Average_Occupations(prefix):
    proj_file = prefix + "band.bandProjections"
    filling_file = prefix + "band.Fillings"
    elec_file = prefix + "out-bands"
    num_e = Get_Num_Elec(elec_file)
    dat, num_states, num_bands, tot_num_ats = Get_Projections(proj_file)
    fillings = Get_Fillings(filling_file,num_states,num_bands)
    total_projections = np.zeros([tot_num_ats])
    atom_typs = []
    species_info = dat["Species_info"]
    typs_dict = {}
    for spec in species_info:
        typs_dict[spec["Type"]] = {"count": spec["Num_Atoms"], "occupation": 0.0}
        for i in range(spec["Num_Atoms"]):
            atom_typs.append(spec["Type"])
    # Loop through all states
    for s_id, state in enumerate(dat["States"]):
        fill = fillings[s_id]
        #State_info contains the info from the lines
        #     0  [ +0.0000000 +0.0000000 +0.0000000 ]  0.000364431  spin +1;
        # for this line
        #State_info['State_Num'] = 0
        #State_info['K_Point'] = [0.0, 0.0, 0.0]
        #State_info['Weight'] = 0.000364431
        #State_info['Spin'] = 1
        #
        #W is the k point weight
        w = state["State_info"]["Weight"]
        # state["Bands"] is a (num_bands)X(num_atoms*num_AOs_per_atom*2) array which contains the projections 
        # for each band at a state.
        bands = state["Bands"]
        bands = np.asarray(bands)
        #print("The shape of state[bands]", np.shape(bands))
        for b_id,band in enumerate(bands):
            sh = np.shape(band)
            occ = fill[b_id]
            projections = []
            off_set = 0
            at_num = 0
            for spec in species_info: # loop through each species
                num_ats = spec["Num_Atoms"] #number of atoms per species
                #nAO is 2x the number of orbitals per atom. the 2 is for the re and im parts of the projections.
                nAO = 2*spec["Num_Orbitals"]
                for i in range(num_ats):
                    val = 0.0
                    end = off_set + nAO
                    for j in range(off_set, end, 2):
                        # sum up the total magnitude of the projections for each atom
                        val += band[j]*band[j] + band[j+1]*band[j+1]
                        off_set +=2
                    projections.append(val*w) # this is just a place holder for print debugguing
                    # add the k point weighted total projection for each atom for each band for each state to total projections
                    total_projections[at_num] += val*w*occ
                    at_num += 1
    tot_e = 0.0
    for p,typ in zip(total_projections,atom_typs):
        num_typ = typs_dict[typ]["count"]
        typs_dict[typ]["occupation"] += p/num_typ
        tot_e+=p
    delta = num_e - tot_e
    if(abs(delta) > 4.0):
        print(f"Warning: Number of projected electrons significantly differs from the expected number of electrons\n")
        print(f"Projected electrons: {tot_e} Expected electrons: {num_e}\n")
    #print(typs_dict)
    return typs_dict

def Get_Full_Occupations(prefix):
    print(prefix)
    proj_file = prefix + "sp.bandProjections"
    filling_file = prefix + "sp.Fillings"
    elec_file = prefix + "out-sp"
    num_e = Get_Num_Elec(elec_file)
    dat, num_states, num_bands, tot_num_ats = Get_Projections(proj_file)
    fillings = Get_Fillings(filling_file,num_states,num_bands)
    total_projections = np.zeros([tot_num_ats])
    atom_typs = []
    species_info = dat["Species_info"]
    typs_dict = {}
    for spec in species_info:
        typs_dict[spec["Type"]] = {"count": spec["Num_Atoms"], "occupation": 0.0}
        for i in range(spec["Num_Atoms"]):
            atom_typs.append(spec["Type"])
    # Loop through all states
    for s_id, state in enumerate(dat["States"]):
        fill = fillings[s_id]
        #State_info contains the info from the lines
        #     0  [ +0.0000000 +0.0000000 +0.0000000 ]  0.000364431  spin +1;
        # for this line
        #State_info['State_Num'] = 0
        #State_info['K_Point'] = [0.0, 0.0, 0.0]
        #State_info['Weight'] = 0.000364431
        #State_info['Spin'] = 1
        #
        #W is the k point weight
        w = state["State_info"]["Weight"]
        # state["Bands"] is a (num_bands)X(num_atoms*num_AOs_per_atom*2) array which contains the projections 
        # for each band at a state.
        bands = state["Bands"]
        bands = np.asarray(bands)
        #print("The shape of state[bands]", np.shape(bands))
        for b_id,band in enumerate(bands):
            #sh = np.shape(band)
            #print(sh)
            occ = fill[b_id]
            projections = []
            off_set = 0
            at_num = 0
            for spec in species_info: # loop through each species
                num_ats = spec["Num_Atoms"] #number of atoms per species
                #nAO is 2x the number of orbitals per atom. the 2 is for the re and im parts of the projections.
                nAO = 2*spec["Num_Orbitals"]
                for i in range(num_ats):
                    val = 0.0
                    end = off_set + nAO
                    for j in range(off_set, end, 2):
                        # sum up the total magnitude of the projections for each atom
                        val += band[j]*band[j] + band[j+1]*band[j+1]
                        off_set +=2
                    projections.append(val*w) # this is just a place holder for print debugguing
                    # add the k point weighted total projection for each atom for each band for each state to total projections
                    total_projections[at_num] += val*w*occ
                    at_num += 1
    tot_e = 0.0
    for p,typ in zip(total_projections,atom_typs):
        num_typ = typs_dict[typ]["count"]
        typs_dict[typ]["occupation"] += p
        tot_e+=p
    delta = num_e - tot_e
    if(abs(delta) > 4.0):
        print(f"Warning: Number of projected electrons significantly differs from the expected number of electrons\n")
        print(f"Projected electrons: {tot_e} Expected electrons: {num_e}\n")
    #print(typs_dict)
    return typs_dict



if __name__ == '__main__':
    mat = "FeNC"
    ad = "CO"
    prefix = f"./RPA-BandProj/{mat}/{mat}-{ad}/"
    occ = Get_Full_Occupations(prefix)
    for k,v in occ.items():
        print(f"{k} = {v}\n")












