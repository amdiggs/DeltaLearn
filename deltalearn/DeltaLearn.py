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


def test_serial_line(M,X,Y):
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

def Bad_GD(M,X,Y):
    #gradient desent
    #Opt M*x - y
    #M and y are known
    func = Ridge
    step = -0.01
    lowest = 10000000.
    prev = 1.0e+18
    val = 0.0
    for i in range(10000):
        step = 0.1
        nab, tot = Comp_Grad(M,X,Y,func)
        for j in range(15):
            dx = -1.0*nab*step
            tmp = np.copy(X)
            tmp += dx
            tmp_val = func(M,tmp,Y)
            if(tmp_val < prev):
                X += dx
                val = func(M,X,Y)
                prev = tmp_val
                break
            else:
                step /= 2.0
        if(val < lowest):
            lowest = val
        if not (i%100):
            print(lowest)
            print(val,tot, step)
    return X

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
    M,X,Y,fmt = Initalize(working_dir)
    Bad_GD(M,X,Y)
    #Plot_Lin_Model(json_file,leave, opt_file)



if __name__ == "__main__":
    opt_file = "Ridge_Regression/rand10_ridge.dat"
    err_file = "Ridge_Regression/leave-rand-error.txt"
    #main(opt_file)
    #Write_Latex_Error(err_file,"ridge_err_latex")
    #Print_Chi_sq(M,x,y)
    #Write_Latex_Params(opt_file,"ridge_params")




