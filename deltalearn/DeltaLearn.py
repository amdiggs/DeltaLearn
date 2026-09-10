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
from . import GetData

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

colors=['mediumblue', 'crimson','darkgreen', 'darkorange', 'black', 'darkorchid','cyan', 'teal','olive','darkred','pink']
markers=['o','D','*','^','s']
Legend_Labels={'count':'count','mu_s':'mu s','U_s': 'U s','mu_p': 'mu p/d','U_p': 'U p/d','mu_d': 'mu p/d','U_d': 'U p/d', 'occupation':'occupation'}

color_dict = ["Ag", "Cu", "Co", "Mn", "Pt", "Fe", "Ru", "C", "N", "O", "H"]

Data_Dir = "./Data/"

def Vec_Line(M,X,y):
    vals = []
    for i in range(len(y)):
        tmp = 0.
        r = M[i]
        for c,x in zip(r,X):
            tmp += c*x
        vals.append(tmp-y[i])
    return vals


def mat_vec_mul(M,x):
    vals = []
    for i in range(np.shape(M)[0]):
        tmp = 0.
        r = M[i]
        for c,xi in zip(r,x):
            tmp += c*xi
        vals.append(tmp)
    return np.asarray(vals)

def Test_Model(M_test, y_test, x, alpha):
    lin = mat_vec_mul(M_test, x)
    xy = np.arange(np.min(y_test)-1.0,np.max(y_test)+1.0)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(y_test, lin)
    ax.plot(xy, xy, linestyle='--')
    #ax.text(0.1,0.85, "$E_{RPA}(\infty)$ = " + "{0:.5f} (eV)".format(par[1]), fontsize = 18,color = 'k', transform = ax.transAxes)
    ax.set_ylabel("$Liner Regression$ (eV)")
    ax.set_xlabel("$RPA$")
    fig.savefig(f"RidgeRegression-{alpha}.pdf", dpi = 300, format = 'pdf', bbox_inches = 'tight')
    plt.show()
    err = []
    rel_err = []
    for l,v in zip(lin,y_test):
        e = math.sqrt((l-v)*(l-v))
        rel_e = e/abs(v)
        err.append(e)
        rel_err.append(rel_e)
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

def Gen_Plot_Dict(labels):
    ret = {}
    legend_labels = []
    plot_params = []
    clr_idx = 0
    for l in labels:
        vals = l.split(':')
        at = vals[0]
        param = vals[1]
        if not param in legend_labels:
            legend_labels.append(param)
        if not at in ret.keys():
            print(len(colors),clr_idx)
            mkr_idx = 0
            ret[at] = {'color':colors[clr_idx], 'markers':[markers[mkr_idx]]}
            clr_idx += 1
            mkr_idx += 1
        else:
            ret[at]['markers'].append(markers[mkr_idx])
            mkr_idx += 1
    count = 0
    for k,v in ret.items():
        clr = v['color']
        mkrs = v['markers']
        for m in mkrs:
            if count < len(legend_labels) and count < 5:
                p = legend_labels[count]
                label = Legend_Labels[p]
                count += 1
            else:
                label = None
            plot_params.append({'color':clr,'marker':m, 'label':label})
    return ret, legend_labels, plot_params


def Gen_Labels():
    ret = []
    ats = ["Ag", "Cu", "Co", "Mn", "Pt", "Fe", "Ru"]
    ats2 = ["C", "N", "O"]
    ats3 = ["H"]
    labels=['count','mu s', 'U s', 'mu d', 'U d']
    labels2=['count','mu s', 'U s', 'mu p', 'U p']
    labels3=['count','mu s', 'U s']
    for a in ats:
        for p in labels:
            lab = f"{a}: {p}"
            ret.append(lab)
    for a in ats2:
        for p in labels2:
            lab = f"{a}: {p}"
            ret.append(lab)
    for a in ats3:
        for p in labels3:
            lab = f"{a}: {p}"
            ret.append(lab)
    return ret

def Write_coeffs(means, std):
    labs = Gen_Labels()
    out = open(f"{dirname}/coef-rho.dat",'w')
    txt = "Parameter   mean   std\n"
    out.write(txt)
    for l,m,s in zip(labs,means, std):
        txt = f"{l}   {m:.4f}   {s:.4f}\n"
        out.write(txt)
    out.close()


def Split_Data(X,y,num_test):
    num_train = int(len(y) - num_test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=num_train, test_size=num_test, shuffle=True)
    return X_train, X_test, y_train, y_test


def Comp_STD(coeffs, mean_coeffs):
    sums = []
    _shape = np.shape(coeffs)
    num_coeffs = _shape[0]
    num_runs = _shape[1]
    print(num_coeffs,num_runs)
    for i in range(num_coeffs):
        tot = 0.0
        for j in range(num_runs):
            #print(f"coeffs = {coeffs[j][i]}")
            #print(f"mean = {mean_coeffs[i]}")
            val = coeffs[i][j] - mean_coeffs[i]
            tot += val*val
        tot /= num_runs
        sums.append(math.sqrt(tot))
    return sums


def Comp_Correlation_Matrix(coeffs, mean_coeffs, std):
    corr = []
    _shape = np.shape(coeffs)
    num_coeffs = _shape[0]
    num_runs = _shape[1]
    div = num_runs - 1
    for i in range(num_coeffs):
        tmp = []
        for j in range(num_coeffs):
            if(abs(mean_coeffs[i]) < 0.00001 or abs(mean_coeffs[j]) < 0.00001):
                tmp.append(0.0)
                continue
            tot = 0.0
            for k in range(num_runs):
                X = (coeffs[i][k] - mean_coeffs[i]) / std[i]
                Y = (coeffs[j][k] - mean_coeffs[j]) / std[j]
                tot += X*Y
            tot /= num_runs
            tmp.append(tot)
        corr.append(tmp)
    return corr

def Comp_Error(M_test, y_test, x):
    lin = mat_vec_mul(M_test, x)
    err = []
    rel_err = []
    mean_error = 0.0
    for l,v in zip(lin,y_test):
        e = math.sqrt((l-v)*(l-v))
        mean_error += e
        rel_e = e/abs(v)
        err.append(e)
        rel_err.append(rel_e)
    mean_error /= float(len(lin))
    return err, rel_err, mean_error

def Sweep_Params(dat,model,alphas,num_split):
    M = dat.matrix
    y = dat.E_vec
    M_train, M_test, y_train, y_test = Split_Data(M,y,num_split)
    test_mats = dat.Test_to_Material_List(M_test)
    error = []
    for a in alphas:
        ridge = model(alpha=a,fit_intercept=False,max_iter=100000)
        ridge.fit(M_train,y_train)
        err, rel_err, mean_err = Comp_Error(M_test, y_test, ridge.coef_)
        error.append(mean_err)
    return error


class DeltaLinearModel:
    def __init__(self,working_dir,params_list,model_typ, num_tests) -> None:
        self.cwd = working_dir
        self.params = params_list
        self.alpha = 0.0
        self.num_test = num_tests
        self.model_name = model_typ
        self.coef = None
        self.coef_err = None
        self.test_error = 0
        self.data = GetData.Get_Data(self.params,'training')
        if(model_typ == 'Lasso'):
            self.model = linear_model.Lasso
        elif(model_typ == 'Ridge'):
            self.model = linear_model.Ridge

    def Set_Num_Test(self,num):
        self.num_test = num


    def Find_Alpha(self):
        #Sweep regularization strength
        alphas = np.logspace(-5, -2, 60)
        # for each alpha run LASSO on a series of test training splits
        err_list=[]
        num_iter = 10
        for i in range(num_iter):
            err = Sweep_Params(self.data, self.model,alphas,self.num_test)
            err_list.append(err)
        ave_errors = np.zeros([len(alphas)])
        for itr in err_list:
            for i,e in enumerate(itr):
                ave_errors[i] += e/num_iter
        self.Plot_Sweep(alphas,ave_errors)
        idx = np.argmin(ave_errors)
        self.alpha = alphas[idx]


    def Plot_Sweep(self,alphas, errors):
        idx = np.argmin(errors)
        min_alpha = alphas[idx]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.stem(alphas,errors)
        top = np.max(errors) + 0.1
        ax.text(.1, 0.9, f"Error is minimized at alpha = {min_alpha:.3e}",transform=ax.transAxes)
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        ax.set_ylim(-0.1,top)
        plt.xlabel("alpha")
        plt.ylabel("Error (eV)")
        plt.title(f"{self.model_name} Error vs Regularization Strength (alpha)")
        #plt.axis("tight")
        plt.savefig(f"{self.cwd}/{self.model_name}-alpha-Errors.png")
        plt.show()
        return

    def Run_LinerModel(self):
        M = self.data.matrix
        y = self.data.E_vec
        M_train, M_test, y_train, y_test = Split_Data(M,y,self.num_test)
        #test_mats = dat.Test_to_Material_List(M_test)
        reg = self.model(alpha=self.alpha,fit_intercept=False,max_iter=100000)
        reg.fit(M_train,y_train)
        err, rel_err, mean_err = Comp_Error(M_test, y_test, reg.coef_)
        return M_test, y_test, reg.coef_, mean_err

    def Run_Regression(self):
        num_iter = 25
        tmp_coefs = []
        mean_err = 0
        for i in range(num_iter):
            M, y, x, err = self.Run_LinerModel()
            mean_err += err
            tmp_coefs.append(x)
        # reshape the coefs
        mean_err /=num_iter
        self.test_error = mean_err
        final_coefs = []
        for i in range(len(x)):
            tmp = []
            for row in tmp_coefs:
                tmp.append(row[i])
            final_coefs.append(tmp)
        # compute the mean coef
        ave_coefs = np.zeros([len(x)])
        for i,c in enumerate(final_coefs):
            ave_coefs[i] = np.mean(c)
        self.coef = ave_coefs
        self.coef_err = Comp_STD(final_coefs,ave_coefs)

    def Run_Tests(self):
        ld = GetData.Get_Data(self.params, 'test')
        labs = ld.row_list
        M = ld.matrix
        y = ld.E_vec
        x = self.coef
        lin = mat_vec_mul(M, x)
        err = []
        rel_err = []
        mean_error = 0.0
        out = open(f"{self.cwd}/{self.model_name}-test-error.txt",'w')
        d1 = lin[1]-lin[0]
        d2 = lin[2]-lin[0]
        dE1 = y[1]-y[0]
        dE2 = y[2]-y[0]
        diff_E1 = dE1 - d1
        diff_E2 = dE2 - d2
        out.write(f"difference Error1 = {diff_E1}\n")
        out.write(f"difference Error2 = {diff_E2}\n")
        for l,v,lab in zip(lin,y,labs):
            raw = l-v
            e = math.sqrt((l-v)*(l-v))
            mean_error += e
            rel_e = e/abs(v)
            err.append(e)
            rel_err.append(rel_e)
            txt = f"{lab}: Rel Err = {rel_e} Err = {raw} (ev)\n"
            out.write(txt)
        mean_error /= float(len(lin))
        out.close()
        return err, rel_err, mean_error

    def Plot_coeffs(self,tag):
        labels = self.data.col_list
        label_dict, legend_labs, pp = Gen_Plot_Dict(labels)
        x = self.coef
        nums = np.arange(1,len(x)+1, 1.0)
        ax = plt.gca()
        tks = []
        tick_labs = []
        count = 2
        for l,v in label_dict.items():
            tick_labs.append(l)
            tks.append(count)
            count += len(v['markers'])
        count = 0
        for c,n,e,p in zip(x,nums,self.coef_err,pp):
            clr = p['color']
            mkr = p['marker']
            lab = p['label']
            ax.errorbar(n,c,yerr=e,color=clr,marker=mkr,label=lab)
        #ax.text(0.0, -3.0, f"Mean RMSE for Test Data = {mean_err:.3f} eV")
        ax.legend()
        plt.xticks(tks,tick_labs)
        plt.title(f"{self.model_name} Coef")
        plt.xlabel("Coef")
        plt.ylabel("value")
        plt.axis("tight")
        plt.savefig(f"{self.cwd}/{self.model_name}-Coeffs-{tag}.png")
        plt.show()



if __name__ == "__main__":
    test_dir = "../Testing"
    test(test_dir)




