import os,sys
import deltalearn as dl

def gen_test():
    out_file = "Data/Tests.json"
    dl.GetData.Write_Tests_JSON(out_file)

def update():
    dl.GetData.Update_Data(params="all")

def main():
    #Read in the data
    model_typ = 'Lasso'
    params = ['count','mu_s', 'U_s','mu_d', 'U_d']
    working_dir = "RPA-U"
    num_tests = 10
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    model = dl.DeltaLearn.DeltaLinearModel(working_dir,params,model_typ,num_tests)
    model.Find_Alpha()
    model.Run_Regression()
    model.Plot_coeffs("Lasso-rpa-count-U")
    model.Run_Tests()



if __name__ == '__main__':
    main()
