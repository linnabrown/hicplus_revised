# python runHiCPlus.py --input_matrix ../../data/binned_data/contact_matrix/MY_113.MY_115/MY_113.MY_115.19.npy --model ../model/pytorch_HindIII_model_40000 --chrN 19

import os
from time import gmtime, strftime
from datetime import datetime
models = ["../model/pytorch_HindIII_model_40000", "../model/pytorch_model_12000"]
suf = "_simulation1_seed1558030855_frac125"
fw = open("../totalTime.log","w")
for model in models:
    mod = model.split("_")[-1] 
    startTime = datetime.now()
    for j in range(1,20):
        inM = "../../txt_data/divided_matrix/MY_113.MY_115%s/MY_113.MY_115%s.%d.npy"%(suf,suf,j)
        coM = "../../txt_data/divided_matrix/MY_113.MY_115/MY_113.MY_115.%d.npy"%(j)
        outD = "../../HiCPlus_output/MY_113.MY_115_model%s/MY_113.MY_115.%d"%(mod,j)
        os.system("python runHiCPlus.py --input_matrix %s\
                                    --compare_matrix   %s\
                                    --model            %s\
                                    --output_dir       %s\
                                    --chrN             %d"%(inM, coM, model, outD, j))
    endTime = datetime.now()
    totalChrTime = endTime - startTime
    fw.write("%s\t%s\t%s\n"%(inM,model,str(totalChrTime)))

fw.close()



