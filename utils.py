import numpy as np
import matplotlib.pyplot as plt
import os


def readSparseMatrix(filename, total_length):
    print ("reading Rao's HiC ")
    infile = open(filename).readlines()
    print (len(infile))
    HiC = np.zeros((total_length,total_length)).astype(np.int16)
    percentage_finish = 0
    for i in range(0, len(infile)):
        if (i %  (len(infile) / 10)== 0):
            print ('finish ', percentage_finish, '%')
            percentage_finish += 10
        nums = infile[i].split('\t')
        x = int(nums[0])
        y = int(nums[1])
        val = int(float(nums[2]))

        HiC[x][y] = val
        HiC[y][x] = val
    return HiC

def readSquareMatrix(filename, total_length):
    print ("reading Rao's HiC ")
    infile = open(filename).readlines()
    print('size of matrix is ' + str(len(infile)))
    print('number of the bins based on the length of chromsomes is ' + str(total_length) )
    result = []
    for line in infile:
        tokens = line.split(" ")
        tokens2=[]
        for item in tokens:
            k = float(item)
            tokens2.append(k)
        line_int = list(map(int, tokens2))
        result.append(line_int)
    result = np.array(result)
    print(result.shape)
    return result


def divide(HiCfile):
    subImage_size = 40
    step = 25
    chrs_length = [195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566]
    input_resolution = 10000
    result = []
    index = []
    chrN = 19
    matrix_name = HiCfile + '_npy_form_tmp.npy'
    if os.path.exists(matrix_name):
        print ('loading ', matrix_name)
        HiCsample = np.load(matrix_name)
    else:
        print (matrix_name, 'not exist, creating')
        print (HiCfile)
        HiCsample = readSquareMatrix(HiCfile, (chrs_length[chrN-1]/input_resolution + 1))
        #HiCsample = np.loadtxt('/home/zhangyan/private_data/IMR90.nodup.bam.chr'+str(chrN)+'.10000.matrix', dtype=np.int16)
        print (HiCsample.shape)
        np.save(matrix_name, HiCsample)
    print (HiCsample.shape)
    path = 'HiCPlus_pytorch_production/'
    if not os.path.exists(path):
        os.makedirs(path)
    total_loci = HiCsample.shape[0]
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (abs(i-j) > 201 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCsample[i:i+subImage_size, j:j+subImage_size]

            result.append([subImage,])
            tag = 'test'
            index.append((tag, chrN, i, j))
    result = np.array(result)
    print (result.shape)
    result = result.astype(np.double)
    index = np.array(index)
    return result, index


if __name__ == "__main__":
    main()
