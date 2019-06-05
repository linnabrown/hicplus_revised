import numpy as np
import os


#chr7 = np.load('GM12878.chr7.obs.KR.10k.txthighres.npy')
#chr8 = np.load('GM12878.chr8.obs.KR.10k.txthighres.npy')

#new = np.concatenate((chr7, chr8), axis=0)
#print(new.shape)

#np.save('GM12878.chr7-8.obs.KR.10k.txthighres.npy', new)

chr7 = np.load('GM12878.chr7.obs.KR.10k.txtlowres.npy')
chr8 = np.load('GM12878.chr8.obs.KR.10k.txtlowres.npy')

new = np.concatenate((chr7, chr8), axis=0)
print(new.shape)

np.save('GM12878.chr7-8.obs.KR.10k.txtlowres.npy', new)

