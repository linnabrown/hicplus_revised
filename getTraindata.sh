
#!/usr/bin/bash

for i in {7,8}
do
    python ~/work/Rotation3/myHiCplus/HiCPlus_pytorch/src/trainPipe.py --input_file GM12878.chr$i.obs.KR.10k.txtsquare.txt --chrN $i
    echo "chr$i completed"
done
