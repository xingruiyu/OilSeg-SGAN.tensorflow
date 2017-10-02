
#!/usr/bin/env bash
PRJROOT=${PWD}
rootpath=datasets
expname=oilseg_sgan
##
for lr in 0.0002
do
    for dataset in syntheticSAR/salt_and_pepper syntheticSAR/multiplicative
    do
        for f in pearson variational helinger cap symkl  
	do
#    : << !
        echo "${exptype}_${f} training"
        python ${PRJROOT}/oilseg_sgan.py \
          --mode train \
          --output_dir ${PRJROOT}/checkpoints.lr.${lr}/${expname}/${dataset}/${f} \
          --max_epochs 200 \
          --input_dir ${rootpath}/${dataset}/train \
          --which_direction AtoB \
          --f_type ${f} \
	  --test_freq 100 \
	  --batch_size 1 \
          --gan_weight 1.0 \
          --lr ${lr}
#!
# : << !
        echo "${exptype}_${f} testing"
        python ${PRJROOT}/oilseg_sgan.py \
          --mode test \
          --output_dir ${PRJROOT}/results.lr.${lr}/${expname}/${dataset}/${f} \
          --input_dir ${rootpath}/${dataset}/test \
          --checkpoint ${PRJROOT}/checkpoints.lr.${lr}/${expname}/${dataset}/${f} \
          --f_type ${f} \
          --lr ${lr}
#!


	done
    done
done
