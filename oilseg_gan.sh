
#!/usr/bin/env bash
PRJROOT=${PWD}
rootpath=datasets

for exp in oilseg_gan 
do
    for dataset in syntheticSAR/salt_and_pepper syntheticSAR/multiplicative 
    do
        for lr in 0.0002 
	do
#:<< !
            python ${PRJROOT}/${exp}.py \
              --mode train \
              --output_dir ${PRJROOT}/checkpoints.lr.${lr}/${exp}/${dataset}/ \
              --max_epochs 200 \
              --input_dir ${rootpath}/${dataset}/train \
              --which_direction AtoB \
	      --display_freq 100 \
	      --batch_size 1 \
              --gan_weight 1.0 \
              --lr ${lr}
#!
#: << !
            python ${PRJROOT}/${exp}.py \
              --mode test \
              --output_dir ${PRJROOT}/results.lr.${lr}/${exp}/${dataset}/${f} \
              --input_dir ${rootpath}/${dataset}/test \
              --checkpoint ${PRJROOT}/checkpoints.lr.${lr}/${exp}/${dataset}/${f} \
              --lr ${lr}
#!

	done
    done
done
