#!/bin/sh
#BSUB -q gpuv100
#BSUB -J nSphere_stism
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -u fmry@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o sendmeemail/error_%J.out 
#BSUB -e sendmeemail/output_%J.err 

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_score.py \
    --manifold nSphere \
    --dim 2 \
    --loss_type ism \
    --load_model 0 \
    --T_sample 0 \
    --t0 0.1 \
    --gamma 1.0 \
    --sigma 100.0 \
    --train_net st \
    --eps 0.01 \
    --T 1.0 \
    --lr_rate 0.001 \
    --epochs 50000 \
    --warmup_epochs 1000 \
    --x0_samples 1 \
    --xt_samples 512 \
    --dt_steps 100 \
    --save_step 100 \
    --seed 2712
