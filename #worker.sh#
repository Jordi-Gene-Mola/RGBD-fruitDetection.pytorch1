#!/bin/bash

param[65]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 65 --anchor 4 --anchor 8 --anchor 16 --anchor_ratio 1 --cuda"
param[66]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 66 --anchor 2 --anchor_ratio 1 --cuda"
param[67]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 67 --anchor 4 --anchor_ratio 1 --cuda"
param[68]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 68 --anchor 8 --anchor_ratio 1 --cuda"
param[69]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 69 --anchor 16 --anchor_ratio 1 --cuda"
param[70]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 70 --anchor 32 --anchor_ratio 1 --cuda"
param[71]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 71 --anchor 2 --anchor 4 --anchor 8 --anchor_ratio 1 --cuda"
param[72]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 72 --anchor 2 --anchor 4 --anchor 8 --anchor 16 --anchor_ratio 1 --cuda"
param[73]="--dataset kinect_fruits --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 73 --anchor 8 --anchor 16 --anchor 32 --anchor_ratio 1 --cuda"

param[74]="--dataset kinect_fruits --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --epochs 30 --o adam --s 74 --anchor 4 --anchor_ratio 1 --cuda"
param[75]="--dataset kinect_fruits --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --NIR --epochs 30 --o adam --s 75 --anchor 4 --anchor_ratio 1 --cuda"
param[76]="--dataset kinect_fruits --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --DEPTH --epochs 30 --o adam --s 76 --anchor 4 --anchor_ratio 1 --cuda"
param[77]="--dataset kinect_fruits --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --epochs 30 --o adam --s 77 --anchor 4 --anchor_ratio 1 --cuda"
param[78]="--dataset kinect_fruits --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --DEPTH --epochs 30 --o adam --s 78 --anchor 4 --anchor_ratio 1 --cuda"
param[79]="--dataset kinect_fruits --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --NIR --DEPTH --epochs 30 --o adam --s 79 --anchor 4 --anchor_ratio 1 --cuda"
param[80]="--dataset kinect_fruits_k --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --epochs 30 --o adam --s 80 --anchor 4 --anchor_ratio 1 --cuda"
param[81]="--dataset kinect_fruits_k --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --NIR --epochs 30 --o adam --s 81 --anchor 4 --anchor_ratio 1 --cuda"
param[82]="--dataset kinect_fruits_k --net vgg16 --bs 4 --lr 0.0001 --lr_decay_step 10 --DEPTH --epochs 30 --o adam --s 82 --anchor 4 --anchor_ratio 1 --cuda"
param[83]="--dataset kinect_fruits_k --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --epochs 30 --o adam --s 83 --anchor 4 --anchor_ratio 1 --cuda"
param[84]="--dataset kinect_fruits_k --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --DEPTH --epochs 30 --o adam --s 84 --anchor 4 --anchor_ratio 1 --cuda"
param[85]="--dataset kinect_fruits_k --net vgg16_4ch --bs 4 --lr 0.0001 --lr_decay_step 10 --NIR --DEPTH --epochs 30 --o adam --s 85 --anchor 4 --anchor_ratio 1 --cuda"
param[86]="--dataset kinect_fruits_k --net vgg16_5ch --bs 4 --lr 0.0001 --lr_decay_step 10 --RGB --NIR --DEPTH --epochs 30 --o adam --s 86 --anchor 4 --anchor_ratio 1 --cuda"

python trainval_net.py ${param[SLURM_ARRAY_TASK_ID]}