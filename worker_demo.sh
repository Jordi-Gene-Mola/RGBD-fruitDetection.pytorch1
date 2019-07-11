#!/bin/bash

param[80]="--dataset kinect_fruits_k --net vgg16  --RGB  --checksession 80 --checkpoint 309  --checkepoch 5  --anchor 4  --anchor_ratio 1  --minconfid 0.85 --image_dir images_kinect_fruits_k --cuda"
param[81]="--dataset kinect_fruits_k --net vgg16  --NIR  --checksession 81 --checkpoint 309  --checkepoch 4  --anchor 4  --anchor_ratio 1  --minconfid 0.85 --image_dir images_kinect_fruits_k --cuda"
param[82]="--dataset kinect_fruits_k --net vgg16  --DEPTH --checksession 82 --checkpoint 309  --checkepoch 7  --anchor 4  --anchor_ratio 1  --minconfid 0.85 --image_dir images_kinect_fruits_k --cuda"
param[83]="--dataset kinect_fruits_k --net vgg16_4ch  --RGB --NIR  --checksession 83 --checkpoint 309  --checkepoch 9  --anchor 4  --anchor_ratio 1  --minconfid 0.85 --image_dir images_kinect_fruits_k --cuda"
param[86]="--dataset kinect_fruits_k --net vgg16_5ch  --RGB --NIR --DEPTH  --checksession 86 --checkpoint 309  --checkepoch 11  --anchor 4  --anchor_ratio 1  --minconfid 0.85 --image_dir images_kinect_fruits_k --cuda"
param[67]="--dataset kinect_fruits --net vgg16_5ch  --RGB  --DEPTH  --NIR   --checksession 67 --checkpoint 309  --checkepoch 12  --anchor 4  --anchor_ratio 1    --minconfid 0.85 --image_dir images_kinect_fruits --cuda"

python demo.py ${param[SLURM_ARRAY_TASK_ID]}
