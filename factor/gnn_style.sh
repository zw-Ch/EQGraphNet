#!/bin/bash

gnn_styles=("gan")
model_style=0
sm_scale="md"
device="cuda:1"

for gnn_style in "${gnn_styles[@]}"
do
  python gnn_style.py $gnn_style $model_style $sm_scale $device
done