# DSPN-CL Windows demo run (PowerShell)
# If you do not have a GPU, change --gpu=0 to --gpu=-1.
# If your data is in a different path, run from the repo root.

python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl
