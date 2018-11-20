CUDA_VISIBLE_DEVICES=2,3 \
python -u main.py  \
    --num_classes 431 \
    -b 64 \
|& tee logs/log.txt
