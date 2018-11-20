CUDA_VISIBLE_DEVICES=2,3 \
python -u main.py  \
    --num_classes 431 \
    -b 64 \
    --resume models/model_best.pth.tar \
    --eval
