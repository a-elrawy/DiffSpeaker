export CUDA_VISIBLE_DEVICES=0
python -m train \
    --cfg configs/diffusion/vox/diffspeaker_wav2vec2_vox.yaml \
    --cfg_assets configs/assets/vox.yaml \
    --batch_size 32 \
    --nodebug \
