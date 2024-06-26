export CUDA_VISIBLE_DEVICES=1

# # use hubert backbone
# python demo_vox.py \
#     --cfg configs/diffusion/vox/diffspeaker_hubert_vox.yaml \
#     --cfg_assets configs/assets/vox.yaml \
#     --template datasets/vox/templates.pkl \
#     --example demo/wavs/speech_british.wav \
#     --ply datasets/vox/templates/FLAME_sample.ply \
#     --checkpoint checkpoints/vox/diffspeaker_hubert_vox.ckpt \
#     --id FaceTalk_170809_00138_TA

# use wav2vec2 backbone
python demo_vox.py \
    --cfg configs/diffusion/vox/diffspeaker_wav2vec2_vox.yaml \
    --cfg_assets configs/assets/vox.yaml \
    --template datasets/vocaset/templates.pkl \
    --example demo/wavs/speech_british.wav \
    --ply datasets/vocaset/templates/FLAME_sample.ply \
    --checkpoint checkpoints/vox/diffspeaker_wav2vec2_vox.ckpt \
    --id FaceTalk_170809_00138_TA

