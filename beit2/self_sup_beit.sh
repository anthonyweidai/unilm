# run in unilm repot
python -m torch.distributed.launch --nproc_per_node=1 run_beitv2_pretraining.py \
        --data_set CIFAR \
        # --data_path /dataset/STL10_unlabelled \
        --output_dir /exp/BEiT \
        --log_dir /exp/BEiT \
        --model beit_base_patch16_192_8k_vocab \
        --shared_lm_head True \
        --early_layers 9 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --second_input_size 96 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --tokenizer_model vqkd_encoder_base_decoder_3x768x12_clip \
        --tokenizer_weight https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D \
        --batch_size 1024 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0.1 \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.999 \
        --opt_eps 1e-8  \
        --epochs 200 \
        --save_ckpt_freq 20 