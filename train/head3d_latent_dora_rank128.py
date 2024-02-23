from trainer import main, parse_args

config = parse_args()
config.cache_dir = ".cache"
config.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
config.resolution = 384 # 512
config.control_lora_linear_rank = 128
config.control_lora_conv2d_rank = 128
config.learning_rate = 1e-4
config.train_batch_size = 4
config.max_train_steps = 75000
config.enable_xformers_memory_efficient_attention = True
config.checkpointing_steps = 5000
config.validation_steps = 5000
config.report_to = "wandb"
config.resume_from_checkpoint = "latest"
# config.push_to_hub = True

config.custom_dataset = "custom_datasets.head3d.Head3dDataset"
config.conditioning_image_column = "hint"
config.image_column = "jpg"
config.caption_column = "txt"
config.num_validation_samples = 3
config.conditioning_type_name = "head3d"
config.proportion_empty_prompts = 0.1

config.tracker_project_name = f"sd-latent-control-dora-rank128-{config.conditioning_type_name}"
config.output_dir = f"output/{config.tracker_project_name}"
config.use_conditioning_latent = True
config.use_dora = True

if __name__ == '__main__':
    main(config)
