from trainer_full import main, parse_args

config = parse_args()
config.cache_dir = ".cache"
config.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
config.resolution = 512
config.control_lora_linear_rank = 32
config.control_lora_conv2d_rank = 32
config.learning_rate = 2e-5
config.train_batch_size = 4
config.max_train_steps = 2500
config.enable_xformers_memory_efficient_attention = False
config.checkpointing_steps = 5000
config.validation_steps = 5000
config.report_to = "wandb"
# config.resume_from_checkpoint = "latest"
config.resume_from_checkpoint = None
# config.push_to_hub = True

config.custom_dataset = "custom_datasets.tutorial.MyDataset"
config.conditioning_image_column = "hint"
config.image_column = "jpg"
config.caption_column = "txt"
config.num_validation_samples = 3
config.conditioning_type_name = "fill50k"

config.tracker_project_name = f"sd-control-lora-{config.conditioning_type_name}"
config.output_dir = f"output/{config.tracker_project_name}"

config.seed = 0

if __name__ == '__main__':
    main(config)
