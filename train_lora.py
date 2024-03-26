from trainer_lora import main, parse_args

config = parse_args()
config.cache_dir = ".cache"
config.report_to = "wandb"

if __name__ == '__main__':
    main(config)
