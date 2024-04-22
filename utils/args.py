def save_config_to_args(config, args):
    """Save config back into args for convenience."""
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

