import json
import os

def record_config(args):
    save_path = os.path.join(args.result_path, "config.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config_dict = vars(args)
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4, default=str)
    print(f"Configuration saved to {save_path}")
    for key, value in config_dict.items():
        print(f"{key}: {value}")
