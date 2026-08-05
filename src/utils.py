import torch
import yaml


def get_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def load_config(path="configs/baseline.yaml"):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config()
    print(config['train'])

    