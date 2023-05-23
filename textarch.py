import torch.nn as nn


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
        
def parse_architecture(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    encoder_layers, decoder_layers = [], []
    current_section = "encoder"
    latent_dim = 0

    for line in lines:
        line = line.strip()
        if line.startswith("latent"):
            current_section = "decoder"
            latent_dim = int(line.split()[1])
            continue

        tokens = line.split()
        layer_type = tokens[0]
        units = str_to_num(tokens[1])
        activation = tokens[2] if len(tokens) > 2 else None
        layer_info = (layer_type, units, activation)

        if current_section == "encoder":
            encoder_layers.append(layer_info)
        else:
            decoder_layers.append(layer_info)

    return encoder_layers, decoder_layers, latent_dim