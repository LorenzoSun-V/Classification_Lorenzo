import torch.nn as nn
import torch
from collections import OrderedDict


def load_weights(model: nn.Module, model_url: str):
	state_dict = torch.load(model_url, map_location=lambda storage, loc: storage)
	model_dict = model.state_dict()
	new_state_dict = OrderedDict()
	matched_layers, discarded_layers = [], []

	for k, v in state_dict.items():
		if k.startswith('module.'):
			k = k[7:]
		elif k.startswith('features.'):
			k = k[9:]
		if k in model_dict and model_dict[k].size() == v.size():
			new_state_dict[k] = v
			matched_layers.append(k)
		else:
			discarded_layers.append(k)

	model_dict.update(new_state_dict)
	model.load_state_dict(model_dict)

	if len(matched_layers) == 0:
		print(f'Error: The pretrained weights from "{model_url}" cannot be loaded')
		exit(0)
	else:
		print(f'Successfully loaded imagenet pretrained weights from {model_url}')
		if len(discarded_layers) > 0:
			print('** The following layers are discarded '
				f'due to unmatched keys or layer size: {discarded_layers}')