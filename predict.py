import argparse
import torch
import MyFunction
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
	description = 'This is to use the network identifying the given data'
	)

parser.add_argument(action = 'store',dest = 'image_path')
parser.add_argument(action = 'store',dest = 'checkpoint')
parser.add_argument(action = 'store',dest = 'model_select')
parser.add_argument('--top_k', action = 'store',dest = 'K', type = int)
parser.add_argument('--category_names', action = 'store',dest = 'category_name')
parser.add_argument('--gpu', action = 'store_true',dest = 'device', default = False)


result = parser.parse_args()

with open(result.category_name, 'r') as f:
    cat_to_name = json.load(f)


device = torch.device("cuda" if result.device else "cpu")
model = MyFunction.load_checkpoint(result.checkpoint, result.model_select)

im = Image.open(result.image_path)
np_image = MyFunction.process_image(im)
inputs_02 = torch.from_numpy(np_image)

probs,classes = MyFunction.predict(result.image_path, model, result.K, device)
print(probs)
print(classes)

flower_name = [cat_to_name.get(key) for key in classes]
print(flower_name)

ps = probs.squeeze()
fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(im)
ax1.axis('off')
ax2.barh(np.arange(K), ps)
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(K))
ax2.set_yticklabels(flower_name, size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()

