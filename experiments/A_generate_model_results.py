import torch
import torchvision

import numpy as np
import pandas as pd

import os.path
from tqdm import tqdm
from pathlib import Path

from imagenet_bg import ImageNetBG

IMAGENET_BG_PATH = '/local/scratch/b/mfdl/datasets/imagenet-bg/images/'
BATCH_SIZE = 1024

def prepare__ResNet50():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    model.eval()
    model.cuda()

    return "ResNet50", model, preprocess

def prepare__ResNet152():
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()

    model.eval()
    model.cuda()

    return "ResNet152", model, preprocess

def calculate_model_df(model, data_loader):
    all_out = []
    data_id = 0
    for i_batch, (inputs, targets, infos) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            logits = model(inputs.cuda())
            for i in range(len(inputs)):
                out = {}
                out["id"] = data_id
                out["label"] = targets[i].item()
                out["quality"] = infos[0][i]
                out["type"] = infos[1][i]
                out["background"] = infos[2][i]
                out["background_real_type"] = infos[3][i]
                out["logits"] = np.array(logits[i].detach().cpu())
                all_out.append(out)
                data_id += 1

    df = pd.DataFrame(all_out)
    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")

    return df

model_funs = [
    prepare__ResNet152,
    prepare__ResNet50,
]

Path("./model_results").mkdir(parents=True, exist_ok=True)

for model_fun in model_funs:
    model_name, model, preprocess = model_fun()
    print(f"> {model_name}")
    if os.path.isfile(f"./model_results/{model_name}.pickle"):
        continue

    dataset = ImageNetBG(
        root_dir=IMAGENET_BG_PATH,
        labels_file='./imagenet_class_index.json',
        transform=preprocess
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    df = calculate_model_df(model, data_loader)
    df.to_pickle(f"./model_results/{model_name}.pickle")
