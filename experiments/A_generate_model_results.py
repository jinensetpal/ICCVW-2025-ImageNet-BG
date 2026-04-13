import torch
import torchvision

import numpy as np
import pandas as pd

import os.path
from tqdm import tqdm
from pathlib import Path
from functools import partial

import torchvision.transforms.v2 as T
from imagenet_bg import ImageNetBG

from src.model.arch import Model
from src import const

from models.resnet_abn import resnet50 as resnet50_abn
from models.resnet import resnet50


IMAGENET_BG_PATH = '/local/scratch/b/mfdl/datasets/imagenet-bg/images/'
BATCH_SIZE = 1024
const.N_CLASSES = 1000

transforms = T.Compose([T.PILToTensor(),
    T.Resize((232, 232), interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(const.IMAGE_SIZE),
    T.ToDtype(torch.float32, scale=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def hook(module, i, o):
    module.feature_map = o

def prepare__ResNet152_CFCE(cam_size=7, run_idx=1):
    const.XL_BACKBONE = True
    const.CAM_SIZE = (cam_size, cam_size)
    const.UPSAMPLING_LEVEL = 1 if cam_size == 14 else 0
    run_suffix = '' if run_idx == 1 else str(run_idx)

    model = Model(return_logits=True, logits_only=True)
    model.name = f'imagenet_contrastive_ablated_only_{cam_size}x{cam_size}cam_xlback_pretrained{run_suffix}'
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{model.name}/cfce_best.pt', map_location=model.device, weights_only=True))
    model.backbone.layer4[-1].relu.register_forward_hook(hook)

    model.eval()
    model.cuda()

    return f"ResNet152_CFCE_{cam_size}x{cam_size}_{run_idx}", model, transforms

def prepare__ResNet50_GALS(ckpt, cam_size=7, run_idx=1):
    run_suffix = '' if run_idx == 1 else str(run_idx)

    model = resnet50(pretrained=False, return_fmaps=False, upsample_cams=cam_size == 14)
    model.load_state_dict(torch.load(ckpt, weights_only=False)['model_state_dict'])
    model.layer4[-1].relu.register_forward_hook(hook)

    model.eval()
    model.cuda()

    return f"ResNet50_GALS_{cam_size}x{cam_size}_{run_idx}", model, transforms

def prepare__ResNet50_ABN(ckpt, run_idx=1):
    model = resnet50_abn(pretrained=False, num_classes=1000, add_after_attention=True)
    model.load_state_dict(torch.load(ckpt, weights_only=False)['model_state_dict'])
    model.layer4[-1].relu.register_forward_hook(hook)

    model.real_fwd = model.forward
    def forward(x):
        return model.real_fwd(x)[1]
    model.forward = forward

    model.eval()
    model.cuda()

    return f"ResNet50_ABN_14x14_{run_idx}", model, transforms

def prepare__ResNet50_CFCE(cam_size=7, run_idx=1):
    const.XL_BACKBONE = False
    const.CAM_SIZE = (cam_size, cam_size)
    const.UPSAMPLING_LEVEL = 1 if cam_size == 14 else 0
    run_suffix = '' if run_idx == 1 else str(run_idx)

    model = Model(return_logits=True, logits_only=True)
    model.name = f'imagenet_contrastive_ablated_only_{cam_size}x{cam_size}cam_pretrained{run_suffix}'
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{model.name}/cfce_best.pt', map_location=model.device, weights_only=True))
    model.backbone.layer4[-1].relu.register_forward_hook(hook)

    model.eval()
    model.cuda()

    return f"ResNet50_CFCE_{cam_size}x{cam_size}_{run_idx}", model, transforms

def prepare__ResNet50():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    model.layer4[-1].relu.register_forward_hook(hook)

    model.eval()
    model.cuda()

    return "ResNet50", model, preprocess

def prepare__ResNet152():
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()
    model.layer4[-1].relu.register_forward_hook(hook)

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

if __name__ == '__main__':
    model_funs = [
        partial(prepare__ResNet152_CFCE, cam_size=14, run_idx=3),
        partial(prepare__ResNet152_CFCE, cam_size=7, run_idx=3),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=3),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=3),
        partial(prepare__ResNet152_CFCE, cam_size=14, run_idx=2),
        partial(prepare__ResNet152_CFCE, cam_size=7, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=2),
        partial(prepare__ResNet152_CFCE, cam_size=14, run_idx=1),
        partial(prepare__ResNet152_CFCE, cam_size=7, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='~/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135435-imagenet_gals_14x14_trial_0/files/best_valacc_0.77_epoch_6.ckpt', cam_size=14, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='~/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135449-imagenet_gals_7x7_trial_0/files/best_valacc_0.76_epoch_1.ckpt', cam_size=7, run_idx=1),
        partial(prepare__ResNet50_ABN, ckpt='~/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135522-imagenet_abn_14x14_trial_0/files/best_valacc_0.77_epoch_1.ckpt', run_idx=1),
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
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

        df = calculate_model_df(model, data_loader)
        df.to_pickle(f"./model_results/{model_name}.pickle")
