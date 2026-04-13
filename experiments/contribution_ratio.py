#!/usr/bin/env python3

from torcheval.metrics import MulticlassAccuracy
from src.model.arch import Model as CRCEModel
from A_generate_model_results import *
import torchvision.transforms.v2 as T
from tqdm import tqdm

@torch.no_grad
def context_dependence_ratio(model, gen):
    metric = MulticlassAccuracy(device=torch.device('cuda:0'))
    fg_contribs = torch.tensor((), device='cuda:0')
    bg_contribs = torch.tensor((), device='cuda:0')

    is_crce = isinstance(model, CRCEModel)
    fc_weights = model.linear.weight if is_crce else model.fc.weight
    hooked_module = model.backbone.layer4[-1].relu if is_crce else model.layer4[-1].relu

    if (model.backbone.layer4[0].downsample[0].stride if is_crce else model.layer4[0].downsample[0].stride) == (1, 1):
        mask_resize = T.Resize(14)
    else:
        mask_resize = T.Resize(7)

    for X, (h, y) in tqdm(gen, total=len(gen)):
        y_pred = model(X.to('cuda:0'))
        y = y.to('cuda:0')

        metric.update(y_pred, y)

        hrc = (fc_weights @ hooked_module.feature_map.flatten(2)).view(*y_pred.shape, *hooked_module.feature_map.shape[2:]) / hooked_module.feature_map[0, 0].numel()
        cams = hrc[tuple(torch.stack([torch.arange(y.numel(), device=y.device), y]).tolist())].unsqueeze(1) - hrc
        h = mask_resize(h.to('cuda:0'))

        fg_contribs = torch.cat([fg_contribs, (cams.abs() * h).sum(dim=(1, 2, 3))])
        bg_contribs = torch.cat([bg_contribs, (cams.abs() * ~h).sum(dim=(1, 2, 3))])

    fg_contribs = fg_contribs.mean().item()
    bg_contribs = bg_contribs.mean().item()
    return fg_contribs, bg_contribs, fg_contribs / (fg_contribs + bg_contribs), metric.compute().item() * 100


if __name__ == '__main__':
    model_funs = [
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=3),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=3),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135435-imagenet_gals_14x14_trial_0/files/best_valacc_0.77_epoch_6.ckpt', cam_size=14, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135449-imagenet_gals_7x7_trial_0/files/best_valacc_0.76_epoch_1.ckpt', cam_size=7, run_idx=1),
        partial(prepare__ResNet50_ABN, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135522-imagenet_abn_14x14_trial_0/files/best_valacc_0.77_epoch_1.ckpt', run_idx=1),
        prepare__ResNet50,
    ]

    for model_fun in model_funs:
        model_name, model, preprocess = model_fun()
        print(f"> {model_name}")

        dataset = ImageNetBG(
            root_dir=IMAGENET_BG_PATH,
            labels_file='./imagenet_class_index.json',
            transform=preprocess,
            return_masks=True,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

        print(context_dependence_ratio(model, data_loader))
