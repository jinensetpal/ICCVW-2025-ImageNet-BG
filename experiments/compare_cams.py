#!/usr/bin/env python3

from src.model.arch import Model as CRCEModel
from A_generate_model_results import *
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad
def context_dependence_ratio(model, gen):
    fg_contribs = torch.tensor((), device='cuda:0')
    bg_contribs = torch.tensor((), device='cuda:0')

    is_crce = isinstance(model, CRCEModel)
    fc_weights = model.linear.weight if is_crce else model.fc.weight
    fc_bias = model.linear.bias if is_crce else model.fc.bias
    hooked_module = model.backbone.layer4[-1].relu if is_crce else model.layer4[-1].relu

    if (model.backbone.layer4[0].downsample[0].stride if is_crce else model.layer4[0].downsample[0].stride) == (1, 1):
        mask_resize = T.Resize(14)
    else:
        mask_resize = T.Resize(7)

    for X, (h, y) in tqdm(gen, total=len(gen)):
        X, y = X.to('cuda:0'), y.to('cuda:0')
        y_pred = model(X)

        hrc = (fc_weights @ hooked_module.feature_map.flatten(2)).view(*y_pred.shape, *hooked_module.feature_map.shape[2:]) / hooked_module.feature_map[0, 0].numel()
        assert torch.allclose(y_pred, hrc.sum(dim=[-1, -2]) + fc_bias.unsqueeze(0), atol=1e-5, rtol=1e-5)

        cams = hrc[tuple(torch.stack([torch.arange(y.numel(), device=y.device), y]).tolist())].unsqueeze(1) - hrc
        h = mask_resize(h.to('cuda:0')).unsqueeze(1)

        fg_contribs = (cams.abs() * h).mean(1).sum(dim=(1, 2))
        bg_contribs = (cams.abs() * ~h).mean(1).sum(dim=(1, 2))

        mask = (bg_contribs / (bg_contribs + fg_contribs)) > 0.5
        X, h, cams, y, y_pred = X[mask], h[mask], cams[mask], y[mask], y_pred[mask]

        for (X_i, h_i, cc_i, y_i, y_pred_i) in zip(X, h, cams, y, y_pred):
            fig = plt.figure(figsize=(14, 14), facecolor='white')
            fig.add_subplot(1, 2, 1)
            plt.imshow(X_i.squeeze().permute(1,2,0).cpu(), alpha=.6)
            plt.imshow(F.interpolate(h.unsqueeze(1), const.IMAGE_SIZE, mode='nearest-exact').squeeze().cpu(), alpha=.4)
            plt.xlabel('Image with ground-truth mask')

            fig.add_subplot(1, 2, 2)

            context_contribution = 100 * (cc * (1-heatmap)).sum() / cc.sum()
            cc = F.interpolate(cc.unsqueeze(1), const.IMAGE_SIZE, mode='bilinear').squeeze()

            plt.imshow(X.squeeze().permute(1,2,0).cpu(), alpha=.5)
            plt.imshow(cc.cpu(), alpha=.5, cmap='turbo', vmin=0)
            plt.xlabel(f'Context Contribution: {context_contribution:.3f}%\nIs correct? {(y_pred.argmax(-1) == y.argmax(-1)).item()}')

            plt.tight_layout()
            plt.savefig(f'hardinet_ce_crce_comparison/{idx}.png')


if __name__ == '__main__':
    model_funs = [
        partial(prepare__ResNet50, cam_size=14, run_idx=1),
        partial(prepare__ResNet50, cam_size=14, run_idx=2),
        partial(prepare__ResNet50, cam_size=14, run_idx=3),
        partial(prepare__ResNet50, cam_size=7, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=14, run_idx=3),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=1),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=2),
        partial(prepare__ResNet50_CFCE, cam_size=7, run_idx=3),
        partial(prepare__ResNet50_ABN, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260413_082437-imagenet_abn_14x14_2/files/best_valacc_0.77_epoch_1.ckpt', run_idx=2),
        partial(prepare__ResNet50_ABN, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260413_183334-imagenet_abn_14x14_3/files/best_valacc_0.77_epoch_1.ckpt', run_idx=3),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135435-imagenet_gals_14x14_trial_0/files/best_valacc_0.77_epoch_6.ckpt', cam_size=14, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260412_192926-imagenet_gals_14x14_2/files/best_valacc_0.77_epoch_6.ckpt', cam_size=14, run_idx=2),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260413_150021-imagenet_gals_14x14_3/files/best_valacc_0.77_epoch_6.ckpt', cam_size=14, run_idx=3),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260402_135449-imagenet_gals_7x7_trial_0/files/best_valacc_0.76_epoch_1.ckpt', cam_size=7, run_idx=1),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260413_001014-imagenet_gals_7x7_2/files/best_valacc_0.76_epoch_7.ckpt', cam_size=7, run_idx=2),
        partial(prepare__ResNet50_GALS, ckpt='/local/scratch/a/jsetpal/feature-alignment/GALS/wandb/imagenet/wandb/run-20260413_150036-imagenet_gals_7x7_3/files/best_valacc_0.76_epoch_7.ckpt', cam_size=7, run_idx=3),
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
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

        print(context_dependence_ratio(model, data_loader))
