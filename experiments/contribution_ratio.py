#!/usr/bin/env python3

from torcheval.metrics import MulticlassAccuracy
from src.model.arch import Model as CRCEModel
from A_generate_model_results import *
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import matplotlib
matplotlib.use('pdf')

@torch.no_grad
def sample_cams(model, gen, cfce_model):
    metric = MulticlassAccuracy(device=torch.device('cuda:0'))

    synset_mapping = json.load(open('/local/scratch/b/mfdl/datasets/imagenet-bg/imagenet_class_index.json'))

    is_crce = isinstance(model, CRCEModel)
    fc_weights = model.linear.weight if is_crce else model.fc.weight
    fc_bias = model.linear.bias if is_crce else model.fc.bias
    hooked_module = model.backbone.layer4[-1].relu if is_crce else model.layer4[-1].relu

    if (model.backbone.layer4[0].downsample[0].stride if is_crce else model.layer4[0].downsample[0].stride) == (1, 1):
        mask_resize = T.Resize(14)
    else:
        mask_resize = T.Resize(7)

    for batch_idx, (X, (h, y)) in tqdm(enumerate(gen), total=len(gen)):
        X = X.to('cuda:0')
        y_pred = model(X)
        y = y.to('cuda:0')

        hrc = (fc_weights @ hooked_module.feature_map.flatten(2)).view(*y_pred.shape, *hooked_module.feature_map.shape[2:]) / hooked_module.feature_map[0, 0].numel()
        assert torch.allclose(y_pred, hrc.sum(dim=[-1, -2]) + fc_bias.unsqueeze(0), atol=1e-5, rtol=1e-5)

        cams = hrc[tuple(torch.stack([torch.arange(y.numel(), device=y.device), y]).tolist())].unsqueeze(1) - hrc
        h = mask_resize(h.to('cuda:0')).unsqueeze(1)

        fg_contribs = (cams.abs() * h).mean(1).sum(dim=(1, 2))
        bg_contribs = (cams.abs() * ~h).mean(1).sum(dim=(1, 2))

        contribution_ratios = bg_contribs / (fg_contribs + bg_contribs)

        y_pred_cfce = cfce_model(X)
        hrc_cfce = (cfce_model.linear.weight @ cfce_model.backbone.layer4[-1].relu.feature_map.flatten(2)).view(*y_pred.shape, *cfce_model.backbone.layer4[-1].relu.feature_map.shape[2:]) / cfce_model.backbone.layer4[-1].relu.feature_map[0, 0].numel()
        assert torch.allclose(y_pred_cfce, hrc_cfce.sum(dim=[-1, -2]) + cfce_model.linear.bias.unsqueeze(0), atol=1e-5, rtol=1e-5)

        cams_cfce = hrc_cfce[tuple(torch.stack([torch.arange(y.numel(), device=y.device), y]).tolist())].unsqueeze(1) - hrc_cfce
        fg_contribs = (cams_cfce.abs() * h).mean(1).sum(dim=(1, 2))
        bg_contribs = (cams_cfce.abs() * ~h).mean(1).sum(dim=(1, 2))

        contribution_ratios_cfce = bg_contribs / (fg_contribs + bg_contribs)

        mask = (contribution_ratios > .5) & (h.sum(dim=[-1, -2, -3]) != 0) & (y_pred.argmax(-1) == y) & (contribution_ratios_cfce < .5) & (y_pred_cfce.argmax(-1) == y)
        for idx, (X_i, h_i, y_i, y_pred_i, cams_i, context_ratio, y_pred_cfce_i, cfce_cams_i, cfce_context_ratio) in enumerate(zip(X[mask], h[mask], y[mask], y_pred[mask], cams[mask], contribution_ratios[mask], y_pred_cfce[mask], cams_cfce[mask], contribution_ratios_cfce[mask])):
            X_viz = X_i - X_i.min()
            X_viz /= X_viz.max()

            fig = plt.figure(figsize=(10, 5))
            fig.add_subplot(1, 3, 1)
            plt.imshow(X_viz.cpu().permute(1,2,0), alpha=.6)
            plt.imshow(F.interpolate(h_i.to(torch.float).unsqueeze(1), const.IMAGE_SIZE, mode='nearest-exact').squeeze().cpu(), alpha=.4)
            plt.xlabel(','.join(synset_mapping[str(y_i.item())]))

            fig.add_subplot(1, 3, 2)
            plt.imshow(X_viz.cpu().permute(1,2,0), alpha=.6)
            plt.imshow(F.interpolate(cams_i.abs().mean(0, keepdim=True).unsqueeze(1), const.IMAGE_SIZE, mode='bilinear').squeeze().cpu(), alpha=.5, vmin=0, cmap='turbo')
            plt.xlabel(f'Context Contribution: {context_ratio*100:.3f}%\nConf: {(y_pred_i.softmax(-1).max()).item()}')

            fig.add_subplot(1, 3, 3)
            plt.imshow(X_viz.cpu().permute(1,2,0), alpha=.5)
            plt.imshow(F.interpolate(cfce_cams_i.abs().mean(0, keepdim=True).unsqueeze(1), const.IMAGE_SIZE, mode='bilinear').squeeze().cpu(), alpha=.5, vmin=0, cmap='turbo')
            plt.xlabel(f'Context Contribution: {cfce_context_ratio*100:.3f}%\nConf: {(y_pred_cfce_i.softmax(-1).max()).item()}')

            plt.savefig(f'comparisons/{batch_idx}_{idx}.png', pad_inches=0, bbox_inches='tight')
            plt.close()

            plt.imshow(X_viz.permute(1,2,0).cpu())
            plt.yticks([])
            plt.xticks([])
            plt.axis('tight')
            plt.axis('image')
            plt.axis('off')
            plt.savefig(f'comparisons/{batch_idx}_{idx}_X.png', pad_inches=0, bbox_inches='tight')
            plt.close()

            plt.imshow(X_viz.cpu().permute(1,2,0), alpha=.5)
            plt.imshow(F.interpolate(cams_i.abs().mean(0, keepdim=True).unsqueeze(1), const.IMAGE_SIZE, mode='bilinear').squeeze().cpu(), alpha=.5, vmin=0, cmap='turbo')

            plt.yticks([])
            plt.xticks([])
            plt.axis('tight')
            plt.axis('image')
            plt.axis('off')
            plt.savefig(f'comparisons/{batch_idx}_{idx}_cam.png', pad_inches=0, bbox_inches='tight')
            plt.close()

            plt.imshow(X_viz.cpu().permute(1,2,0), alpha=.5)
            plt.imshow(F.interpolate(cfce_cams_i.abs().mean(0, keepdim=True).unsqueeze(1), const.IMAGE_SIZE, mode='bilinear').squeeze().cpu(), alpha=.5, vmin=0, cmap='turbo')

            plt.yticks([])
            plt.xticks([])
            plt.axis('tight')
            plt.axis('image')
            plt.axis('off')
            plt.savefig(f'comparisons/{batch_idx}_{idx}_cfce_cam.png', pad_inches=0, bbox_inches='tight')
            plt.close()


@torch.no_grad
def context_dependence_ratio(model, gen):
    metric = MulticlassAccuracy(device=torch.device('cuda:0'))
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
        y_pred = model(X.to('cuda:0'))
        y = y.to('cuda:0')

        metric.update(y_pred, y)

        hrc = (fc_weights @ hooked_module.feature_map.flatten(2)).view(*y_pred.shape, *hooked_module.feature_map.shape[2:]) / hooked_module.feature_map[0, 0].numel()
        assert torch.allclose(y_pred, hrc.sum(dim=[-1, -2]) + fc_bias.unsqueeze(0), atol=1e-5, rtol=1e-5)

        cams = hrc[tuple(torch.stack([torch.arange(y.numel(), device=y.device), y]).tolist())].unsqueeze(1) - hrc
        h = mask_resize(h.to('cuda:0')).unsqueeze(1)

        fg_contribs = torch.cat([fg_contribs, (cams.abs() * h).mean(1).sum(dim=(1, 2))])
        bg_contribs = torch.cat([bg_contribs, (cams.abs() * ~h).mean(1).sum(dim=(1, 2))])

    fg_contribs = fg_contribs.mean().item()
    bg_contribs = bg_contribs.mean().item()
    return fg_contribs, bg_contribs, fg_contribs / (fg_contribs + bg_contribs), metric.compute().item() * 100


if __name__ == '__main__':
    model_name, model, preprocess = prepare__ResNet50(cam_size=14, run_idx=1)
    _, cfce_model, _ = prepare__ResNet50_CFCE(cam_size=14, run_idx=1)
    print(f"> {model_name}")

    dataset = ImageNetBG(
        root_dir=IMAGENET_BG_PATH,
        labels_file='./imagenet_class_index.json',
        transform=preprocess,
        return_masks=True,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    sample_cams(model, data_loader, cfce_model)
    #print(context_dependence_ratio(model, data_loader))
