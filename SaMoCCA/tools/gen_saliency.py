r""" Offline saliency generation with a frozen U^2-Net.

Generates dense, class-agnostic saliency maps for support images. PASCAL maps
already exist under `datasets/VOC2012/U2net_pascalAug/`; this script targets
COCO and FSS (and can regenerate PASCAL if needed).

Saliency layout produced (mirrors the image layout):
  - coco:   datasets/COCO2014/saliency/<train2014|val2014>/<name>.png
  - fss:    datasets/FSS-1000/<class>/<id>_saliency.png
  - pascal: datasets/VOC2012/U2net_pascalAug/<name>.png

Example:
  python tools/gen_saliency.py --benchmark coco \
      --datapath ../datasets --u2net ../backbones/u2net.pth
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base.u2net import U2NET


def preprocess(pil_img, size=320):
    r""" Replicate U^2-Net inference preprocessing (RescaleT + ToTensorLab). """
    img = pil_img.convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    mx = arr.max()
    arr = arr / mx if mx > 1e-6 else arr
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def predict_saliency(net, pil_img, device):
    with torch.no_grad():
        x = preprocess(pil_img).to(device)
        d0 = net(x)
        sal = d0[0, 0].cpu().numpy()
    mn, mx = sal.min(), sal.max()
    sal = (sal - mn) / (mx - mn + 1e-8)
    sal = (sal * 255).astype(np.uint8)
    return Image.fromarray(sal).resize(pil_img.size, Image.BILINEAR)


def iter_image_targets(benchmark, datapath):
    r""" Yield (image_path, output_saliency_path) pairs per benchmark. """
    if benchmark == 'coco':
        base = os.path.join(datapath, 'COCO2014')
        for split in ['train2014', 'val2014']:
            img_dir = os.path.join(base, split)
            out_dir = os.path.join(base, 'saliency', split)
            for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
                name = os.path.splitext(os.path.basename(img_path))[0]
                yield img_path, os.path.join(out_dir, name + '.png')
    elif benchmark == 'fss':
        base = os.path.join(datapath, 'FSS-1000')
        for img_path in glob.glob(os.path.join(base, '*', '*.jpg')):
            cls = os.path.basename(os.path.dirname(img_path))
            name = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = os.path.join(base, cls)
            yield img_path, os.path.join(out_dir, name + '_saliency.png')
    elif benchmark == 'pascal':
        base = os.path.join(datapath, 'VOC2012')
        img_dir = os.path.join(base, 'JPEGImages')
        out_dir = os.path.join(base, 'U2net_pascalAug')
        for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
            name = os.path.splitext(os.path.basename(img_path))[0]
            yield img_path, os.path.join(out_dir, name + '.png')
    else:
        raise ValueError('Unknown benchmark: %s' % benchmark)


def main():
    parser = argparse.ArgumentParser(description='Offline U^2-Net saliency generation')
    parser.add_argument('--benchmark', type=str, required=True, choices=['coco', 'fss', 'pascal'])
    parser.add_argument('--datapath', type=str, default='../datasets')
    parser.add_argument('--u2net', type=str, default='../backbones/u2net.pth')
    parser.add_argument('--overwrite', action='store_true', help='regenerate even if output exists')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(args.u2net, map_location='cpu'))
    net.to(device).eval()

    targets = list(iter_image_targets(args.benchmark, args.datapath))
    print('Generating saliency for %d images (%s)' % (len(targets), args.benchmark))

    for i, (img_path, out_path) in enumerate(targets):
        if not args.overwrite and os.path.exists(out_path):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print('Skip %s (%s)' % (img_path, e))
            continue
        sal = predict_saliency(net, pil_img, device)
        sal.save(out_path)
        if (i + 1) % 200 == 0:
            print('  %d / %d' % (i + 1, len(targets)))

    print('Done.')


if __name__ == '__main__':
    main()
