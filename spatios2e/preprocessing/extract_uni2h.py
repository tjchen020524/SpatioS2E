"""Extract frozen UNI2-h vectors from one image and an explicit barcode/coordinate table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from spatios2e.preprocessing.extract_histology_embeddings import crop_patch, preprocess_patch
from spatios2e.preprocessing.prepare_decima_vectors import sha256


def load_model(checkpoint, device):
    import timm
    model = timm.create_model(
        'vit_giant_patch14_224', pretrained=False, img_size=224, patch_size=14, depth=24,
        num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    return model.requires_grad_(False).eval().to(device)


def white_crop(image, x, y, side):
    left, top = int(round(x - side / 2)), int(round(y - side / 2))
    canvas = Image.new('RGB', (side, side), (255, 255, 255))
    box = (max(left, 0), max(top, 0), min(left + side, image.width), min(top + side, image.height))
    if box[2] > box[0] and box[3] > box[1]:
        canvas.paste(image.crop(box), (box[0] - left, box[1] - top))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--coordinates', type=Path, required=True, help='TSV: barcode, pixel_x, pixel_y')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-mode', choices=['visium', 'her2st'], required=True)
    parser.add_argument('--spot-diameter-fullres', type=float, help='Required for Visium')
    parser.add_argument('--image-scale', type=float, default=1.0,
                        help='Multiply coordinate values and Visium full-resolution diameter by this value')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.coordinates, sep='\t', dtype={'barcode': str})
    if not {'barcode', 'pixel_x', 'pixel_y'} <= set(frame) or frame['barcode'].duplicated().any():
        raise ValueError('Coordinates require unique barcodes and pixel_x/pixel_y columns')
    points = frame[['pixel_x', 'pixel_y']].to_numpy(np.float32) * args.image_scale
    if len(points) < 1 or not np.isfinite(points).all() or args.batch_size < 1 or args.image_scale <= 0:
        raise ValueError('Invalid coordinates, scale or batch size')
    if args.crop_mode == 'visium':
        if args.spot_diameter_fullres is None or args.spot_diameter_fullres <= 0:
            parser.error('Visium requires a positive --spot-diameter-fullres')
        side = max(2, int(round(args.spot_diameter_fullres * args.image_scale)))
    else:
        if len(points) < 2:
            parser.error('HER2ST spacing requires at least two spots')
        distances = torch.cdist(torch.tensor(points), torch.tensor(points))
        distances.fill_diagonal_(float('inf'))
        # torch.median intentionally reproduces the historical lower-median rule.
        side = max(32, int(round(float(distances.min(dim=1).values.median()) * 0.5)))
    source = Image.open(args.image).convert('RGB')
    pixels = np.asarray(source)
    model = load_model(args.checkpoint, args.device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(points), args.batch_size):
            crops = [white_crop(source, x, y, side) if args.crop_mode == 'her2st'
                     else crop_patch(pixels, x, y, side) for x, y in points[start:start + args.batch_size]]
            batch = torch.stack([preprocess_patch(patch, 224, (0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)) for patch in crops])
            chunks.append(model(batch.to(args.device)).cpu().float().numpy())
    embeddings = np.concatenate(chunks)
    if embeddings.shape != (len(frame), 1536) or not np.isfinite(embeddings).all():
        raise ValueError('Unexpected UNI2-h output shape or nonfinite output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, barcodes=frame['barcode'].to_numpy(dtype=str), embeddings=embeddings)
    record = {'model': 'UNI2-h', 'crop_mode': args.crop_mode, 'crop_side_pixels': side,
              'image_scale': args.image_scale, 'n_spots': len(frame), 'output_sha256': sha256(args.output),
              'inputs': {key: sha256(getattr(args, key)) for key in ('image', 'coordinates', 'checkpoint')}}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
