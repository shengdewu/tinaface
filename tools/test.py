import argparse

import torch
import sys
import os

parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine
import random
from torch_script.model import ScriptMode


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--save', help='the result of image')
    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint, script_path=''):

    if len(script_path) > 0:
        engine = ScriptMode(cfg.val_engine.model, checkpoint, script_path)
    else:
        engine = build_engine(cfg.val_engine)
        load_weights(engine.model, checkpoint, map_location='cpu')

        device = torch.cuda.current_device()
        engine = MMDataParallel(
            engine.to(device), device_ids=[torch.cuda.current_device()])

        engine.eval()

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def test(engine, data_loader):
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        results.append(result)
        batch_size = len(data['img_metas'].data) if not isinstance(data['img_metas'], list) else len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint, args.save)

    results = test(engine, data_loader)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)

    os.makedirs(args.save, exist_ok=True)
    data_loader.dataset.evaluate(results, args.save)


if __name__ == '__main__':
    main()
