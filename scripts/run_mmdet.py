#!/usr/bin/env python3
import logging
import multiprocessing as mp
import os
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import sys
from typing import Callable, List

import cv2
import numpy as np
import torch.utils.data
from PIL import Image
from mmseg.models import build_segmentor
from torch.utils.data import Dataset

import mmcv.runner
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate
import mmdet
import mmdet.datasets

from tqdm import tqdm

from src.data.image_dataset import MultiGlobDataset
from src.palette import cocostuff_crude_palette

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    #filename='test.log',
                    #filemode='w',
                    stream=sys.stdout
                    )

#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)


class DetectorProcess(mp.Process):
    def __init__(self, config: Path, checkpoint: Path,
                 dataset: Dataset, res_queue: mp.Queue,
                 batch_size=1, device='cpu', cuda_device=None):
        super().__init__()
        logger.info('DetectionRunner.__init__ start')
        self.dataset = dataset
        self.res_queue = res_queue

        self.config = mmcv.Config.fromfile(str(config))
        # Set pretrained to be None since we do not need pretrained model here
        self.config.model.pretrained = None

        # Initialize the detector
        self.model = build_detector(self.config.model)
        checkpoint = load_checkpoint(self.model, str(checkpoint), map_location='cpu')

        # Set the classes of models for inference
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.meta = checkpoint['meta']

        # We need to set the model's cfg for inference
        self.model.cfg = config

        self.batch_size = batch_size
        self.device = device
        self.cuda_device = cuda_device

        # Convert the model into evaluation mode
        self.model.eval()
        logger.info('DetectionRunner.__init__ done')

    @staticmethod
    def my_collate(samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return []
        batch = collate(samples, samples_per_gpu=1)
        batch['img_metas'] = [v.data[0] for v in batch['img_metas']]
        assert len(batch['img']) == 1
        logger.debug(f'collating {batch}')
        return batch

    def infer(self, ds: Dataset):
        transform = mmdet.datasets.pipelines.Compose(self.config.data.test.pipeline)
        ds.set_transform(transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=1, collate_fn=self.my_collate)
        logger.debug(f'Starting DL loop')
        for batch in dl:
            logger.debug(f'batch of size {len(batch)}')

            if len(batch) == 0:
                continue

            batch['img'][0] = batch['img'][0].to(self.device)

            with torch.no_grad():
                logger.debug(f'Infer {batch}')
                results = self.model(return_loss=False, rescale=False, **batch)

                for i in range(len(results)):
                    meta = batch['img_metas'][0][i]
                    res = {'filename': meta['filename'],
                           'infer_size': (meta['img_shape'][1], meta['img_shape'][0]),
                           'result': results[i]}

                    logger.debug(f'Infer result {res}')
                    yield res

    def run(self):
        logger.info(f'running on device {self.device} {self.cuda_device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_device

        self.model.to(self.device)

        for result in self.infer(self.dataset):
            self.res_queue.put(result)

        # enough poison for all
        for k in range(100):
            self.res_queue.put(None)


class SegmentorProcess(DetectorProcess):
    def __init__(self, config: Path, checkpoint: Path,
                 dataset: Dataset, res_queue: mp.Queue,
                 batch_size=1, device='cpu', cuda_device=None):
        mp.Process.__init__(self)
        logger.info('SegmentorProcess.__init__ start')
        self.dataset = dataset
        self.res_queue = res_queue

        self.config = mmcv.Config.fromfile(config)
        self.config.model.pretrained = None
        self.config.model.train_cfg = None

        model = build_segmentor(self.config.model)
        checkpoint = load_checkpoint(model, str(checkpoint), map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
        model.eval()
        self.model = model

        self.batch_size = batch_size
        self.device = device
        self.cuda_device = cuda_device


class BaseConsumer(mp.Process):
    def __init__(self, result_q: mp.Queue, out: Path, model: torch.nn.Module):
        super().__init__()
        logger.info('BaseConsumer.__init__')
        self.result_q = result_q
        self.out = out
        self.model = model


class DumpingConsumer(BaseConsumer):
    def run(self):
        with open(self.out, 'ab') as f:
            pbar = tqdm(desc='Processing')
            while True:
                res = self.result_q.get()
                if res is None:
                    logger.info('Finishing')
                    break

                pickle.dump(res, f)
                pbar.update()
                pbar.write(f'Processed {res["filename"]}')


class RenderingConsumer(BaseConsumer):
    def __init__(self, result_q: mp.Queue, out: Path, model: torch.nn.Module, skip_prefix: str, flat=False):
        super().__init__(result_q, out, model=model)
        self.skip_prefix = skip_prefix
        self.flat = flat

    def render_image(self, scaled_img_arr: np.array, infer_result):
        raise NotImplementedError()

    def run(self):
        pbar = tqdm(desc='Rendering results')
        while True:
            res = self.result_q.get()
            if res is None:
                logger.info('Finishing')
                break

            fn = res['filename']
            assert fn.startswith(self.skip_prefix), f"file {fn} and prefix {self.skip_prefix} don't match each other"

            img_arr = cv2.imread(fn)
            scaled_img_arr = cv2.resize(img_arr, res['infer_size'], interpolation=cv2.INTER_LINEAR)

            seg_img_arr = self.render_image(scaled_img_arr, res['result'])

            if self.flat:
                out_fn = self.out / Path(fn).name
            else:
                out_fn = self.out / fn[len(self.skip_prefix):]
            if out_fn.exists():
                logger.warning(f'Overwriting existing {out_fn}')
            else:
                logger.info(f'saving {out_fn}')
            out_fn.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(out_fn), seg_img_arr)
            pbar.update()


class SemanticSegRenderingConsumer(RenderingConsumer):
    def __init__(self, opacity: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.opacity = opacity

    def render_image(self, scaled_img_arr, infer_result):
        seg_img_arr = self.model.show_result(scaled_img_arr, [infer_result],
                                             opacity=self.opacity, palette=cocostuff_crude_palette())
        return seg_img_arr


class InstanceSegRenderingConsumer(RenderingConsumer):
    def __init__(self, score_thr: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.score_thr = score_thr

    def render_image(self, scaled_img_arr, infer_result):
        seg_img_arr = self.model.show_result(scaled_img_arr, infer_result, score_thr=self.score_thr, font_size=20)
        return seg_img_arr


class DumpingPipeline:
    def __init__(self, input_patterns: List[str], processor_cls: Callable, consumer_cls: Callable,
                 config: Path, checkpoint: Path, skiplist=None, n_inferrers: int = 1, n_consumers: int = 1, cuda: bool = True):
        self.result_q = mp.Queue()

        ds = MultiGlobDataset(input_patterns, skiplist=skiplist)

        if cuda:
            device = 'cuda'
            visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count()))))
        else:
            device = 'cpu'
            visible_devices = ['0']

        procs = []
        for k in range(n_inferrers):
            assert n_inferrers == 1, 'Multiple inferrers are not supported, need to split dataset'
            d = visible_devices[k % len(visible_devices)]
            inferrer = processor_cls(config, checkpoint, dataset=ds, res_queue=self.result_q, batch_size=1,
                                     device=device, cuda_device=d)
            procs.append(inferrer)
        self.inferrers = procs

        procs = []
        for k in range(n_consumers):
            cons = consumer_cls(result_q=self.result_q, model=self.inferrers[0].model)
            procs.append(cons)

        self.consumers = procs

    @staticmethod
    def read_already_processed(path: Path):
        path = Path(path)
        already_processed = []
        if path.exists():
            with open(path, 'rb') as f:
                while True:
                    try:
                        row = pickle.load(f)
                        already_processed.append(row['filename'])
                    except EOFError:
                        logger.info(f'Found {len(already_processed)} already processed images')
                        break

        return already_processed

    def process(self):
        for p in self.inferrers:
            p.start()
        for p in self.consumers:
            p.start()

        logger.info('joining inferrers')
        for p in self.inferrers:
            p.join()
        logger.info('joining consumers')
        for p in self.consumers:
            p.join()


def main():
    mp.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument('--input_patterns', nargs='+', required=True, help='Globbing patterns for locating input images')
    parser.add_argument('--out', type=str, required=True, help='Output directiry')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_inferrers', type=int, default=1, help='Number of inferring workers')
    parser.add_argument('--n_consumers', type=int, default=1, help='Number of result saving workers')
    parser.add_argument('--mode', choices='dump render'.split(), required=True, help='Either dump the results or render demo images')
    parser.add_argument('--task', choices="instance_seg semantic_seg".split(), required=True, help='Either detect object instances or just classify all the image pixels')
    parser.add_argument('--score_thr', type=float, default=0.5, help='Threshold for instance segmentation')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity for rendering segmentation results')
    parser.add_argument('--flat', action='store_true', help='If set, render all the results in a single directory')
    args = parser.parse_args()

    logger.info('main()')

    if args.task == 'instance_seg':
        config = Path('mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py')
        checkpoint = Path('checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
        worker = DetectorProcess
    elif args.task == 'semantic_seg':
        config = 'mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_4x4_320k_coco-stuff164k.py'
        checkpoint = 'checkpoints/pspnet_r50-d8_512x512_4x4_320k_coco-stuff164k_20210707_152004-be9610cc.pth'
        worker = SegmentorProcess
    else:
        raise ValueError('Unknown task', args.task)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    if args.mode == 'dump':
        consumer = partial(DumpingConsumer, out=out)
        skiplist = DumpingPipeline.read_already_processed(args.out)
    elif args.mode == 'render':
        if args.task == 'instance_seg':
            consumer = partial(InstanceSegRenderingConsumer, skip_prefix='/mnt/media/Photo/Походы/', out=out,
                               score_thr=args.score_thr, flat=args.flat)
        elif args.task == 'semantic_seg':
            consumer = partial(SemanticSegRenderingConsumer, skip_prefix='/mnt/media/Photo/Походы/', out=out,
                               opacity=args.opacity, flat=args.flat)
        else:
            raise ValueError('Unknown task', args.task)
        skiplist = []
    else:
        raise ValueError('Unknown mode', args.mode)

    proc = DumpingPipeline(input_patterns=args.input_patterns, processor_cls=worker, consumer_cls=consumer,
                           n_inferrers=args.n_inferrers, n_consumers=args.n_consumers,
                           config=config, checkpoint=checkpoint, cuda=args.cuda, skiplist=skiplist)
    proc.process()


if __name__ == '__main__':
    main()
