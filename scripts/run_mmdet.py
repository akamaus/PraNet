#!/usr/bin/env python3
import logging
import multiprocessing as mp
import os
import os.path as osp
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import sys

import torch.utils.data
from torch.utils.data import Dataset

import mmcv.runner
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate
import mmdet
import mmdet.datasets

import multiprocessing as mp

from tqdm import tqdm

from src.data.image_dataset import MultiGlobDataset


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


class DetectorRunner(mp.Process):
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
        checkpoint = load_checkpoint(self.model, str(checkpoint), map_location=device)

        # Set the classes of models for inference
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.meta = checkpoint['meta']

        # We need to set the model's cfg for inference
        self.model.cfg = config

        self.batch_size = batch_size
        self.device = device
        self.cuda_device = cuda_device

        # Convert the model to GPU
        self.model.to(device)
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
                results = self.model(return_loss=False, **batch)
                logger.debug('Infer done')

                for i in range(len(results)):
                    fn = batch['img_metas'][0][i]['filename']
                    res = {'filename': fn, 'result': results[i]}
                    yield res

    def run(self):
        logger.info(f'running on device {self.device} {self.cuda_device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_device

        for result in self.infer(self.dataset):
            self.res_queue.put(result)

        self.res_queue.put(None)


def main():
    mp.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_inferrers', type=int, default=1)
    args = parser.parse_args()

    logger.info('main()')

    config = Path('mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py')
    checkpoint = Path('checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')

    if args.cuda:
        device = 'cuda'
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count()))))
    else:
        device = 'cpu'
        visible_devices = ['0']

    already_processed = []
    if osp.exists(args.out):
        with open(args.out, 'rb') as f:
            while True:
                try:
                    row = pickle.load(f)
                    already_processed.append(row['filename'])
                except EOFError:
                    logger.info(f'Found {len(already_processed)} already processed images')
                    break

    res_q = mp.Queue()

    ds = MultiGlobDataset(['/mnt/media/Photo/Походы/2021-05-Пра-Клепики-Деулино/*/*',
                          '/mnt/media/Photo/Походы/2022-06-Пра-Деулино-ББор/*/*'], skiplist=already_processed)

    procs = []
    for k in range(args.n_inferrers):
        assert args.n_inferrers == 1, 'Multiple inferrers are not supported, need to split dataset'
        d = visible_devices[k % len(visible_devices)]
        inferrer = DetectorRunner(config, checkpoint, dataset=ds, res_queue=res_q, batch_size=1,
                                  device=device, cuda_device=d)
        inferrer.start()
        procs.append(inferrer)

    with open(args.out, 'ab') as f:
        pbar = tqdm(desc='Processing')
        while True:
            res = res_q.get()
            if res is None:
                logger.info('Finishing')
                break

            pickle.dump(res, f)
            pbar.update()
            pbar.write(f'Processed {res["filename"]}')

    logger.info('joining inferrers')
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()