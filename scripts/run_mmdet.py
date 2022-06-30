#!/usr/bin/env python3
import logging
import multiprocessing as mp
import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch.utils.data
from torch.utils.data import Dataset

import mmcv.runner
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate
import mmdet
import mmdet.datasets

import multiprocessing as mp

from src.data.image_dataset import MultiGlobDataset


logging.basicConfig(level=logging.INFO)


class DetectorRunner(mp.Process):
    def __init__(self, config: Path, checkpoint: Path,
                 dataset: Dataset, res_queue: mp.Queue,
                 batch_size=1, device='cpu'):
        super().__init__()
        logging.info('DetectionRunner.__init__ start')
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

        # Convert the model to GPU
        self.model.to(device)
        # Convert the model into evaluation mode
        self.model.eval()
        logging.info('DetectionRunner.__init__ done')

    def infer(self, ds: Dataset):
        transform = mmdet.datasets.pipelines.Compose(self.config.data.test.pipeline)
        ds.set_transform(transform)
        my_collate = partial(collate, samples_per_gpu=self.batch_size)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=1, collate_fn=my_collate)
        logging.debug(f'Starting DL loop')
        for batch in dl:
            logging.debug(f'batch of size {len(batch)}')
            batch['img_metas'] = [v.data[0] for v in batch['img_metas']]
            assert len(batch['img']) == 1
            batch['img'][0] = batch['img'][0].to(self.device)

            print(batch)
            with torch.no_grad():
                logging.debug(f'Infer {batch}')
                results = self.model(return_loss=False, **batch)
                logging.debug('Infer done')

                for i in range(len(results)):
                    fn = batch['img_metas'][0][i]['filename']
                    res = {'filename': fn, 'result': results[i]}
                    yield res

    def run(self):
        logging.debug('running')

        for result in self.infer(self.dataset):
            self.res_queue.put(result)

        self.res_queue.put(None)


def main():
    parser = ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    logging.info('main()')

    config = Path('mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py')
    checkpoint = Path('checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')

    device = 'cuda:0'
    device = 'cpu'

    res_q = mp.Queue()

    ds = MultiGlobDataset('/mnt/media/Photo/Походы/2021-05-Пра-Клепики-Деулино/Дима/*')

    inferrer = DetectorRunner(config, checkpoint, dataset=ds, res_queue=res_q, device=device, batch_size=1)
    #inferrer.start()

    with open(args.out, 'wb') as f:
        while True:
            # res = res_q.get()
            # if res is None:
            #     logging.info('Finishing')
            #     break

            #for i in range(len(res)):
            for res in inferrer.infer(ds):
                pickle.dump(res, f)
                logging.info(f'Processed {res["filename"]}')

    #inferrer.join()


if __name__ == '__main__':
    main()