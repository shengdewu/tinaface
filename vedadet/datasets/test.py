import os
import numpy as np
from torch.utils.data import Dataset

import os.path as osp
import tempfile

import vedacore.fileio as fileio

from vedacore.misc import registry
from .pipelines import Compose

import cv2

import json


@registry.register_module('dataset')
class TestDataset(Dataset):

    CLASSES = ['face']

    def __init__(self,
                 pipeline,
                 img_prefix='',
                 max_num=-1,
                 test_mode=True):
        self.img_prefix = img_prefix

        self.data_infos = list()
        for name in os.listdir(self.img_prefix):
            self.data_infos.append({'filename': name})

        if max_num > 0:
            self.data_infos = self.data_infos[:max_num]

        self.pipeline = Compose(pipeline)
        return

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['proposal_file'] = []
        results['bbox_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        data = self.prepare_test_img(idx)
        # print(data['img'][0].shape)
        return data

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.data_infos[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = 'face'
                    json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions and
        they have different data types. This method will automatically
        recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            fileio.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            fileio.dump(json_results[0], result_files['bbox'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            fileio.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 out_put=''):

        json_path = os.path.join(out_put, 'json')
        os.makedirs(json_path, exist_ok=True)

        for idx in range(len(self)):
            img_name = self.data_infos[idx]
            img = cv2.imread(os.path.join(self.img_prefix, img_name['filename']))

            result = results[idx]

            with open(os.path.join(json_path, '{}.json'.format(img_name)), mode='w') as w:
                json.dump(result, w)

            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i].tolist()[:4]
                    score = float(bboxes[i][4])
                    if score <= 0.5:
                        continue
                    poly_box = [int(p + 0.5) for p in bbox]
                    cv2.rectangle(img, (poly_box[0], poly_box[1]), (poly_box[2], poly_box[3]), color=[0,255,0], thickness=8)
                    cv2.putText(img, '{}'.format(round(score,3)), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 4, [255,0,0], thickness=4)
            cv2.imwrite('{}/{}'.format(out_put, img_name['filename']), img)
        return