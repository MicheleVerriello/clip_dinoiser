import os
import mmcv
from mmseg.datasets import CustomDataset

class PascalVOCDataset(CustomDataset):
    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, limit=None, **kwargs):
        self.limit = limit
        super(PascalVOCDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert os.path.exists(self.img_dir) and self.split is not None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(split)
            if self.limit:
                lines = lines[:self.limit]
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                seg_map = img_name + seg_map_suffix
                img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        return img_infos

    def __getitem__(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        img = results['img'].data
        gt_semantic_seg = results['gt_semantic_seg'].data
        return img, gt_semantic_seg