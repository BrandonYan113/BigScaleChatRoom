import configparser
import csv
import itertools
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from utils.annotation import Annotator


class MOTDetection(Dataset):
    """
        Data class for detection
        Loads all images in all sequences at once
        To be used for training
    """

    def __init__(
            self,
            root="../datasets/",
            dataset="MOT16",
            vis_threshold=0.1,
    ):
        predefined_datasets = ["MOT16", "MOT17", "MOT20"]
        assert dataset in predefined_datasets, \
            f"Provided dataset name '{dataset}' is not in predefined datasets: {predefined_datasets}"

        self.root = os.path.join(root, dataset, "train")
        self._vis_threshold = vis_threshold
        self._classes = ("__background__", "pedestrian")
        self._global_id_counter = 0
        self._local_to_global_dict = {}
        self._global_to_local_dict = {}
        self._img_paths = []
        self._aspect_ratios = []
        self.sequence_len_set = []

        for f in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, f)
            config_file = os.path.join(path, "seqinfo.ini")

            assert os.path.exists(config_file), f"Path does not exist: {config_file}"

            config = configparser.ConfigParser()  # configparser用来读配置文件
            config.read(config_file)
            seq_len = int(config["Sequence"]["seqLength"])
            im_width = int(config["Sequence"]["imWidth"])
            im_height = int(config["Sequence"]["imHeight"])
            im_ext = config["Sequence"]["imExt"]
            im_dir = config["Sequence"]["imDir"]

            self.sequence_len_set.append(seq_len)

            _imDir = os.path.join(path, im_dir)
            aspect_ratio = im_width / im_height

            # Collect global gt_id
            self.process_ids(path)

            for i in range(1, seq_len + 1):
                img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
                assert os.path.exists(img_path), \
                    "Path does not exist: {img_path}"
                self._img_paths.append(img_path)
                self._aspect_ratios.append(aspect_ratio)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def num_ids(self):
        return self._global_id_counter

    def _get_annotation(self, index):
        """
            Obtain annotation from gt file
        """
        img_path = self._img_paths[index]
        file_index = int(os.path.basename(img_path).split(".")[0])

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "gt", "gt.txt")
        seq_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

        assert os.path.exists(gt_file), f"GT file does not exist: {gt_file}"

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                visibility = float(row[8])
                local_id = f"{seq_name}-{int(row[1])}"
                if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and \
                        visibility > self._vis_threshold:
                    bb = {}
                    bb["gt_id"] = self._local_to_global_dict[local_id]
                    bb["bb_left"] = int(row[2])
                    bb["bb_top"] = int(row[3])
                    bb["bb_width"] = int(row[4])
                    bb["bb_height"] = int(row[5])
                    bb["visibility"] = visibility

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)
        gt_ids = torch.zeros((num_objs), dtype=torch.int64)

        for i, bb in enumerate(bounding_boxes):
            x1 = bb["bb_left"]  # GS
            y1 = bb["bb_top"]
            x2 = x1 + bb["bb_width"]
            y2 = y1 + bb["bb_height"]
            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb["visibility"]
            gt_ids[i] = bb["gt_id"]

        return {"boxes": boxes,
                "labels": torch.ones((num_objs,), dtype=torch.int64),
                "image_id": torch.tensor([index]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
                "visibilities": visibilities,
                "frame_id": torch.tensor([file_index]),
                "gt_ids": gt_ids}

    def process_ids(self, path):
        """
            Global id is 0-based, indexed across all sequences
            All ids are considered, regardless of used or not
        """
        seq_name = os.path.basename(path)
        if seq_name not in self._global_to_local_dict.keys():
            self._global_to_local_dict[seq_name] = {}
        gt_file = os.path.join(path, "gt", "gt.txt")
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                local_id = f"{seq_name}-{int(row[1])}"
                if int(row[6]) == 1 and int(row[7]) == 1:
                    if local_id not in self._local_to_global_dict.keys():
                        self._local_to_global_dict[local_id] = self._global_id_counter
                        self._global_to_local_dict[seq_name][self._global_id_counter] = int(row[1])
                        self._global_id_counter += 1

    def __getitem__(self, index):
        # Load image
        img_path = self._img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get annotation
        target = self._get_annotation(index)
        # Apply augmentation transforms

        return img, target

    def __len__(self):
        return len(self._img_paths)


class SplitBatch(Dataset):
    def __init__(self,
                 root=r"D:\\BanYanDeng\MOTDataset",
                 dataset="MOT16",
                 split_len=5,
                 batch_size=4,
                 split_batch_size=4,
                 level_num=4,
                 box_num=3,
                 min_area=16. * 16.,
                 pre_transforms=None,
                 transform=None,
                 scal_factor=16,
                 box_label_func='default',
                 vis_threshold=0.1,
                 label_min_overlap_ratio=0.5
                 ):
        super(SplitBatch, self).__init__()

        self.mot_dets = MOTDetection(root, dataset, vis_threshold)
        self.pre_transform = pre_transforms
        self.transform = transform
        self.split_len = split_len
        self.batch_size = batch_size
        self.split_batch_size = split_batch_size
        self.scale_factor = scal_factor
        self.box_label_func = box_label_func
        self.level_num = level_num
        self.box_num = box_num
        self.min_area = min_area

        self.Annotator = Annotator(scal_factor, level_num, min_area, split_len,
                                   label_min_overlap_ratio=label_min_overlap_ratio)

        self.len_dict = {}
        start_index = 0
        sequence_num = len(self.mot_dets.sequence_len_set)

        total_split = 0
        for i, sl in enumerate(self.mot_dets.sequence_len_set):
            self.len_dict[str(i)] = {}
            self.len_dict[str(i)]['less_len'] = sl
            self.len_dict[str(i)]['start_index'] = start_index
            start_index += sl
            total_split += int(sl / split_len)

        self.yiled_index = []

        total_frames = len(self.mot_dets)
        for i in range(split_batch_size):
            self.yiled_index.append(IndexStack(split_len))

        while True:
            # less split can't be grouped as a video batch
            if total_split < split_batch_size:
                break

            for stack in self.yiled_index:
                available_sequence = [i for i in range(sequence_num) if
                                      self.len_dict[str(i)]['less_len'] >= split_len]
                avail_num = len(available_sequence)
                if avail_num > 0:
                    r = available_sequence[np.random.randint(avail_num)]
                    stack.push(self.len_dict[str(r)]['start_index'])

                    self.len_dict[str(r)]['start_index'] += split_len
                    self.len_dict[str(r)]['less_len'] -= split_len
                    total_frames -= split_len

            total_split -= split_batch_size

        self.single_frames_set = []  # record frame index in mot_dets
        self.record = 0
        while total_frames > 0:
            for i in range(sequence_num):
                if self.len_dict[str(i)]['less_len'] > 0:
                    self.single_frames_set.append(self.len_dict[str(i)]['start_index'])
                    self.len_dict[str(i)]['start_index'] += 1
                    self.len_dict[str(i)]['less_len'] -= 1
                    total_frames -= 1

        ll = len(self.single_frames_set)
        self.split_batch_num = len(self.yiled_index[0].start_indexes)
        self.less_split_num = total_split
        self.image_batch_num = int(np.ceil(ll / float(batch_size)))
        self.len = self.split_batch_num + self.less_split_num + self.image_batch_num

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        imgs, targets = [], []
        if index < self.split_batch_num + self.less_split_num:
            output_type = 'video'
            for i in range(self.split_len):  # split
                for index_stack in self.yiled_index:  # batch
                    ind = index_stack.pop()
                    if ind is not None:
                        img, t = self.mot_dets.__getitem__(ind)
                        imgs.append(img)
                        targets.append(t)
        else:
            output_type = 'image'
            for i in range(len(self.single_frames_set)):
                ind = self.single_frames_set.pop(0)
                img, target = self.mot_dets.__getitem__(ind)
                imgs.append(img)
                targets.append(target)
                if i == self.batch_size - 1:
                    break

        spimgs, sptargets = self.pre_transform(imgs, targets)
        spimgs, sptargets = self.transform(spimgs, sptargets)

        bboxes = [target['boxes'] for target in sptargets]
        visibilities = [target['visibilities'] for target in sptargets]
        ids = [target['gt_ids'] for target in sptargets]
        # frame_id = [target['image_id'].view(1, -1) for target in sptargets]
        spimgs = spimgs.tensors
        h, w = spimgs.shape[-2:]
        batch_shape = (int(h / self.scale_factor), int(w / self.scale_factor))

        locations1, locations2 = None, None
        if output_type == 'video':
            locations1, locations2 = self.Annotator.last_frames(bboxes, batch_shape, ids, visibilities)

        boxes, heat_maps = self.Annotator.current_frame(bboxes,
                                                        batch_shape,
                                                        visibilities,
                                                        box_label_func=self.box_label_func,
                                                        )
        if output_type == "video":
            return spimgs, boxes, heat_maps, output_type, locations1, locations2
        elif output_type == 'image':
            return spimgs, boxes, heat_maps, output_type


class IndexStack:
    def __init__(self, max_continuous_len):
        self.start_indexes = []
        self.pop_len = 0
        self.mcl = max_continuous_len

    def pop(self):
        if not self.is_empty() or self.pop_len < self.mcl:
            if self.pop_len == self.mcl or self.pop_len == 0:
                self.pop_index = self.start_indexes[0]
                del self.start_indexes[0]
                self.pop_len = 1
            else:
                self.pop_index += 1
                self.pop_len += 1

            return self.pop_index
        else:
            return None

    def push(self, start_index):
        if self.is_empty():
            self.bottom = -1

        self.start_indexes.append(start_index)

    def is_empty(self):
        return len(self.start_indexes) == 0


class MOTTracking(torch.utils.data.Dataset):
    """
        Data class for tracking
        Loads one sequence at a time
        To be used for tracking
    """

    def __init__(
            self,
            root="../datasets/",
            dataset="MOT17",
            which_set="train",
            sequence="02",
            public_detection=None,
            vis_threshold=0.1,
            transform=None
    ):
        # Check dataset
        predefined_datasets = ["MOT16", "MOT17", "MOT20"]
        assert dataset in predefined_datasets, \
            f"Provided dataset name '{dataset}' is not in predefined datasets: {predefined_datasets}"
        # Different public detections for MOT17
        if dataset == "MOT17":
            assert public_detection in ["DPM", "FRCNN", "SDP"], "Incorrect public detection provided"
            public_detection = f"-{public_detection}"
        # No public detection names for MOT16 and MOT20
        else:
            assert public_detection == None, f"No public detection should be provided for {dataset}"
            public_detection = ""
        # Check train/test
        assert which_set in ["train", "test"], "Invalid choice between 'train' and 'test'"
        # Check sequence, convert to two-digits string format
        assert sequence.isdigit(), "Non-digit sequence provided"
        sequence = f"{int(sequence):02d}"
        dict_sequences = {
            "MOT16": {
                "train": ["02", "04", "05", "09", "10", "11", "13"],
                "test": ["01", "03", "06", "07", "08", "12", "14"],
            },
            "MOT17": {
                "train": ["02", "04", "05", "09", "10", "11", "13"],
                "test": ["01", "03", "06", "07", "08", "12", "14"],
            },
            "MOT20": {
                "train": ["01", "02", "03", "05"],
                "test": ["04", "06", "07", "08"],
            }
        }
        assert sequence in dict_sequences[dataset][which_set], \
            f"Sequence for {dataset}/{which_set} must be in [{dict_sequences[dataset][which_set]}]"

        self._img_paths = []
        self._vis_threshold = vis_threshold

        # Load images
        self.path = os.path.join(root, dataset, which_set, f"{dataset}-{sequence}{public_detection}")
        self.transform = transform
        # print('hree %s'%root, self.path, os.path.exists(root))
        config_file = os.path.join(self.path, "seqinfo.ini")

        assert os.path.exists(config_file), f"Path does not exist: {config_file}"

        config = configparser.ConfigParser()
        config.read(config_file)
        seq_len = int(config["Sequence"]["seqLength"])
        im_ext = config["Sequence"]["imExt"]
        im_dir = config["Sequence"]["imDir"]

        _imDir = os.path.join(self.path, im_dir)

        for i in range(1, seq_len + 1):
            img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
            assert os.path.exists(img_path), \
                "Path does not exist: {img_path}"
            self._img_paths.append(img_path)

    def _get_annotation(self, index):
        """
            Obtain annotation for detections (train/test) and ground truths (train only)
        """
        img_path = self._img_paths[index]
        file_index = int(os.path.basename(img_path).split(".")[0])

        det_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "det", "det.txt")
        assert os.path.exists(det_file), \
            f"Det file does not exist: {det_file}"
        det_boxes, _, det_scores, _ = read_mot_file(det_file, file_index, self._vis_threshold, is_gt=False)

        # No GT for test set
        if "test" in self.path:
            return det_boxes, None, None, None, None

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), "gt", "gt.txt")
        assert os.path.exists(gt_file), \
            f"GT file does not exist: {gt_file}"
        gt_boxes, gt_ids, _, gt_visibilities = read_mot_file(gt_file, file_index, self._vis_threshold, is_gt=True)

        return det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities

    def __getitem__(self, index):
        # Load image
        img_path = self._img_paths[index]
        img0 = cv2.imread(img_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = to_tensor(img0)[None]
        if self.transform is not None:
            img, _ = self.transform(img)

        # Get annotation
        det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities = self._get_annotation(index)

        return img, det_boxes, det_scores, gt_boxes, gt_ids, gt_visibilities, img_path, img0

    def __len__(self):
        return len(self._img_paths)


def read_mot_file(file, file_index, vis_threshold=0.1, is_gt=False):
    """
        Read data from mot files, gt or det or tracking result
    """
    bounding_boxes = []
    with open(file, "r") as inf:
        reader = csv.reader(inf, delimiter=",")
        for row in reader:
            visibility = float(row[8]) if is_gt else -1.0
            if int(row[0]) == file_index and \
                    ((is_gt and (int(row[6]) == 1 and int(row[7]) == 1 and visibility > vis_threshold)) or
                     not is_gt):  # Only requires class=pedestrian and confidence=1 for gt
                bb = {}
                bb["gt_id"] = int(row[1])
                bb["bb_left"] = float(row[2])
                bb["bb_top"] = float(row[3])
                bb["bb_width"] = float(row[4])
                bb["bb_height"] = float(row[5])
                bb["bb_score"] = float(row[6]) if not is_gt else 1
                bb["visibility"] = visibility
                bounding_boxes.append(bb)

    num_objs = len(bounding_boxes)
    boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
    scores = torch.zeros((num_objs), dtype=torch.float32)
    visibilities = torch.zeros((num_objs), dtype=torch.float32)
    ids = torch.zeros((num_objs), dtype=torch.int64)
    for i, bb in enumerate(bounding_boxes):
        x1 = bb["bb_left"]  # GS
        y1 = bb["bb_top"]
        x2 = x1 + bb["bb_width"]
        y2 = y1 + bb["bb_height"]
        boxes[i, 0] = x1
        boxes[i, 1] = y1
        boxes[i, 2] = x2
        boxes[i, 3] = y2
        scores[i] = bb["bb_score"]
        visibilities[i] = bb["visibility"]
        ids[i] = bb["gt_id"]

    return boxes, ids, scores, visibilities


def collate_fn(batch):
    """
        Function for dataloader
    """

    return tuple(zip(*batch))


def get_seq_names(dataset, which_set, public_detection, sequence):
    """
        Get name of all required sequences
    """
    # Process inputs
    if public_detection == "all":
        if dataset == "MOT17":
            public_detection_list = ["DPM", "FRCNN", "SDP"]
        else:
            public_detection_list = ["None"]
    else:
        public_detection_list = [public_detection]

    if sequence == "all":
        if dataset == "MOT20":
            if which_set == "train":
                sequence_list = ["01", "02", "03", "05"]
            else:
                sequence_list = ["04", "06", "07", "08"]
        else:
            if which_set == "train":
                sequence_list = ["02", "04", "05", "09", "10", "11", "13"]
            else:
                sequence_list = ["01", "03", "06", "07", "08", "12", "14"]
    else:
        sequence_list = [sequence]
    # Iterate through all sequences
    full_names = []
    seqs = []
    pds = []  # public detections for each sequence
    for pd, seq in list(itertools.product(public_detection_list, sequence_list)):
        seqs.append(seq)
        pd_suffix = f"-{pd}" if dataset == "MOT17" else ""
        pds.append(pd)
        curr_seq = f"{dataset}-{seq}{pd_suffix}"
        full_names.append(curr_seq)
    return full_names, seqs, pds
