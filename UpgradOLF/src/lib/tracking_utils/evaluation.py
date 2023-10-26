import os
import sys
import numpy as np
import copy
import motmetrics as mm
import shutil
mm.lap.default_solver = 'lap'
import subprocess
from src.lib.tracking_utils.io import read_results, unzip_objs


class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events, trk_tlwhs, trk_ids

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        # frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        frames = sorted(list(set(result_frame_dict.keys())))
        keep_trks = {}
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            _, keep_tlwhs, keep_ids = self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
            keep_trks[str(frame_id)] = {"tlwhs": keep_tlwhs,
                                        "ids": keep_ids
                                        }

        return self.acc, keep_trks

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


class TrackEval():
    def __init__(self, track_eval_dir=None, challenge_name=None, exp_name=None):
        # HOTA challenge
        if track_eval_dir is not None:
            if os.path.isdir(os.path.join(track_eval_dir, "trackeval")):
                if challenge_name is None:
                    raise ValueError("track_eval_dir not None but challenge_name is None")

                self.suffix = str(challenge_name).split("-")[-1]
                if self.suffix not in ["val", "train", "test"]:
                    raise ValueError("challenge_name should be end with [-train, -val, -test]")
                self.mot_challenge = os.path.join(track_eval_dir, "data/gt/mot_challenge", challenge_name)
                if not os.path.exists(self.mot_challenge):
                    os.makedirs(self.mot_challenge)

                self.tracker = os.path.join(track_eval_dir, "data/trackers/mot_challenge", challenge_name)
                if not os.path.exists(self.tracker):
                    os.makedirs(self.tracker)

                self.exp_name_path = os.path.join(self.tracker, exp_name)
                if not os.path.exists(self.exp_name_path):
                    os.mkdir(self.exp_name_path)

                self.traker_data = os.path.join(self.exp_name_path, "data")
                if not os.path.exists(self.traker_data):
                    os.mkdir(self.traker_data)

            else:
                raise ValueError("not a valid track_eval_dir")

        self.track_eval_dir = track_eval_dir  # track evaluator for support HOTA evaluation
        self.challenge_name = challenge_name
        self.exp_name = exp_name

    def eval_seq(self, seq_name, trks, gt_root):
        # write gt file
        dir_list = os.listdir(gt_root)
        if "seqinfo.ini" not in dir_list:
            msg = f"not found seqinfo.ini in {gt_root}"
            raise ValueError(msg)
        src_gt_file = open(os.path.join(gt_root, "gt/gt.txt"), 'r')
        src_lines = src_gt_file.readlines()
        gt_lines = []
        for line in src_lines:
            line = line.replace("\n", "").split(",")
            label = int(float(line[7]))
            mark = int(float(line[6]))
            if label == 1 and not mark == 0:
                if len(line) > 10:
                    raise ValueError("gt file not valid, line length > 10")
                gt_lines.append(f"{int(float(line[0]))},{int(float(line[1]))}," + ",".join(line[2:])
                                + ",-1" * (10 - len(line)) + "\n")

        make_gt_root = os.path.join(self.mot_challenge, seq_name+f"_{self.suffix}", "gt")
        if not os.path.exists(make_gt_root):
            os.makedirs(make_gt_root)

        gt_file = open(os.path.join(make_gt_root, "gt.txt"), 'w')
        gt_file.writelines(gt_lines)
        gt_file.close()

        # seqinfo.ini
        shutil.copy(os.path.join(gt_root, "seqinfo.ini"), os.path.dirname(make_gt_root))

        # write dets file
        lines = []
        for frame_id, trk in trks.items():
            for tlwh, id in zip(trk["tlwhs"], trk["ids"]):
                line = "{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},-1,-1,-1,-1\n".format(int(frame_id), int(int(id)),
                                                                                float(tlwh[0]), float(tlwh[1]),
                                                                                float(tlwh[2]), float(tlwh[3]))

                lines.append(line)

        det_file = open(os.path.join(self.traker_data, seq_name + f"_{self.suffix}.txt"), "w")
        det_file.writelines(lines)
        det_file.close()

    def sys_do_eval(self):
        benchmark = self.challenge_name.split("-")[0]
        cmd = (f"python scripts/run_mot_challenge.py --BENCHMARK {benchmark} "
               f"--SPLIT_TO_EVAL val --TRACKERS_TO_EVAL {self.exp_name}"
               " --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1")
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=self.track_eval_dir)
        print(process.stderr)
        # print(process.stdout)

    def print_keys_value(self, keys):
        text_path = os.path.join(self.exp_name_path, "pedestrian_summary.txt")
        cmd = (f"python data_format_tools/print_eval_.py --text_path {text_path} "
               f"--print_keys {','.join(keys)}")
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=self.track_eval_dir)
        print(process.stdout)

