import motmetrics as mm
import numpy as np
import cv2
from src.utils import Annotator


img = cv2.imread(r'D:\BanYanDeng\MOTDataset\MOT16\train\MOT16-11\img1\000001.jpg')
dets = [[544.06,249.9,112.14,338.41], [402.13,275.46,104.56,315.68],
        [929.93,127.67,210.12,632.36], [793,329,39,119],
        [1,1,341.97,1027.9], [864.82,276.69,90.896,274.69], [745,329,39,119,-0.48158]]

test = [[1032.20,274.07,103.49,345.16],[849.62,337.71,44.60,105.58], [1323.35,336.51,35.52,92.27],
        [741.23,331.43,44.99,114.93], [700.73,342.39,37.33,114.48], [1637.61,346.48,25.53,89.00],
        [884.01,173.96,207.22,578.57], [549.51,288.17,91.09,283.31], [1350.24,347.44,30.41,84.82],
        [401.88,289.99,101.34,294.22], [1856.65,334.18,33.99,96.09], [797.31,338.29,35.38,106.97],
        [1795.54,342.66,34.92,95.09], [1660.68,345.32,27.21,95.38], [1375.53,326.76,40.13,117.73]]
annt = Annotator(img)
for det in dets:
    det[2] += det[0]
    det[3] += det[1]
    annt.box_label(box=det, color=(0, 255, 0))

for det in test:
    det[2] += det[0]
    det[3] += det[1]
    annt.box_label(box=det, color=(0, 0, 255))

img = annt.result()
cv2.imshow('test', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
print(img.shape)



from src.utils import ious
gt_bbox = [[1, 1, 11, 21], [5, 5, 15, 25], [10, 10, 20, 30]]
det_bbox = [[2, 2, 11, 21], [10, 12, 21, 34]]
print(ious(gt_bbox, det_bbox))

gt = r'D:\BanYanDeng\MOTDataset\MOT16\train\MOT16-11\det\det.txt'
det = r'D:\BanYanDeng\MOT16\MOT16\train\MOT16-02\det\det.txt'
det = 'D:\BanYanDeng\MOTDataset\detects\MOT16-11\detect.txt'
lines = open(gt, 'r').readlines()
det_lines = open(det, 'r').readlines()

# gt = 'test_gt.txt'
# det = 'test_det.txt'
#评价指标
metrics = list(mm.metrics.motchallenge_metrics)

gt_file = mm.io.loadtxt(gt, 'mot15-2D')
det_file = mm.io.loadtxt(det, 'mot15-2D')

accs = []
names = []
for th in np.arange(0, 1, 0.1):
    accs.append(mm.utils.compare_to_groundtruth(gt_file, det_file, 'iou', distth=th))
    names.append(f'{th:.2f}')

mh = mm.metrics.create()
summary = mh.compute_many(accs, metrics, names=names)
eval = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)

print(eval.format('\t'))


