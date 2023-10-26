import numpy as np
import cv2
from numpy import random
import math


def augmentation(hsv=False, fliplr=False, rotation=False,
                 rotation_max_angle=10.,
                 rotation_scale=[0.8, 1.],
                 ):
    transforms = []
    # Convert PIL images to tensors
    # transforms.append(HaveEnoughBoxes(min_area_ratio))
    if hsv:
        transforms.append(HSVTransform())

    if rotation:
        transforms.append(Rotation(rotation_max_angle, rotation_scale[0], rotation_scale[1]))

    if fliplr:
        transforms.append(Fliplr())
    return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, targets=None):
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


class HSVTransform:
    def __init__(self):
        pass

    def __call__(self, imgs, targets=None):
        # SV augmentation by 50%
        for i, img in enumerate(imgs):
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
            imgs[i] = img_hsv


class Rotation:
    def __init__(self, max_angle, min_scale, max_scale, translate=(0.1, 0.1), shear=(-2, 2)):
        self.max_angle = max_angle
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.translate = translate
        self.shear = shear

    def __call__(self, imgs, targets):
        angle = random.random * self.max_angle
        scale = random.random() * (self.max_scale - self.min_scale) + self.min_scale
        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * self.translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * self.translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)

        # x shear (deg)
        S[0, 1] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)

        # y shear (deg)
        S[1, 0] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)
        new_imgs, new_targets = [], []
        for img, tar in zip(imgs, targets):
            img, tar = random_affine(img, angle, T, scale, S, tar)
            new_imgs.append(img)
            new_targets.append(tar)

        return new_imgs, new_targets


class Fliplr:
    def __init__(self):
        pass

    def __call__(self, imgs, targets):
        for i, (img, labels) in enumerate(zip(imgs, targets)):
            # random left-right flip
            imgs[i] = np.fliplr(img)
            if len(labels) > 0:
                labels[:, 2] = 1 - labels[:, 2]


def random_affine(img, angle, T, scale, S, targets=None, borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(img.shape[1] / 2, img.shape[0] / 2), scale=scale)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]
            targets = targets[targets[:, 2] < width]
            targets = targets[targets[:, 4] > 0]
            targets = targets[targets[:, 3] < height]
            targets = targets[targets[:, 5] > 0]

        return imw, targets, M
    else:
        return imw


if __name__ == '__main__':
    pass

