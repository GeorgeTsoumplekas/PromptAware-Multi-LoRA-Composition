import math

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from PIL import Image


def _gaussian(
    size=3,
    sigma=0.25,
    amplitude=1,
    normalize=False,
    width=None,
    height=None,
    sigma_horz=None,
    sigma_vert=None,
    mean_horz=0.5,
    mean_vert=0.5,
):
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(
                -(
                    math.pow((j + 1 - center_x) / (sigma_horz * width), 2) / 2.0
                    + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0
                )
            )
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1:
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [
        int(max(1, -ul[0])),
        int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0])),
    ]
    g_y = [
        int(max(1, -ul[1])),
        int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1])),
    ]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert g_x[0] > 0 and g_y[1] > 0
    image[img_y[0] - 1 : img_y[1], img_x[0] - 1 : img_x[1]] = (
        image[img_y[0] - 1 : img_y[1], img_x[0] - 1 : img_x[1]]
        + g[g_y[0] - 1 : g_y[1], g_x[0] - 1 : g_x[1]]
    )
    image[image > 1] = 1

    return image


def transform(point, center, scale, resolution, invert=False):
    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=torch.float32)
    else:
        center = center.float()

    _pt = torch.ones(3, dtype=torch.float32)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3, dtype=torch.float32)
    scale_factor = float(resolution / h)
    t[0, 0] = scale_factor
    t[1, 1] = scale_factor
    t[0, 2] = float(resolution * (-center[0].item() / h + 0.5))
    t[1, 2] = float(resolution * (-center[1].item() / h + 0.5))

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop_torch(image, center, scale, resolution=256.0):
    l1 = transform([1, 1], center, scale, resolution, True)
    l2 = transform([resolution, resolution], center, scale, resolution, True)

    new_img = torch.zeros(
        (image.shape[0], image.shape[1], l2[1] - l1[1], l2[0] - l1[0])
    )
    height, width = image.shape[2], image.shape[3]

    new_x = torch.Tensor([max(1, -l1[0] + 1), min(l2[0], width) - l1[0]])
    new_y = torch.Tensor([max(1, -l1[1] + 1), min(l2[1], height) - l1[1]])
    old_x = torch.Tensor([max(1, l1[0] + 1), min(l2[0], width)])
    old_y = torch.Tensor([max(1, l1[1] + 1), min(l2[1], height)])

    new_img[
        :,
        :,
        int(new_y[0].data.item()) - 1 : int(new_y[1].data.item()),
        int(new_x[0].data.item()) - 1 : int(new_x[1].data.item()),
    ] = image[
        :,
        :,
        int(old_y[0].data.item()) - 1 : int(old_y[1].data.item()),
        int(old_x[0].data.item()) - 1 : int(old_x[1].data.item()),
    ]

    transformations = transforms.Resize((256, 256))
    new_img = transformations(new_img)

    return new_img


def get_preds_fromhm(hm, center=None, scale=None):
    _, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx = idx + 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            p_x, p_y = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if p_x > 0 and p_x < 63 and p_y > 0 and p_y < 63:
                diff = torch.FloatTensor(
                    [
                        hm_[p_y, p_x + 1] - hm_[p_y, p_x - 1],
                        hm_[p_y + 1, p_x] - hm_[p_y - 1, p_x],
                    ]
                )
                preds[i, j].add_(diff.sign_().mul_(0.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True
                )

    return preds, preds_orig


def shuffle_lr(parts, pairs=None):
    if pairs is None:
        pairs = [
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            26,
            25,
            24,
            23,
            22,
            21,
            20,
            19,
            18,
            17,
            27,
            28,
            29,
            30,
            35,
            34,
            33,
            32,
            31,
            45,
            44,
            43,
            42,
            47,
            46,
            39,
            38,
            37,
            36,
            41,
            40,
            54,
            53,
            52,
            51,
            50,
            49,
            48,
            59,
            58,
            57,
            56,
            55,
            64,
            63,
            62,
            61,
            60,
            67,
            66,
            65,
        ]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts


def flip(tensor, is_label=False):
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor


def pad_img_to_fit_bbox(img, x1, x2, y1, y2, crop_box):
    img_or = img.copy()
    img = cv2.copyMakeBorder(
        img,
        -min(0, y1),
        max(y2 - img.shape[0], 0),
        -min(0, x1),
        max(x2 - img.shape[1], 0),
        cv2.BORDER_REFLECT,
    )

    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)

    pad = crop_box
    pad = (
        max(-pad[0], 0),
        max(-pad[1], 0),
        max(pad[2] - img_or.shape[1], 0),
        max(pad[3] - img_or.shape[0], 0),
    )

    h, w, _ = img.shape
    y, x, _ = np.ogrid[:h, :w, :1]
    pad_arr = np.array(pad, dtype=np.float32)
    pad_arr[pad_arr == 0] = 1e-10
    mask = np.maximum(
        1.0
        - np.minimum(np.float32(x) / pad_arr[0], np.float32(w - 1 - x) / pad_arr[2]),
        1.0
        - np.minimum(np.float32(y) / pad_arr[1], np.float32(h - 1 - y) / pad_arr[3]),
    )
    img = np.array(img, dtype=np.float32)
    blur = 5.0
    img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
        mask * 3.0 + 1.0, 0.0, 1.0
    )
    img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)

    return img, x1, x2, y1, y2


def crop_from_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2, bbox)
    return img[y1:y2, x1:x2]


def crop_using_landmarks(image, landmarks):
    image_size = 256
    center = ((landmarks.min(0) + landmarks.max(0)) / 2).round().astype(int)
    size = int(
        max(
            landmarks[:, 0].max() - landmarks[:, 0].min(),
            landmarks[:, 1].max() - landmarks[:, 1].min(),
        )
    )
    try:
        center[1] -= size // 6
    except Exception:
        return None

    img = Image.fromarray(image)
    crop_box = (center[0] - size, center[1] - size, center[0] + size, center[1] + size)
    image = crop_from_bbox(image, crop_box)
    try:
        img = Image.fromarray(image.astype(np.uint8))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        pix = np.array(img)
        return pix
    except Exception:
        return None


def read_image_opencv(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype("uint8")


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        scale = r
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        scale = r

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized, scale
