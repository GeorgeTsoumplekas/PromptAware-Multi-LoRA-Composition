import numpy as np
import torch
import torch.nn.functional as F

from .models import s3fd


class FaceDetector(object):
    """An abstract class representing a face detector."""

    def __init__(self, device):
        self.device = device

    def detect_from_batch(self, tensor):
        raise NotImplementedError

    @property
    def reference_scale(self):
        raise NotImplementedError


def decode(loc, priors, variances):
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def batch_detect(net, img_batch):
    bb, _, _, _ = img_batch.size()

    with torch.no_grad():
        olist = net(img_batch.float())

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    bboxlists = []

    olist = [oelem.data.cpu() for oelem in olist]
    for j in range(bb):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))

            for _, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
                )
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)

    if 0 == len(bboxlists):
        bboxlists = np.zeros((1, 1, 5))

    return bboxlists


def nms(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None):
        super(SFDDetector, self).__init__(device)

        self.device = device
        model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(self.device)
        self.face_detector.eval()

    def detect_from_batch(self, tensor):
        bboxlists = batch_detect(self.face_detector, tensor)

        new_bboxlists = []
        for i in range(bboxlists.shape[0]):
            bboxlist = bboxlists[i]
            keep = nms(bboxlist, 0.3)
            if len(keep) > 0:
                bboxlist = bboxlist[keep, :]
                bboxlist = [x for x in bboxlist if x[-1] > 0.5]
                new_bboxlists.append(bboxlist)
            else:
                new_bboxlists.append([])

        return new_bboxlists

    @property
    def reference_scale(self):
        return 195
