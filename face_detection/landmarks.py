import os

import torch
from torch.utils.model_zoo import load_url

from .detection import SFDDetector
from .enums import LandmarksType, NetworkSize
from .models import FAN, ResNetDepth
from .ops import crop_torch, draw_gaussian, flip, get_preds_fromhm

models_urls = {
    "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
    "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar",
    "depth": "https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar",
}


class LandmarksEstimation:
    def __init__(
        self,
        type="3D",
        path_to_detector="./models/pretrained_models/s3fd-619a316812.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_size = NetworkSize.LARGE
        network_size = int(network_size)
        if type == "3D":
            self.landmarks_type = LandmarksType._3D
        else:
            self.landmarks_type = LandmarksType._2D
        self.flip_input = False

        if not os.path.exists(path_to_detector):
            print(
                "Pretrained model of SFD face detector does not exist in {}".format(
                    path_to_detector
                )
            )
            exit()
        self.face_detector = SFDDetector(
            device=self.device, path_to_detector=path_to_detector
        )

        self.face_alignment_net = FAN(network_size)
        if self.landmarks_type == LandmarksType._2D:
            network_name = "2DFAN-" + str(network_size)
        else:
            network_name = "3DFAN-" + str(network_size)
        fan_weights = load_url(
            models_urls[network_name], map_location=lambda storage, _loc: storage
        )
        self.face_alignment_net.load_state_dict(fan_weights)
        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()

        if self.landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_weights = load_url(
                models_urls["depth"], map_location=lambda storage, _loc: storage
            )
            depth_dict = {
                k.replace("module.", ""): v
                for k, v in depth_weights["state_dict"].items()
            }
            self.depth_prediciton_net.load_state_dict(depth_dict)
            self.depth_prediciton_net.to(self.device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, face, image):
        center = torch.FloatTensor(
            [(face[2] + face[0]) / 2.0, (face[3] + face[1]) / 2.0]
        )

        center[1] = center[1] - (face[3] - face[1]) * 0.12
        scale = (
            face[2] - face[0] + face[3] - face[1]
        ) / self.face_detector.reference_scale

        inp = crop_torch(image, center, scale).float().cuda()
        inp = inp.div(255.0)

        out = self.face_alignment_net(inp)[-1]

        if self.flip_input:
            out = out + flip(self.face_alignment_net(flip(inp))[-1], is_label=True)
        out = out.cpu()

        pts, pts_img = get_preds_fromhm(out, center, scale)
        out = out.cuda()
        if self.landmarks_type == LandmarksType._3D:
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            heatmaps = torch.zeros((68, 256, 256), dtype=torch.float32)
            for i in range(68):
                if pts[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)

            heatmaps = heatmaps.unsqueeze(0)

            heatmaps = heatmaps.to(self.device)
            depth_pred = self.depth_prediciton_net(torch.cat((inp, heatmaps), 1)).view(
                68, 1
            )

            pts_img = pts_img.cuda()
            pts_img = torch.cat(
                (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1
            )
        else:
            pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)

        return pts_img, out

    def detect_landmarks(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        if self.device == "cuda":
            image = image.cuda()

        with torch.no_grad():
            detected_faces = self.face_detector.detect_from_batch(image)

            if self.landmarks_type == LandmarksType._3D:
                landmarks = torch.empty((1, 68, 3))
            else:
                landmarks = torch.empty((1, 68, 2))

            found = False
            for face in detected_faces[0]:
                conf = face[4]
                if conf > 0.99:
                    pts_img, _ = self.get_landmarks(face, image)
                    landmarks[0] = pts_img
                    found = True

        if not found:
            return None

        return landmarks
