import os

import cv2
import numpy as np
import torch

from .ops import crop_using_landmarks, image_resize, read_image_opencv


def preprocess_image(image_path, landmarks_est, save_filename=None):
    if os.path.isfile(image_path):
        image = read_image_opencv(image_path)
    else:
        image = image_path
    image, _ = image_resize(image, width=1000)
    image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).float().cuda()

    with torch.no_grad():
        landmarks = landmarks_est.detect_landmarks(image_tensor.unsqueeze(0))
        if landmarks is None:
            print(f"No face detected in {image_path}, skipping.")
            return None
        landmarks = landmarks[0].detach().cpu().numpy()
        landmarks = np.asarray(landmarks)

        img = crop_using_landmarks(image, landmarks)
        if img is not None and save_filename is not None:
            cv2.imwrite(save_filename, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
        if img is not None:
            return img
        else:
            print("Error with image preprocessing")
            exit()
