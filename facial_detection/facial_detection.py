#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

class FacialDetection:

    def __init__(self, visualize=False):
        # mac gpu device
        self.device = "cpu"

        self.mtcnn  = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=not visualize,
            device=self.device
        )
        self.model = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)

    
    def detect_face(self, img: Image):
        """
        Detects faces in an image and returns image of cropped face. Returns only one face.
        To return all faces, set keep_all=True in mtcnn instance.
        """
        # resize image
        aligned = []
        x_aligned, prob = self.mtcnn(img, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)

        return torch.stack(aligned).to(self.device)
    
    def get_facial_embeddings(self, cropped_imgs: list[torch.Tensor]):
        """
        Returns facial embeddings of the cropped image.
        """
        embeddings = self.model(cropped_imgs).detach().cpu()
        return embeddings
    

    def get_embedding_distance(self, facial_embedding1, facial_embedding2):
        return (facial_embedding1 - facial_embedding2).norm().item()
