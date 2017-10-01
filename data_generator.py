# -*- coding: utf-8 -*-
"""Batch generator definition."""

import cv2
import numpy as np


class ImageBatchGenerator(object):
    """Batch generator for training on general images."""

    def __init__(self, input_files, batch_size, height, width, channel=3, shuffle=False):
        assert batch_size > 0, batch_size

        self._input_files = input_files
        self._batch_size = batch_size
        self._height = height
        self._width = width
        self._channel = channel  # not used
        self._shuffle = shuffle

        for ifile in input_files:
            image = cv2.imread(ifile, cv2.IMREAD_UNCHANGED)
            assert isinstance(image, np.ndarray)
            assert image.shape[:2] == (
                height, width), (image.shape[:2], (height, width))
            print('verify ' + ifile)

        self._batch_generator = self.__get_batch_generator()

    def __get_batch_generator(self):
        batch = []

        while True:
            if self._shuffle:
                file_index = np.random.permutation(self.n_samples)
            else:
                file_index = range(self.n_samples)

            for idx in file_index:
                image = cv2.imread(self._input_files[idx], cv2.IMREAD_COLOR)
                image = image.transpose((2, 0, 1))
                image = image.astype(np.float32)
                image = ((image / 255.) - 0.5) * 2.

                batch.append(image)

                if len(batch) == self._batch_size:
                    yield np.asarray(batch)
                    batch = []

    @property
    def n_samples(self):
        return len(self._input_files)

    def next(self):
        self._batch = self._batch_generator.next()
        return self._batch