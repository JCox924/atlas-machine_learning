#!/usr/bin/env python3
"""
Module 0-yolo contains class:
    Yolo
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class contains methods:
        __init__(self, model_path, classes_path, class_t, nms_t, anchors)
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize Yolo class instance

        Arguments:
                model_path {string} -- path to where a
                    Darknet Keras model is stored

                classes_path {string} -- path to where the list of class
                    names used for the Darknet model,
                        listed in order of index, can be found

                class_t {float} -- representing the box score threshold
                    for the initial filtering step

                nms_t {float} -- shape (outputs, anchor_boxes, 2)
                    containing all the anchor boxes

                anchors {numpy.ndarray} -- shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                self.class_names.append(line.rstrip())
            self.class_t = class_t
            self.nms_t = nms_t
            self.anchors = anchors
