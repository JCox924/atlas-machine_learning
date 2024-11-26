#!/usr/bin/env python3
"""
Module 2-yolo contains class:
    Yolo
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class contains methods:
        __init__(self, model_path, classes_path, class_t, nms_t, anchors)
        process_outputs(self, outputs, image_size)
        filter_boxes(self, boxes, box_confidences, box_class_probs)
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize Yolo class instance

        Args:
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

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs for a single image

        Args:
            outputs: list of ndarrays of the predictions from the
                Darknet model for a single image
            image_size: numpy.ndarray containing the image's original size
                [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_shape = self.model.input.shape
        input_height = input_shape[1]
        input_width = input_shape[2]
        image_height, image_width = image_size

        for output, anchors in zip(outputs, self.anchors):
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            tx = 1 / (1 + np.exp(-tx))
            ty = 1 / (1 + np.exp(-ty))

            c_x = np.tile(np.arange(grid_width),
                          grid_height).reshape(grid_height, grid_width)
            c_y = np.tile(np.arange(grid_height),
                          grid_width).reshape(grid_width, grid_height).T

            c_x = c_x[..., np.newaxis]
            c_y = c_y[..., np.newaxis]

            bx = (tx + c_x) / grid_width
            by = (ty + c_y) / grid_height

            anchors = anchors.reshape((1, 1, anchor_boxes, 2))
            pw = anchors[..., 0]
            ph = anchors[..., 1]

            tw = np.exp(tw)
            th = np.exp(th)

            bw = (pw * tw) / input_width
            bh = (ph * th) / input_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes_per_output = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(boxes_per_output)

            box_confidence = 1 / (1 + np.exp(-output[..., 4]))
            box_confidence = box_confidence[..., np.newaxis]
            box_confidences.append(box_confidence)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters out boxes based on objectness score and class probabilities.

        Parameters:
        - boxes: list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 4)
        - box_confidences: list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 1)
        - box_class_probs: list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, classes)

        Returns:
        - tuple of (filtered_boxes, box_classes, box_scores):
          - filtered_boxes: numpy.ndarray of shape (?, 4)
          - box_classes: numpy.ndarray of shape (?,)
          - box_scores: numpy.ndarray of shape (?)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, c, p in zip(boxes, box_confidences, box_class_probs):
            scores = c * p
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        if len(filtered_boxes) == 0:
            return (np.array([]), np.array([]), np.array([]))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
