#!/usr/bin/env python3
"""
Module 5-yolo contains class:
    Yolo
"""
import numpy as np
import os
import cv2
import tensorflow.keras as K


class Yolo:
    """
    Yolo class contains methods:
        __init__(self, model_path, classes_path, class_t, nms_t, anchors)
        process_outputs(self, outputs, image_size)
        filter_boxes(self, boxes, box_confidences, box_class_probs)
        non_max_suppression(self, filtered_boxes, box_classes, box_scores)
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

    @staticmethod
    def load_images(folder_path):
        """
        Args:
            folder_path: {string} representing the path to the
                folder holding all the images to load
        Returns:
            a tuple of (images, image_paths)
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses the image

        Args:
            image: list of images as numpy.ndarrays
        Returns:
            a tuple of (pimages, image_shapes)
        """
        input_height = self.model.input.shape[1]
        input_width = self.model.input.shape[2]

        pimages = []
        pimage_shapes = []

        for image in images:
            pimage_shapes.append(image.shape[:2])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(image, (input_width, input_height),
                                       interpolation=cv2.INTER_CUBIC)

            simage = resized_image / 255.0
            pimages.append(simage)

        pimages = np.array(pimages)
        pimage_shapes = np.array(pimage_shapes)

        return pimages, pimage_shapes

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

            c_x = np.arange(grid_width)
            c_y = np.arange(grid_height)
            c_x, c_y = np.meshgrid(c_x, c_y)

            c_x = c_x[..., np.newaxis]
            c_y = c_y[..., np.newaxis]

            bx = (tx + c_x) / grid_width
            by = (ty + c_y) / grid_height

            anchors = anchors.reshape((1, 1, anchor_boxes, 2))
            pw = anchors[..., 0]
            ph = anchors[..., 1]

            tw = np.exp(tw) * pw / input_width
            th = np.exp(th) * ph / input_height

            x1 = (bx - tw / 2) * image_width
            y1 = (by - th / 2) * image_height
            x2 = (bx + tw / 2) * image_width
            y2 = (by + th / 2) * image_height

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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max suppression to filtered bounding boxes.

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4)
                containing the filtered bounding boxes
            box_classes: numpy.ndarray of shape (?,)
                containing the class number for the class
            box_scores: numpy.ndarray of shape (?)
                containing the box scores for each box

        Returns:
            tuple of (box_predictions,
                      predicted_box_classes,
                      predicted_box_scores):
                box_predictions: numpy.ndarray of shape (?, 4)
                predicted_box_classes: numpy.ndarray of shape (?,)
                predicted_box_scores: numpy.ndarray of shape (?)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_box_scores = box_scores[idxs]

            sorted_idx = np.argsort(-cls_box_scores)
            cls_boxes = cls_boxes[sorted_idx]
            cls_box_scores = cls_box_scores[sorted_idx]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                box_area = ((cls_boxes[0, 2] - cls_boxes[0, 0])
                            * (cls_boxes[0, 3] - cls_boxes[0, 1]))
                cls_boxes_areas = ((cls_boxes[1:, 2] - cls_boxes[1:, 0])
                                   * (cls_boxes[1:, 3] - cls_boxes[1:, 1]))

                iou = inter_area / (box_area + cls_boxes_areas - inter_area)

                keep_idxs = np.where(iou < self.nms_t)[0]
                cls_boxes = cls_boxes[keep_idxs + 1]
                cls_box_scores = cls_box_scores[keep_idxs + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
