import matplotlib.pyplot as plt
import numpy as np
import napari
from typing import List, Tuple
import tensorflow as tf

def visualize_sample(patch: np.ndarray, heatmap: np.ndarray, annotations: np.ndarray, slice_index: int = None):
    if slice_index is None:
        slice_index = patch.shape[0] // 2  # Middle slice

    # Extract the slice
    patch_slice = patch[slice_index]
    heatmap_slice = heatmap[slice_index]

    # Plotting
    plt.figure(figsize=(18, 6))

    # Original Image Slice
    plt.subplot(1, 3, 1)
    plt.imshow(patch_slice, cmap='gray')
    plt.title('Original Patch Slice')
    plt.axis('off')

    # Heatmap Slice
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_slice, cmap='hot')
    plt.title('Heatmap Slice')
    plt.axis('off')

    # Overlay Annotations on Image Slice
    plt.subplot(1, 3, 3)
    plt.imshow(patch_slice, cmap='gray')

    if annotations.size > 0:
        # Find annotations in the current slice
        slice_annotations = annotations[np.abs(annotations[:, 0] - slice_index) < 1]
        if len(slice_annotations) > 0:
            plt.scatter(slice_annotations[:, 2], slice_annotations[:, 1], c='red', s=20, marker='x')
    plt.title('Annotations Overlay')
    plt.axis('off')

    plt.show()

def visualize_batch(dataset: tf.data.Dataset, num_samples: int = 1):
    """
    Visualize a batch of samples from the dataset.

    Parameters:
    - dataset (tf.data.Dataset): The dataset to visualize samples from.
    - num_samples (int): Number of samples to visualize.
    """
    for batch in dataset.take(1):
        patches, heatmaps = batch
        for i in range(min(num_samples, patches.shape[0])):
            patch = patches[i].numpy().squeeze()
            heatmap = heatmaps[i].numpy().squeeze()
            # Create an empty annotations array with shape (0, 3)
            visualize_sample(patch, heatmap, annotations=np.empty((0, 3)))

def visualize_with_napari(patch: np.ndarray, annotations: np.ndarray, heatmap: np.ndarray = None):
    """
    Use Napari to visualize a 3D patch with annotations and optional heatmap.

    Parameters:
    - patch (np.ndarray): The 3D image patch.
    - annotations (np.ndarray): Array of annotation coordinates.
    - heatmap (np.ndarray): The corresponding heatmap (optional).
    """
    viewer = napari.Viewer()
    viewer.add_image(patch, name='Patch', colormap='gray', blending='additive')

    if heatmap is not None:
        viewer.add_image(heatmap, name='Heatmap', colormap='hot', blending='additive', opacity=0.5)

    if len(annotations) > 0:
        # Annotations need to be in (z, y, x) format
        viewer.add_points(annotations, size=5, face_color='red', name='Annotations')

    napari.run()
