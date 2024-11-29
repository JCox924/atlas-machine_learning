"""
Proprocessing data for Conv3d model
"""
import numpy as np
from tensorflow import keras as K
import zarr
import napari
import random
from scipy.ndimage import gaussian_filter

load_annotations = __import__('1-data_exploration').load_annotations


def add_gaussian_noise(data, mean=0, std=0.05):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise


def apply_gaussian_blur(data, sigma=1.0):
    return gaussian_filter(data, sigma=sigma)


def random_flip(data, axis=None):
    if axis is None:
        axis = random.choice([0, 1, 2])
    return np.flip(data, axis=axis)


def random_rotate_90(data):
    k = random.choice([1, 2, 3])  # Number of 90-degree rotations
    axes = random.sample(range(data.ndim), 2)
    return np.rot90(data, k=k, axes=axes)


def preprocess_data(path, mode="3D", crop=True, patch_size=(32, 32, 32), slice_axis=0, augment=None):
    """
        Preprocess tomograms with optional augmentations for training and analysis.

        Parameters:
        - path (str): Path to the tomogram zarr file.
        - mode (str): Processing mode, either "3D" (patches) or "2D" (slices).
        - crop (bool): Whether to crop to regions with annotations.
        - patch_size (tuple): Size of 3D patches to extract (only used in "3D" mode).
        - slice_axis (int): Axis along which to extract 2D slices (only used in "2D" mode).
        - augmentations (list): List of augmentation functions to apply to the data.

        Returns:
        - processed_data (list): List of augmented patches or slices.
        - annotations (list): List of annotations mapped to the output data.
        """
    zarr_data = zarr.open(path, mode='r')[0]
    processed_data = []
    updated_annotations = []
    updated_particle_types = []

    # Load annotations
    voxel_spacing = [10.0, 10.0, 10.0]
    annotations, particle_types = load_annotations(path, voxel_spacing)

    if mode == "3D":
        for x in range(0, zarr_data.shape[0] - patch_size[0] + 1, patch_size[0]):
            for y in range(0, zarr_data.shape[1] - patch_size[1] + 1, patch_size[1]):
                for z in range(0, zarr_data.shape[2] - patch_size[2] + 1, patch_size[2]):
                    patch = zarr_data[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]

                    patch_annotations = []
                    patch_particle_types = []
                    for ann, particle_type in zip(annotations, particle_types):
                        if x <= ann[0] < x + patch_size[0] and y <= ann[1] < y + patch_size[1] and z <= ann[2] < z + \
                                patch_size[2]:
                            patch_annotations.append(
                                [ann[0] - x, ann[1] - y, ann[2] - z]
                            )
                            patch_particle_types.append(particle_type)

                    if crop and not patch_annotations:
                        continue

                    if augmentations:
                        patch = apply_augment(patch, augmentations)

                    processed_data.append(patch)
                    updated_annotations.append(patch_annotations)
                    updated_particle_types.append(patch_particle_types)


    elif mode == "2D":

        slice_ranges = range(zarr_data.shape[slice_axis])

        for idx in slice_ranges:

            if slice_axis == 0:

                slice_data = zarr_data[idx, :, :]

            elif slice_axis == 1:

                slice_data = zarr_data[:, idx, :]

            else:

                slice_data = zarr_data[:, :, idx]

            slice_annotations = []

            slice_particle_types = []

            for ann, particle_type in zip(annotations, particle_types):

                if (

                        (slice_axis == 0 and ann[0] == idx) or

                        (slice_axis == 1 and ann[1] == idx) or

                        (slice_axis == 2 and ann[2] == idx)

                ):
                    slice_annotations.append(ann)

                    slice_particle_types.append(particle_type)

            if crop and not slice_annotations:
                continue

            if augmentations:
                slice_data = apply_augment(slice_data, augmentations)

            processed_data.append(slice_data)

            updated_annotations.append(slice_annotations)

            updated_particle_types.append(slice_particle_types)

    return processed_data, updated_annotations, updated_particle_types

def apply_augment(data, augments):
    """
    Apply a list of augmentation functions to the data.

    Parameters:
    - data (np.ndarray): The input data to augment.
    - augmentations (list): List of augmentation functions.

    Returns:
    - np.ndarray: Augmented data.
    """
    for augmentation in augments:
        data = augmentation(data)
    return data


def visualize_patches_with_napari(patches, annotations=None, particle_types=None, colors=None):
    """
    Visualize 3D patches interactively with Napari, including color-coded annotations.

    Parameters:
    - patches (list): List of 3D patches (numpy arrays).
    - annotations (list): Corresponding annotations for each patch (optional).
    - particle_types (list): List of particle types for annotations.
    - colors (dict): Dictionary mapping particle types to colors.
    """
    viewer = napari.Viewer()

    for i, patch in enumerate(patches):
        viewer.add_image(patch, name=f"Patch {i}", colormap="gray")

        if annotations and i < len(annotations):
            for annotation, particle_type in zip(annotations[i], particle_types[i]):
                local_annotation = np.array(annotation)  # Local coordinates
                if colors and particle_type in colors:
                    color = colors[particle_type]
                else:
                    color = "red"

                viewer.add_points(
                    local_annotation,
                    name=f"Annotations {i} ({particle_type})",
                    size=3,
                    face_color=color
                )

    napari.run()



if __name__ == '__main__':
    augmentations = [
        lambda x: add_gaussian_noise(x, std=0.05),
        lambda x: apply_gaussian_blur(x, sigma=1.5),
    ]

    patches, patch_annotations, patch_particle_types = preprocess_data(
        path="train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr",
        mode="3D",
        crop=True,
        patch_size=(128, 128, 128),
        augment=augmentations
    )
    print(f"Number of augmented 3D patches: {len(patches)}")

    particle_colors = {
        "apo-ferritin": "blue",
        "beta-amylase": "orange",
        "beta-galactosidase": "green",
        "ribosome": "yellow",
        "thyroglobulin": "purple",
        "virus-like-particle": "red"
    }

    visualize_patches_with_napari(patches, patch_annotations, patch_particle_types, colors=particle_colors)
