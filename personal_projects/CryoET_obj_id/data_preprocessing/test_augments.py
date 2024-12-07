import tensorflow as tf
import numpy as np
import napari
from tensorflow.python.ops.numpy_ops import np_config
from data_preprocessing.data_preprocessor import get_dataset
from data_preprocessing.augmentations import (
    add_gaussian_noise,
    apply_gaussian_blur,
    random_flip_3d,
    random_rotate_90,
    apply_augmentations
)
np_config.enable_numpy_behavior()

tf.config.run_functions_eagerly(True)

tomogram_path = '../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/wbp.zarr'
train_tomogram_paths = [tomogram_path]
patch_size = (128, 128, 128)
batch_size = 1
voxel_spacing = [10.0, 10.0, 10.0]
sigma = 3


def test_augmentation(augmentation_fn, image, annotations, augmentation_name):
    """
    Test an augmentation visually with Napari.

    Parameters:
    - augmentation_fn: Function to apply the augmentation.
    - image: Original image data (numpy array).
    - annotations: Original annotations (numpy array of shape Nx3).
    - augmentation_name: String name of the augmentation.

    Returns:
    - Tuple of augmented image and augmented annotations.
    """
    print(f"Annotations shape: {annotations.shape}")
    print(f"Annotations: {annotations}")

    if annotations.size == 0:
        annotations = np.empty((0, 3), dtype=np.float32)

    augmented_image, augmented_annotations = augmentation_fn(image, annotations)

    if augmented_annotations.size == 0:
        augmented_annotations = np.empty((0, 3), dtype=np.float32)

    # Visualize using Napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="Original Image", colormap="gray")
        viewer.add_image(augmented_image, name=f"Augmented: {augmentation_name}", colormap="gray")
        viewer.add_points(annotations, size=5, name="Original Annotations")
        viewer.add_points(augmented_annotations, size=5, name="Augmented Annotations")

    return augmented_image, augmented_annotations


def main():

    test_dataset = get_dataset(
        tomogram_paths=train_tomogram_paths,
        patch_size=patch_size,
        batch_size=batch_size,
        voxel_spacing=voxel_spacing,
        sigma=sigma
    )

    # Extract one sample from the dataset
    for patches, heatmaps, annotations_batch in test_dataset.take(1):
        test_image = patches[0].numpy().squeeze()
        test_annotations = annotations_batch[0].numpy()
        break

    # List of augmentations to test
    augmentations_to_test = [
        (lambda img, ann: add_gaussian_noise(img, ann), "Gaussian Noise"),
        (lambda img, ann: apply_gaussian_blur(img, ann, kernel_size=3), "Gaussian Blur"),
        (random_flip_3d, "Random Flip 3D"),
        (random_rotate_90, "Random Rotate 90")
    ]

    for aug_fn, aug_name in augmentations_to_test:
        augmented_image, augmented_annotations = test_augmentation(
            aug_fn, test_image, test_annotations, augmentation_name=aug_name
        )

    combined_augmentations = [
        lambda img, ann: add_gaussian_noise(img, ann),
        lambda img, ann: apply_gaussian_blur(img, ann, kernel_size=5),
        random_flip_3d,
        random_rotate_90
    ]
    augmented_image, augmented_annotations = apply_augmentations(
        tf.convert_to_tensor(test_image, dtype=tf.float32),
        tf.convert_to_tensor(test_annotations, dtype=tf.float32),
        augmentations=combined_augmentations
    )
    test_augmentation(
        lambda img, ann: (augmented_image, augmented_annotations),
        test_image,
        test_annotations,
        augmentation_name="Combined Augmentations"
    )


if __name__ == "__main__":
    main()
