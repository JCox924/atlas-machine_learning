import tensorflow as tf
import numpy as np
import zarr
from torch import dtype

from data_preprocessing.utils import load_annotations, generate_heatmap


def load_tomogram(tomogram_path):
    zarr_data = zarr.open(tomogram_path, mode='r')

    # Print the Zarr file structure
    print(f"Zarr file structure for {tomogram_path}:")
    print(zarr_data.tree())

    # Check if zarr_data is a group or array
    if isinstance(zarr_data, zarr.hierarchy.Group):
        # Navigate through the group to find the array(s)
        # Adjust the keys based on the printed structure
        try:
            # Example: data stored under '0: float32, float32, float32'
            tomogram = zarr_data['0'][:]
        except KeyError as e:
            print(f"KeyError accessing data: {e}")
            # Handle the error or try alternative paths
            # You might need to adjust this part based on your data
            raise
    elif isinstance(zarr_data, zarr.core.Array):
        # zarr_data is an array; read it directly
        tomogram = zarr_data[:]
    else:
        raise TypeError(f"Unexpected zarr_data type: {type(zarr_data)}")

    # Ensure tomogram is a NumPy array
    tomogram = np.asarray(tomogram)
    print(f"Tomogram loaded. Shape: {tomogram.shape}, dtype: {tomogram.dtype}")
    return tomogram


def filter_annotations(annotations, particle_types, image_shape):
    """
    Filters out annotations that are outside the image boundaries.

    Parameters:
    - annotations: Tensor of shape (N, 3)
    - particle_types: Tensor of shape (N,)
    - image_shape: Shape of the image tensor

    Returns:
    - Filtered annotations tensor of shape (M, 3)
    - Filtered particle_types tensor of shape (M,)
    """
    valid_mask = tf.reduce_all(
        tf.logical_and(
            annotations >= 0,
            annotations < tf.cast(image_shape[:3], tf.float32)
        ),
        axis=1
    )

    annotations = tf.boolean_mask(annotations, valid_mask)
    particle_types = tf.boolean_mask(particle_types, valid_mask)
    return annotations, particle_types


def data_generator(tomogram_paths, patch_size, voxel_spacing, particle_type_mapping, crop=True):
    for path in tomogram_paths:
        # Load tomogram data
        zarr_data = zarr.open(path, mode='r')[0]
        zarr_shape = zarr_data.shape

        # Load annotations (now use the mapping from outside)
        annotations_with_types, _ = load_annotations(path, voxel_spacing, particle_type_mapping)  # Integer labels

        annotations = annotations_with_types[:, :3]  # Coordinates (z, y, x)
        particle_type_labels = annotations_with_types[:, 3]

        # Generate patches
        for x in range(0, zarr_shape[0] - patch_size[0] + 1, patch_size[0]):
            for y in range(0, zarr_shape[1] - patch_size[1] + 1, patch_size[1]):
                for z in range(0, zarr_shape[2] - patch_size[2] + 1, patch_size[2]):
                    # Extract patch
                    patch = zarr_data[
                            x:x + patch_size[0],
                            y:y + patch_size[1],
                            z:z + patch_size[2]
                            ]

                    # Find annotations within the patch
                    indices = np.where(
                        (annotations[:, 0] >= x) & (annotations[:, 0] < x + patch_size[0]) &
                        (annotations[:, 1] >= y) & (annotations[:, 1] < y + patch_size[1]) &
                        (annotations[:, 2] >= z) & (annotations[:, 2] < z + patch_size[2])
                    )[0]

                    if crop and len(indices) == 0:
                        continue  # Skip patches without annotations

                    patch_annotations = annotations[indices] - np.array([x, y, z])
                    patch_particle_types = particle_type_labels[indices]

                    # Combine annotations and particle types
                    patch_annotations_with_types = np.hstack([
                        patch_annotations,
                        patch_particle_types[:, np.newaxis]
                    ])

                    yield patch.astype(np.float32), patch_annotations_with_types.astype(np.float32)


def get_dataset(
    tomogram_paths,
    patch_size,
    batch_size,
    voxel_spacing,
    augmentations=None,
    shuffle_buffer_size=100,
    particle_type_mapping={
        "apo-ferritin": 0,
        "beta-amylase": 1,
        "beta-galactosidase": 2,
        "ribosome": 3,
        "thyroglobulin": 4,
        "virus-like-particle": 5,
    },
    sigma=3  # Gaussian width for heatmaps
):
    def generator():
        for tomogram_path in tomogram_paths:
            # Load the tomogram data
            tomogram = load_tomogram(tomogram_path)

            # Load annotations and ensure proper format
            annotations, _ = load_annotations(tomogram_path, voxel_spacing, particle_type_mapping)
            if not isinstance(annotations, np.ndarray):
                annotations = np.array(annotations)

            coords = annotations[:, :3]
            labels = annotations[:, 3]

            # Number of patches per tomogram
            num_patches_per_tomogram = 50  # Adjust as needed

            for _ in range(num_patches_per_tomogram):
                # Randomly select a starting point
                z = np.random.randint(0, max(1, tomogram.shape[0] - patch_size[0]))
                y = np.random.randint(0, max(1, tomogram.shape[1] - patch_size[1]))
                x = np.random.randint(0, max(1, tomogram.shape[2] - patch_size[2]))

                # Extract the patch
                patch = tomogram[z: z + patch_size[0], y: y + patch_size[1], x: x + patch_size[2]]

                # Adjust annotations to the patch coordinate system
                patch_coords = []
                patch_labels = []
                for coord, label in zip(coords, labels):
                    voxel_coord = coord.astype(int)
                    if (
                            voxel_coord[0] >= z
                            and voxel_coord[0] < z + patch_size[0]
                            and voxel_coord[1] >= y
                            and voxel_coord[1] < y + patch_size[1]
                            and voxel_coord[2] >= x
                            and voxel_coord[2] < x + patch_size[2]
                    ):
                        patch_coords.append(voxel_coord - np.array([z, y, x]))
                        patch_labels.append(label)

                # Handle cases with or without annotations
                if len(patch_coords) == 0:  # No annotations
                    annotations_with_types = tf.RaggedTensor.from_tensor(
                        tf.zeros([0, 4], dtype=tf.float32), ragged_rank=1
                    )
                else:  # Valid annotations
                    patch_coords = np.array(patch_coords, dtype=np.float32)  # Convert to numpy array
                    patch_labels = np.array(patch_labels, dtype=np.float32)  # Convert to numpy array
                    annotations_with_types = tf.RaggedTensor.from_tensor(
                        tf.convert_to_tensor(np.hstack([patch_coords, patch_labels[:, np.newaxis]]), dtype=tf.float32),
                        ragged_rank=1
                    )

                # Generate heatmap for the patch
                heatmap = generate_heatmap(patch_size, patch_coords, sigma=sigma)

                # Expand dimensions to add channel
                patch = np.expand_dims(patch, axis=-1)
                heatmap = np.expand_dims(heatmap, axis=-1)

                # Apply augmentations if any
                if augmentations:
                    for aug in augmentations:
                        patch, heatmap = aug(patch, heatmap)

                print(f"Patch coords: {patch_coords}")
                print(f"Patch labels: {patch_labels}")
                print(f"Annotations with types (shape): {annotations_with_types.shape}")

                # Yield the patch, heatmap, and annotations
                yield patch.astype(np.float32), heatmap.astype(np.float32), annotations_with_types

    # Create a TensorFlow Dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=patch_size + (1,), dtype=tf.float32),  # Patch
            tf.TensorSpec(shape=patch_size + (1,), dtype=tf.float32),  # Heatmap
            tf.RaggedTensorSpec(shape=[None, 4], dtype=tf.float32),    # Annotations with types
        )
    )

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
