import tensorflow as tf
import numpy as np
import zarr


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

def create_label_volume(patch_coords, patch_labels, patch_size, num_classes):
    """
    Create a dense label volume from sparse particle coordinates and labels.

    Args:
    - patch_coords: Numpy array of shape (N, 3) containing the coordinates of particles within the patch.
    - patch_labels: Numpy array of shape (N,) containing the labels for the particles.
    - patch_size: Tuple representing the size of the patch (depth, height, width).
    - num_classes: Number of particle types.

    Returns:
    - label_volume: Dense label volume of shape (depth, height, width, 1).
    """
    label_volume = np.zeros(patch_size, dtype=np.int32)  # Initialize the label volume with zeros

    for coord, label in zip(patch_coords, patch_labels):
        z, y, x = map(int, coord)
        if 0 <= z < patch_size[0] and 0 <= y < patch_size[1] and 0 <= x < patch_size[2]:
            label_volume[z, y, x] = int(label)  # Assign the label to the corresponding voxel

    label_volume = np.expand_dims(label_volume, axis=-1)  # Add the channel dimension
    return label_volume


def data_generator(tomogram_paths, patch_size, voxel_spacing, particle_type_mapping, crop=True):
    for path in tomogram_paths:
        # Load tomogram and annotations
        zarr_data = zarr.open(path, mode='r')[0]
        annotations_with_types, _ = load_annotations(path, voxel_spacing, particle_type_mapping)
        annotations = annotations_with_types[:, :3]  # Coordinates
        particle_type_labels = annotations_with_types[:, 3]

        # Generate patches
        for x in range(0, zarr_data.shape[0] - patch_size[0] + 1, patch_size[0]):
            for y in range(0, zarr_data.shape[1] - patch_size[1] + 1, patch_size[1]):
                for z in range(0, zarr_data.shape[2] - patch_size[2] + 1, patch_size[2]):
                    # Extract patch
                    patch = zarr_data[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]

                    # Find annotations in the patch
                    indices = np.where(
                        (annotations[:, 0] >= x) & (annotations[:, 0] < x + patch_size[0]) &
                        (annotations[:, 1] >= y) & (annotations[:, 1] < y + patch_size[1]) &
                        (annotations[:, 2] >= z) & (annotations[:, 2] < z + patch_size[2])
                    )[0]

                    if crop and len(indices) == 0:
                        continue  # Skip patches without annotations

                    patch_annotations = annotations[indices] - np.array([x, y, z])
                    patch_particle_types = particle_type_labels[indices]

                    # Create label volume
                    label_volume = create_label_volume(
                        patch_annotations,
                        patch_particle_types,
                        patch_size,
                        num_classes=len(particle_type_mapping)
                    )

                    # One-hot encode labels
                    one_hot_labels = one_hot_encode_labels(label_volume, num_classes=len(particle_type_mapping))

                    # Expand dimensions for the patch
                    patch = np.expand_dims(patch, axis=-1)

                    # Ensure only patch and labels are yielded
                    yield patch, one_hot_labels


def one_hot_encode_labels(labels, num_classes):
    """
    Converts integer labels to one-hot encoded labels.

    Args:
    - labels: Tensor of shape (depth, height, width, 1).
    - num_classes: Number of particle types.

    Returns:
    - One-hot encoded labels of shape (depth, height, width, num_classes).
    """
    print("Input labels shape:", labels.shape)  # Expect (depth, height, width, 1)
    if labels.shape[-1] != 1:
        raise ValueError(f"Expected labels to have shape (depth, height, width, 1), but got {labels.shape}")
    labels = tf.squeeze(labels, axis=-1)  # Remove the channel dimension
    one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)  # Add the class dimension
    print("One-hot encoded shape:", one_hot.shape)  # Expect (depth, height, width, num_classes)
    return one_hot


def get_dataset(
    tomogram_paths,
    patch_size,
    batch_size,
    voxel_spacing,
    augmentations=None,
    shuffle_buffer_size=0,
    particle_type_mapping={},
    num_classes=6,  # Number of particle types
):
    def generator():
        for patch, one_hot_labels in data_generator(
                tomogram_paths, patch_size, voxel_spacing, particle_type_mapping
        ):
            # Debug: Print shapes
            print(f"Patch shape before augmentations: {patch.shape}")  # Should be (128, 128, 128, 1)
            print(f"One-hot labels shape before augmentations: {one_hot_labels.shape}")  # Should be (128, 128, 128, 6)

            # Optionally apply augmentations
            if augmentations:
                for aug in augmentations:
                    patch, one_hot_labels = aug(patch, one_hot_labels)

            print(f"Patch shape after augmentations: {patch.shape}")
            print(f"One-hot labels shape after augmentations: {one_hot_labels.shape}")

            yield patch, one_hot_labels

    # Create a TensorFlow Dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=patch_size + (1,), dtype=tf.float32),  # Patch
            tf.TensorSpec(shape=patch_size + (num_classes,), dtype=tf.float32),  # One-hot labels
        )
    )

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Shuffle, batch, and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
