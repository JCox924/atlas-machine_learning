import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def add_gaussian_noise(image, annotations, mean=0.0, stddev_range=(0.01, 0.1)):
    stddev = tf.random.uniform([], *stddev_range)
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
    image = image + noise
    return image, annotations

def apply_gaussian_blur(image, annotations, kernel_size=3, sigma_range=(0.5, 2.0)):
    sigma = tf.random.uniform([], *sigma_range)
    image = tfa.image.gaussian_filter2d(image, filter_shape=kernel_size, sigma=sigma)
    return image, annotations

def random_flip_3d(image, annotations_with_types):
    axes = [0, 1, 2]  # D, H, W axes
    image_shape = tf.shape(image)
    for axis in axes:
        should_flip = tf.random.uniform([]) > 0.5

        def flip():
            flipped_image = tf.reverse(image, axis=[axis])
            flipped_annotations_with_types = flip_annotations(annotations_with_types, image_shape, axis)
            return flipped_image, flipped_annotations_with_types

        def no_flip():
            return image, annotations_with_types

        image, annotations_with_types = tf.cond(should_flip, flip, no_flip)

    return image, annotations_with_types


def rotate_image_90(image, k, axes):
    # Ensure k is within 0-3
    k = k % 4
    rank = tf.rank(image)

    if k == 0:
        return image
    elif k == 1:
        # Rotate 90 degrees
        perm = get_permutation(axes, rank)
        image = tf.transpose(image, perm=perm)
        image = tf.reverse(image, axis=[axes[1]])
    elif k == 2:
        # Rotate 180 degrees
        image = tf.reverse(image, axis=list(axes))
    elif k == 3:
        # Rotate 270 degrees
        perm = get_permutation(axes, rank)
        image = tf.transpose(image, perm=perm)
        image = tf.reverse(image, axis=[axes[0]])
    return image


def swap_elements(tensor, index1, index2):
    # Create indices for the elements to swap
    indices = [[index1], [index2]]
    # Gather the elements at the specified indices
    elems = tf.gather_nd(tensor, indices)
    # Create updates in swapped order
    updates = [elems[1], elems[0]]
    # Update the tensor with swapped elements
    tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)
    return tensor


def get_permutation(axes, rank):
    # Create a tensor of dimensions
    perm = tf.range(rank)
    # Swap the axes for transposition using TensorFlow operations
    perm = swap_elements(perm, axes[0], axes[1])
    return perm


def create_rotation_matrix(angle, axis1, axis2):
    c = tf.cos(angle)
    s = tf.sin(angle)
    rotation_matrix = tf.eye(3)
    indices = [[axis1, axis1], [axis1, axis2], [axis2, axis1], [axis2, axis2]]
    updates = [c, -s, s, c]
    rotation_matrix = tf.tensor_scatter_nd_update(rotation_matrix, indices, updates)
    return rotation_matrix


def rotate_3d_annotations(annotations, image_shape, k, axes):
    """
    Rotate 3D annotations (coordinates) by 90-degree increments.

    Args:
    - annotations: Tensor of shape [N, 4] where the first three columns are (z, y, x).
    - image_shape: Shape of the image tensor.
    - k: Number of 90-degree rotations (0, 1, 2, or 3).
    - axes: Axes along which to perform the rotation (e.g., (1, 2) for y and x).

    Returns:
    - Rotated annotations: Tensor of shape [N, 4] with rotated coordinates.
    """
    coords = annotations[:, :3]  # Extract (z, y, x) coordinates
    particle_types = annotations[:, 3:]  # Extract particle types if present

    # Select axes for rotation
    axis1, axis2 = axes

    # Define rotation matrices for 90-degree increments
    if k == 1:  # 90 degrees
        rotation_matrix = tf.constant([[0, -1], [1, 0]], dtype=tf.float32)
    elif k == 2:  # 180 degrees
        rotation_matrix = tf.constant([[-1, 0], [0, -1]], dtype=tf.float32)
    elif k == 3:  # 270 degrees
        rotation_matrix = tf.constant([[0, 1], [-1, 0]], dtype=tf.float32)
    else:  # 0 degrees (no rotation)
        return annotations

    # Extract and rotate coordinates along the specified axes
    coords_to_rotate = tf.stack([coords[:, axis1], coords[:, axis2]], axis=1)  # Shape: [N, 2]
    rotated_coords = tf.matmul(coords_to_rotate, rotation_matrix)  # Shape: [N, 2]

    # Update the rotated coordinates into the original coordinates tensor
    coords_rotated = tf.identity(coords)
    coords_rotated = tf.tensor_scatter_nd_update(
        coords_rotated,
        tf.expand_dims(tf.range(tf.shape(coords_rotated)[0]), axis=-1),
        tf.concat([coords[:, :axis1], rotated_coords[:, :], coords[:, axis2 + 1:]], axis=1)
    )

    # Combine the rotated coordinates with particle types (if any)
    if particle_types.shape[1] > 0:  # Check if particle_types is not empty
        rotated_annotations = tf.concat([coords_rotated, particle_types], axis=1)
    else:
        rotated_annotations = coords_rotated

    return rotated_annotations

def random_rotate_90(image, annotations_with_types):
    """
    Randomly rotates a 3D image and its annotations by 90-degree increments.

    Args:
    - image: 3D image tensor of shape [depth, height, width, channels].
    - annotations_with_types: Tensor of shape [N, 4] containing (z, y, x, label).

    Returns:
    - Rotated image.
    - Rotated annotations_with_types.
    """
    axes = (1, 2)  # Example: rotate along y and x axes
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    # Rotate the image
    rotated_image = tf.image.rot90(image, k=k)

    # Rotate the annotations
    image_shape = tf.shape(image)
    if tf.size(annotations_with_types) > 0:  # Ensure annotations are not empty
        rotated_annotations = rotate_3d_annotations(annotations_with_types, image_shape, k, axes)
    else:
        rotated_annotations = annotations_with_types

    return rotated_image, rotated_annotations


def flip_annotations(annotations_with_types, image_shape, axis):
    """
    Flip annotations along the specified axis.

    Parameters:
    - annotations_with_types: Tensor of shape (N, 4) or (N, 3)
    - image_shape: Shape of the image tensor
    - axis: Axis along which to flip (0, 1, or 2)

    Returns:
    - Flipped annotations_with_types tensor of shape (N, 4)
    """
    # Handle empty annotations case
    if annotations_with_types.shape[0] == 0:
        print("No annotations to flip. Returning empty tensor.")
        return annotations_with_types

    # Ensure the tensor has the correct shape
    if annotations_with_types.shape[1] < 3:
        raise ValueError(
            f"Expected annotations_with_types to have at least 3 columns, "
            f"but got shape {annotations_with_types.shape}"
        )

    coords = annotations_with_types[:, :3]
    particle_types = annotations_with_types[:, 3] if annotations_with_types.shape[1] > 3 else None

    max_coord = tf.cast(image_shape[axis], tf.float32) - 1
    coords_flipped = tf.identity(coords)

    # Flip the coordinate along the specified axis
    coord_flipped = max_coord - coords_flipped[:, axis]
    coords_flipped = tf.concat([
        coords_flipped[:, :axis],
        tf.expand_dims(coord_flipped, axis=1),
        coords_flipped[:, axis+1:]
    ], axis=1)

    # Combine flipped coordinates with particle types (if present)
    if particle_types is not None:
        annotations_flipped = tf.concat([coords_flipped, tf.expand_dims(particle_types, axis=1)], axis=1)
    else:
        annotations_flipped = coords_flipped

    return annotations_flipped

def rotate_annotations(annotations, image_shape, k, axis):
    rotated_annotations = []
    for ann in annotations:
        ann = np.array(ann)
        dims = [ann[axis[0]], ann[axis[1]]]
        rotated_dims = np.rot90(
            np.array(dims).reshape(1, -1),
            k=k,
            axes=(1, 0)
        ).reshape(-1)
        ann[axis[0]], ann[axis[1]] = rotated_dims[0], rotated_dims[1]
        rotated_annotations.append(ann)
    return rotated_annotations

def apply_augmentations(image, annotations, augmentations, visualize=False):
    if visualize:
        # Visualize before augmentation
        from data_preprocessing.data_explorer import visualize_sample
        visualize_sample(image.numpy(), np.zeros_like(image.numpy()), annotations.numpy())

    for aug in augmentations:
        image, annotations = aug(image, annotations)

    if visualize:
        # Visualize after augmentation
        from data_preprocessing.data_explorer import visualize_sample
        visualize_sample(image.numpy(), np.zeros_like(image.numpy()), annotations.numpy())

    return image, annotations
