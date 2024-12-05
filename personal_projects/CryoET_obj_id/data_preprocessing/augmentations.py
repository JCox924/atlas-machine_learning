import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def add_gaussian_noise(image, annotations, mean=0.0, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
    image = image + noise
    return image, annotations

def apply_gaussian_blur(image, annotations, kernel_size=3, sigma=1.0):
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

def flip_annotations(annotations_with_types, image_shape, axis):
    coords = annotations_with_types[:, :3]
    particle_types = annotations_with_types[:, 3]

    max_coord = tf.cast(image_shape[axis], tf.float32) - 1
    coords_flipped = tf.identity(coords)

    # Flip the coordinate along the specified axis
    coord_flipped = max_coord - coords_flipped[:, axis]
    coords_flipped = tf.concat([
        coords_flipped[:, :axis],
        tf.expand_dims(coord_flipped, axis=1),
        coords_flipped[:, axis+1:]
    ], axis=1)

    # Combine flipped coordinates with particle types
    annotations_flipped = tf.concat([coords_flipped, tf.expand_dims(particle_types, axis=1)], axis=1)

    return annotations_flipped


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


def rotate_3d_annotations(annotations_with_types, image_shape, k, axes):
    coords = annotations_with_types[:, :3]
    particle_types = annotations_with_types[:, 3]

    # Center the annotations
    center = tf.cast(image_shape[:3], tf.float32) / 2.0
    coords_centered = coords - center

    # Create rotation matrix
    angle = tf.cast(k, tf.float32) * (np.pi / 2)
    rotation_matrix = create_rotation_matrix(angle, axes[0], axes[1])

    # Rotate coordinates
    coords_rotated = tf.matmul(coords_centered, rotation_matrix)

    # Shift back
    coords_rotated += center

    # Combine rotated coordinates with particle types
    annotations_rotated = tf.concat([coords_rotated, tf.expand_dims(particle_types, axis=1)], axis=1)

    return annotations_rotated

def random_rotate_90(image, annotations_with_types):
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    axes = (0, 1)  # Adjust axes as needed

    # Rotate image manually
    image = rotate_image_90(image, k, axes)

    # Rotate annotations
    annotations_with_types = rotate_3d_annotations(annotations_with_types, tf.shape(image), k, axes)

    return image, annotations_with_types


def flip_annotations(annotations_with_types, image_shape, axis):
    """
    Flip annotations along the specified axis.

    Parameters:
    - annotations_with_types: Tensor of shape (N, 4)
    - image_shape: Shape of the image tensor
    - axis: Axis along which to flip (0, 1, or 2)

    Returns:
    - Flipped annotations_with_types tensor of shape (N, 4)
    """
    coords = annotations_with_types[:, :3]
    particle_types = annotations_with_types[:, 3]

    max_coord = tf.cast(image_shape[axis], tf.float32) - 1
    coords_flipped = tf.identity(coords)

    # Flip the coordinate along the specified axis
    coord_flipped = max_coord - coords_flipped[:, axis]
    coords_flipped = tf.concat([
        coords_flipped[:, :axis],
        tf.expand_dims(coord_flipped, axis=1),
        coords_flipped[:, axis+1:]
    ], axis=1)

    # Combine flipped coordinates with particle types
    annotations_flipped = tf.concat([coords_flipped, tf.expand_dims(particle_types, axis=1)], axis=1)

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
