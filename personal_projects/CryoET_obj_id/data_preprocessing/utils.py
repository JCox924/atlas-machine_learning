import tensorflow as tf
import numpy as np
import json
from scipy.ndimage import gaussian_filter
from pathlib import Path


def load_annotations(tomogram_path, voxel_spacing, particle_type_mapping=None):
    """
    Loads, normalizes, and reorders annotations by particle type.
    :param tomogram_path: Path to the Zarr file of the tomogram.
    :param voxel_spacing: Physical spacing per voxel for each dimension.
    :return: Numpy array of normalized particle annotations and their types.
    """
    tomogram_path = Path(tomogram_path).resolve()
    print(f"Tomogram path: {tomogram_path}")

    # Extract parts of the path
    parts = tomogram_path.parts

    # Identify experiment name from 'ExperimentRuns'
    try:
        exp_runs_index = parts.index('ExperimentRuns')
        experiment_name = parts[exp_runs_index + 1]
    except ValueError:
        raise ValueError("'ExperimentRuns' not found in the tomogram_path.")

    print(f"Experiment name: {experiment_name}")

    # Determine base path (train/test directory)
    train_test_index = next((i for i in range(len(parts)) if parts[i] in ['train', 'test']), None)
    if train_test_index is None:
        raise ValueError("'train' or 'test' not found in the tomogram_path.")
    base_path = Path(*parts[:train_test_index + 1])

    print(f"Base path: {base_path}")

    # Construct annotation directory
    annotation_dir = base_path / 'overlay' / 'ExperimentRuns' / experiment_name / 'Picks'
    print(f"Annotation directory: {annotation_dir}")

    # Initialize particle type mapping if not provided
    if particle_type_mapping is None:
        particle_type_mapping = {}

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory {annotation_dir} does not exist.")

    # Process annotation files
    annotations = []
    current_label = len(particle_type_mapping)

    for json_path in annotation_dir.glob('*.json'):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {json_path}. Skipping.")
            continue

        particle_type = data.get('pickable_object_name')
        if not particle_type:
            print(f"Warning: Missing 'pickable_object_name' in {json_path}. Skipping.")
            continue

        points = data.get('points', [])
        if not points:
            print(f"Warning: No points found in {json_path}. Skipping.")
            continue

        if particle_type not in particle_type_mapping:
            particle_type_mapping[particle_type] = current_label
            current_label += 1

        label = particle_type_mapping[particle_type]

        for point in points:
            location = point.get('location', {})
            x, y, z = location.get('x'), location.get('y'), location.get('z')
            if x is None or y is None or z is None:
                print(f"Warning: Incomplete location data in {json_path}. Skipping.")
                continue

            coord = np.array([x, y, z], dtype=np.float32)
            voxel_coord = coord / np.array(voxel_spacing, dtype=np.float32) if voxel_spacing else coord
            annotations.append(np.append(voxel_coord, label))

    # Convert to numpy array
    annotations = np.array(annotations, dtype=np.float32) if annotations else np.empty((0, 4), dtype=np.float32)

    return annotations, particle_type_mapping


def generate_heatmap(shape, centers, sigma=3):
    heatmap = np.zeros(shape, dtype=np.float32)
    for center in centers:
        z, y, x = map(int, center)
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            heatmap[z, y, x] = 1
    # Use Gaussian filter from scipy or TensorFlow
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()  # Normalize to [0, 1]
    return heatmap


def generate_heatmap_tf(shape, centers, sigma=3):
    shape = tf.cast(shape, tf.int32)
    heatmap = tf.zeros(shape, dtype=tf.float32)

    num_centers = tf.shape(centers)[0]

    def compute_heatmap():
        # Create coordinate grids
        z = tf.range(shape[0], dtype=tf.float32)
        y = tf.range(shape[1], dtype=tf.float32)
        x = tf.range(shape[2], dtype=tf.float32)
        zz, yy, xx = tf.meshgrid(z, y, x, indexing='ij')  # Shape: (Z, Y, X)

        zz = tf.expand_dims(zz, axis=-1)  # Shape: (Z, Y, X, 1)
        yy = tf.expand_dims(yy, axis=-1)
        xx = tf.expand_dims(xx, axis=-1)

        centers_expanded = tf.expand_dims(centers, axis=0)  # Shape: (1, N, 3)

        dz = zz - centers_expanded[..., 0]
        dy = yy - centers_expanded[..., 1]
        dx = xx - centers_expanded[..., 2]

        distance_squared = dz**2 + dy**2 + dx**2
        gaussian = tf.exp(-distance_squared / (2 * sigma**2))
        gaussian = tf.reduce_sum(gaussian, axis=-1)  # Sum over centers

        max_value = tf.reduce_max(gaussian)
        heatmap_normalized = tf.cond(
            max_value > 0,
            lambda: gaussian / max_value,
            lambda: gaussian
        )
        return heatmap_normalized

    heatmap = tf.cond(
        num_centers > 0,
        compute_heatmap,
        lambda: heatmap
    )

    return heatmap
