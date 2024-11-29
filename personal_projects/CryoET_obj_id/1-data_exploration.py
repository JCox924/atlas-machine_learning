import os
import json
import numpy as np
import zarr
import napari
from skimage import exposure
import matplotlib.pyplot as plt


def load_annotations(tomogram_path, voxel_spacing):
    """
    Automatically loads and normalizes annotations based on the tomogram path.
    :param tomogram_path: Path to the Zarr file of the tomogram.
    :param voxel_spacing: Physical spacing (e.g., angstroms) per voxel for each dimension.
    :return: Numpy array of normalized particle annotations.
    """
    experiment = tomogram_path.split("/")[-3]  # Extract experiment name
    base_dir = "/".join(tomogram_path.split("/")[:-5])  # Go up to the base directory
    annotation_dir = os.path.join(base_dir, "overlay", "ExperimentRuns", experiment, "Picks")

    print(f"Looking for annotations in: {annotation_dir}")

    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory {annotation_dir} does not exist.")

    annotations = []
    for json_file in os.listdir(annotation_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(annotation_dir, json_file), 'r') as f:
                data = json.load(f)
                print(f"Loaded raw data from {json_file}: type={type(data)}, sample={str(data)[:500]}")

                # Extract and normalize annotation coordinates
                points = data.get('points', [])
                for point in points:
                    if 'location' in point:
                        loc = point['location']
                        normalized = [
                            loc['x'] / voxel_spacing[0],
                            loc['y'] / voxel_spacing[1],
                            loc['z'] / voxel_spacing[2]
                        ]
                        annotations.append(normalized)

    points_array = np.array(annotations)
    print(f"Extracted {len(points_array)} normalized annotations.")
    return points_array


def see_mid(path):
    """
    Visualizes the middle slice of the tomogram with annotations overlaid.
    """
    zarr_data = zarr.open(path, mode='r')[0]
    print("Tomogram shape:", zarr_data.shape)

    # Determine voxel spacing (example values; replace with actual data)
    voxel_spacing = [10.0, 10.0, 10.0]  # Replace with metadata or documentation values

    # Load annotations and normalize
    annotations = load_annotations(path, voxel_spacing)

    # Select the middle slice and filter annotations
    middle_index = zarr_data.shape[0] // 2
    middle_slice = zarr_data[middle_index]
    annotations_in_slice = annotations[np.abs(annotations[:, 2] - middle_index) < 1]  # Adjust tolerance if needed

    plt.imshow(middle_slice, cmap='gray')
    plt.scatter(annotations_in_slice[:, 0], annotations_in_slice[:, 1], c='red', s=10, label='Annotations')
    plt.title("Middle Slice of Tomogram with Annotations")
    plt.legend()
    plt.show()


def show_3D(path):
    """
    Opens the 3D tomogram in the Napari viewer with annotations overlaid.
    :param path: Path to the Zarr file of the tomogram.
    """
    zarr_data = zarr.open(path, mode='r')[0]
    zarr_normalized = exposure.rescale_intensity(zarr_data, in_range='image', out_range=(0, 1))

    # Define voxel spacing (adjust if the actual values are known or need to be inferred)
    voxel_spacing = [10.0, 10.0, 10.0]  # Replace with actual values if available

    # Load annotations with the given voxel spacing
    annotations = load_annotations(path, voxel_spacing)

    # Visualize in Napari
    viewer = napari.Viewer()
    viewer.add_image(zarr_normalized, name="Tomogram")
    viewer.add_points(
        annotations,
        name="Annotations",
        size=10,
        face_color="red",
        border_color="black"  # Updated for Napari compatibility
    )

    napari.run()


path = "train/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/denoised.zarr"

see_mid(path)
show_3D(path)
