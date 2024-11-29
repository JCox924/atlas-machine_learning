import os
import json
import numpy as np
import zarr
import napari
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba


def load_annotations(tomogram_path, voxel_spacing):
    """
    Loads, normalizes, and reorders annotations by particle type.
    :param tomogram_path: Path to the Zarr file of the tomogram.
    :param voxel_spacing: Physical spacing per voxel for each dimension.
    :return: Numpy array of normalized particle annotations and their types.
    """
    experiment = tomogram_path.split("/")[-3]
    base_dir = "/".join(tomogram_path.split("/")[:-5])
    annotation_dir = os.path.join(base_dir, "overlay", "ExperimentRuns", experiment, "Picks")

    print(f"Looking for annotations in: {annotation_dir}")

    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory {annotation_dir} does not exist.")

    annotations = []
    particle_types = []
    for json_file in os.listdir(annotation_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(annotation_dir, json_file), 'r') as f:
                data = json.load(f)
                print(f"Loaded raw data from {json_file}: {data.get('pickable_object_name')}")

                particle_type = data.get("pickable_object_name", "Unknown")
                points = data.get("points", [])
                for point in points:
                    if 'location' in point:
                        loc = point['location']
                        normalized = [
                            loc['x'] / voxel_spacing[0],
                            loc['y'] / voxel_spacing[1],
                            loc['z'] / voxel_spacing[2]
                        ]
                        # Reorder to zyx
                        annotations.append([normalized[2], normalized[1], normalized[0]])
                        particle_types.append(particle_type)

    points_array = np.array(annotations)
    print(f"Extracted {len(points_array)} annotations.")
    return points_array, particle_types


def see_mid(path):
    """
    Visualizes the middle slice of the tomogram with color-coded annotations by particle type.
    """
    zarr_data = zarr.open(path, mode='r')[0]
    print("Tomogram shape:", zarr_data.shape)

    voxel_spacing = [10.0, 10.0, 10.0]  # Replace with actual values if known
    annotations, particle_types = load_annotations(path, voxel_spacing)

    middle_index = zarr_data.shape[0] // 2
    middle_slice = zarr_data[middle_index]

    plt.imshow(middle_slice, cmap='gray')

    # Generate unique colors for each particle type
    unique_types = list(set(particle_types))
    colors = dict(zip(unique_types, mcolors.TABLEAU_COLORS))

    for p_type in unique_types:
        indices = [i for i, t in enumerate(particle_types) if t == p_type]
        points = annotations[indices]
        points_in_slice = points[np.abs(points[:, 0] - middle_index) < 1]  # Filter along z-axis
        plt.scatter(points_in_slice[:, 2], points_in_slice[:, 1], label=p_type, c=colors[p_type], s=10)

    plt.title("Middle Slice of Tomogram with Annotations")
    plt.legend()
    plt.show()



def show_3D(path):
    """
    Opens the 3D tomogram in the Napari viewer with color-coded annotations by particle type.
    """
    zarr_data = zarr.open(path, mode='r')[0]
    zarr_normalized = exposure.rescale_intensity(zarr_data, in_range='image', out_range=(0, 1))

    voxel_spacing = [10.0, 10.0, 10.0]  # Replace with actual values if known
    annotations, particle_types = load_annotations(path, voxel_spacing)

    viewer = napari.Viewer()
    viewer.add_image(zarr_normalized, name="Tomogram")

    unique_types = list(set(particle_types))
    colors = {p_type: to_rgba(color)[:3] for p_type, color in zip(unique_types, mcolors.TABLEAU_COLORS.values())}

    point_colors = [colors[t] for t in particle_types]

    viewer.add_points(
        annotations,
        name="Annotations",
        size=5,
        face_color=point_colors,  # Array of colors for each point
        border_color="black"
    )

    napari.run()


from matplotlib.colors import to_hex

def show_3D_with_legend(path):
    """
    Opens the 3D tomogram in the Napari viewer with color-coded annotations by particle type,
    and displays a separate matplotlib legend for the colors.
    """
    zarr_data = zarr.open(path, mode='r')[0]
    zarr_normalized = exposure.rescale_intensity(zarr_data, in_range='image', out_range=(0, 1))

    voxel_spacing = [10.0, 10.0, 10.0]  # Replace with actual values if known
    annotations, particle_types = load_annotations(path, voxel_spacing)

    viewer = napari.Viewer()
    viewer.add_image(zarr_normalized, name="Tomogram")

    unique_types = list(set(particle_types))
    colors = {p_type: to_hex(to_rgba(color)[:3]) for p_type, color in zip(unique_types, mcolors.TABLEAU_COLORS.values())}
    print("Colors dictionary:", colors)  # Debugging output

    for p_type in unique_types:
        indices = [i for i, t in enumerate(particle_types) if t == p_type]
        points = annotations[indices]
        viewer.add_points(
            points,
            name=p_type,
            size=5,
            face_color=colors[p_type],  # Assign hex string color to each layer
            border_color="black"
        )

    legend_patches = [Patch(color=color, label=p_type) for p_type, color in colors.items()]
    plt.figure(figsize=(4, 4))
    plt.legend(handles=legend_patches, title="Particle Types", loc="center")
    plt.axis('off')
    plt.show()

    napari.run()




path = "train/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/denoised.zarr"

see_mid(path)
# show_3D(path)
show_3D_with_legend(path)
