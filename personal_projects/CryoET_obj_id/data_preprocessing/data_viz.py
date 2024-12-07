import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing.data_preprocessor import get_dataset
from data_preprocessing.augmentations import random_flip_3d, random_rotate_90
from utils import load_annotations

# Define paths and parameters
tomogram_path = '../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/wbp.zarr'
train_tomogram_paths = [tomogram_path]
patch_size = (128, 128, 128)
batch_size = 2
voxel_spacing = [10.0, 10.0, 10.0]
augmentations = []

# Load annotations and particle_type_mapping
annotations, particle_type_mapping = load_annotations(tomogram_path, voxel_spacing)

# Create the dataset
train_dataset = get_dataset(
    tomogram_paths=train_tomogram_paths,
    patch_size=patch_size,
    batch_size=batch_size,
    voxel_spacing=voxel_spacing,
    augmentations=augmentations,
    shuffle_buffer_size=100,
    particle_type_mapping=particle_type_mapping
)

def calculate_centroids(label_volume, annotations):
    """
    Calculate centroids for each unique label in the dense label volume
    while retaining labels from the annotations.
    """
    from scipy.ndimage import center_of_mass
    centroids = []
    labels = np.unique(label_volume)
    for label in labels:
        if label == 0:  # Skip background
            continue
        # Filter annotations by the current label
        label_annotations = annotations[annotations[:, 3] == label]
        if label_annotations.size == 0:
            continue
        # Compute centroid from filtered annotations
        centroid_coords = center_of_mass(label_volume == label)
        centroids.append((*centroid_coords, label))
    return np.array(centroids)

def visualize_all_annotations(patch, annotations, particle_type_mapping):
    """
    Visualize all loaded annotations in Napari alongside a tomogram patch.
    """
    import napari

    # Debug: Ensure annotations are available
    if annotations.size == 0:
        print("No annotations to display.")
        return

    # Extract coordinates and labels
    coords = annotations[:, :3]
    particle_types = annotations[:, 3].astype(int)

    # Map particle type labels to names
    label_to_particle_type = {v: k for k, v in particle_type_mapping.items()}
    particle_type_names = [label_to_particle_type[label] for label in particle_types]

    # Assign colors based on particle types
    unique_types = np.unique(particle_types)
    num_types = len(unique_types)
    colormap = plt.get_cmap('hsv')  # Get the colormap
    colors = colormap(np.linspace(0, 1, num_types))

    # Map particle types to colors
    type_to_color = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
    annotation_colors = np.array([type_to_color[ptype] for ptype in particle_types])

    # Create Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(patch, name='Patch', colormap='gray')

    # Add annotations as points
    viewer.add_points(
        coords,
        size=5,  # Adjust size as needed
        symbol='o',
        name='Annotations',
        face_color=annotation_colors,
        border_color='black',
        properties={'particle_type': particle_type_names},
        text={'string': '{particle_type}', 'size': 10, 'color': 'white'}
    )

    napari.run()


# Process and visualize dataset samples
for batch in train_dataset.take(1):
    patches, one_hot_labels = batch
    for i in range(patches.shape[0]):
        patch = patches[i].numpy().squeeze()

        # Visualize all annotations loaded from `load_annotations`
        visualize_all_annotations(patch, annotations, particle_type_mapping)
