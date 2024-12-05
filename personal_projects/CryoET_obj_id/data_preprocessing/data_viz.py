import numpy as np
import matplotlib.pyplot as plt
import napari
from data_preprocessing.data_preprocessor import get_dataset, load_annotations
from data_preprocessing.augmentations import random_flip_3d, random_rotate_90

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


def visualize_3d_sample(patch, annotations_with_types, particle_type_mapping):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(patch, name='Patch', colormap='gray')

        if annotations_with_types.size > 0:
            coords = annotations_with_types[:, :3]
            particle_types = annotations_with_types[:, 3].astype(int)

            # Check and log coordinate values
            print("Coords before filtering:", coords)

            # Remove invalid entries
            valid_mask = particle_types >= 0  # Assuming invalid types are < 0
            coords = coords[valid_mask]
            particle_types = particle_types[valid_mask]

            print("Coords after filtering:", coords)

            if coords.size > 0:
                # Map particle type labels to names
                label_to_particle_type = {v: k for k, v in particle_type_mapping.items()}
                particle_type_names = [label_to_particle_type[label] for label in particle_types]

                # Assign colors based on particle types
                unique_types = np.unique(particle_types)
                num_types = len(unique_types)
                colormap = plt.cm.get_cmap('hsv', num_types)
                colors = colormap(np.linspace(0, 1, num_types))

                # Map particle types to colors
                type_to_color = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
                annotation_colors = np.array([type_to_color[ptype] for ptype in particle_types])

                # Add points to Napari viewer
                viewer.add_points(
                    coords,
                    size=5,
                    symbol='o',
                    name='Annotations',
                    face_color=annotation_colors,
                    edge_color='black',
                    properties={'particle_type': particle_type_names},
                    text={'string': '{particle_type}', 'size': 10, 'color': 'white'}
                )
            else:
                print("No valid annotations to display.")


for batch in train_dataset.take(1):
    patches, heatmaps, annotations_batch = batch
    for i in range(patches.shape[0]):
        patch = patches[i].numpy().squeeze()
        annotations_with_types = annotations_batch[i].numpy()

        visualize_3d_sample(patch, annotations_with_types, particle_type_mapping)
