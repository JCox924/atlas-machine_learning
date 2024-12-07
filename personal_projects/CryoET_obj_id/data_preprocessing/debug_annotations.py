import numpy as np
import napari

from utils import load_annotations  # Adjust the import path as needed

# Define the tomogram path and related parameters
tomogram_path = '../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/wbp.zarr'
voxel_spacing = [10.0, 10.0, 10.0]  # Physical spacing of the tomogram
patch_size = (128, 128, 128)  # Size of a patch (for later use)
batch_size = 2  # For training
tomogram_shape = (256, 256, 256)  # Replace with the true tomogram shape if known


def inspect_annotations(annotations, tomogram_shape, voxel_spacing):
    """
    Inspects annotations for validity and alignment with the tomogram.
    """
    print("\n--- Annotations Inspection ---")
    print(f"Number of annotations: {len(annotations)}")

    if annotations.size == 0:
        print("No annotations found. Exiting.")
        return

    print(f"Shape of annotations: {annotations.shape}")
    print(f"Annotation data types: {annotations.dtype}")
    print(f"First few annotations:\n{annotations[:5]}")

    # Split coordinates and labels
    coordinates = annotations[:, :3]
    labels = annotations[:, 3]
    print(f"Unique labels: {np.unique(labels)}")

    if voxel_spacing is not None:
        print(f"Voxel spacing: {voxel_spacing}")

    # Check for out-of-bounds coordinates
    out_of_bounds = np.any((coordinates < 0) | (coordinates >= tomogram_shape), axis=1)
    if np.any(out_of_bounds):
        print("WARNING: Some annotations are out of tomogram bounds!")
        print(f"Out-of-bounds annotations:\n{annotations[out_of_bounds]}")

    print("Inspection complete.")


def visualize_annotations(tomogram_shape, annotations):
    """
    Visualizes the annotations in Napari.
    """
    print("\n--- Launching Napari Viewer ---")

    # Create a blank tomogram volume for visualization
    tomogram_volume = np.zeros(tomogram_shape, dtype=np.float32)

    # Add annotation points to the volume
    for annotation in annotations:
        z, y, x, label = annotation
        z, y, x = map(int, [z, y, x])  # Convert to integer indices
        if 0 <= z < tomogram_shape[0] and 0 <= y < tomogram_shape[1] and 0 <= x < tomogram_shape[2]:
            tomogram_volume[z, y, x] = label  # Assign label value at the annotation location

    # Use Napari to visualize the tomogram and annotations
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(tomogram_volume, name="Tomogram Volume", scale=voxel_spacing)
        print("Napari visualization launched.")


# Load annotations
annotations, particle_type_mapping = load_annotations(tomogram_path, voxel_spacing=voxel_spacing)

# Inspect annotations
inspect_annotations(annotations, tomogram_shape, voxel_spacing)

# Visualize annotations in Napari
visualize_annotations(tomogram_shape, annotations)