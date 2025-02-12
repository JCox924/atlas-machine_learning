import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing.data_preprocessor import get_dataset
from data_preprocessing.utils import load_annotations


def load_test_dataset(test_tomogram_paths, patch_size, batch_size, voxel_spacing, particle_type_mapping, num_classes=6):
    """
    Load and preprocess the test dataset.

    Args:
        test_tomogram_paths (list): List of paths to test tomogram data.
        patch_size (tuple): Size of each patch (depth, height, width).
        batch_size (int): Batch size for testing.
        voxel_spacing (list): Voxel spacing for annotations.
        particle_type_mapping (dict): Mapping of particle types to class indices.
        num_classes (int): Number of particle types/classes.

    Returns:
        tf.data.Dataset: Test dataset ready for evaluation.
    """
    return get_dataset(
        tomogram_paths=test_tomogram_paths,
        patch_size=patch_size,
        batch_size=batch_size,
        voxel_spacing=voxel_spacing,
        augmentations=None,  # No augmentations for testing
        shuffle_buffer_size=0,  # No shuffling for testing
        particle_type_mapping=particle_type_mapping,
        num_classes=num_classes,
    )


def evaluate_model(model_path, test_dataset):
    """
    Evaluate a model on the test dataset.

    Args:
        model_path (str): Path to the saved model (.h5 file).
        test_dataset (tf.data.Dataset): Test dataset.

    Returns:
        dict: Evaluation results (loss and accuracy).
    """
    print(f"Evaluating model: {model_path}")
    model = load_model(model_path)
    results = model.evaluate(test_dataset, verbose=1)
    print(f"Results: Loss = {results[0]}, Accuracy = {results[1]}")
    return results


def main():
    # Paths and parameters
    model_dir = r"C:\Users\Joshua\PycharmProjects\Machine_Learning_master\atlas-machine_learning\personal_projects\CryoET_obj_id\models"  # Directory containing saved models
    test_tomogram_paths = [
        "test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr",
        # Add more test tomogram paths as needed
    ]
    patch_size = (128, 128, 128)
    batch_size = 1
    voxel_spacing = [10.0, 10.0, 10.0]

    # Corrected train tomogram path
    train_tomogram_path = (
        "C:/Users/Joshua/PycharmProjects/Machine_Learning_master/atlas-machine_learning/"
        "personal_projects/CryoET_obj_id/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr"
    )

    annotations, particle_type_mapping = load_annotations(train_tomogram_path, voxel_spacing)

    # Load the test dataset
    test_dataset = load_test_dataset(
        test_tomogram_paths,
        patch_size,
        batch_size,
        voxel_spacing,
        particle_type_mapping,
    )

    # Search for .h5 files recursively
    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".h5"):
                model_files.append(os.path.join(root, file))

    if not model_files:
        print(f"No models found in the directory: {model_dir}")
        return

    # Evaluate all models found
    for model_file in model_files:
        print(f"Evaluating model at {model_file}")
        results = evaluate_model(model_file, test_dataset)
        print(f"Model: {model_file}, Loss: {results[0]}, Accuracy: {results[1]}")


if __name__ == "__main__":
    main()
