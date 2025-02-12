import pickle
import matplotlib.pyplot as plt
import argparse

def load_history(file_path):
    """
    Load a pickle file and return its contents.

    Parameters:
        file_path (str): Path to the .pkl file.

    Returns:
        data (dict): Data loaded from the pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def plot_history(data):
    """
    Plot training and validation loss and accuracy from history data.

    Parameters:
        data (dict): Dictionary containing history data.
    """
    if not isinstance(data, dict):
        print("Invalid data format. Expected a dictionary.")
        return

    if 'loss' in data and 'val_loss' in data:
        plt.figure()
        plt.plot(data['loss'], label='Training Loss')
        plt.plot(data['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

    if 'accuracy' in data and 'val_accuracy' in data:
        plt.figure()
        plt.plot(data['accuracy'], label='Training Accuracy')
        plt.plot(data['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="View and visualize training history from a .pkl file.")
    parser.add_argument('file_path', type=str, help="Path to the .pkl file containing the history data.")
    args = parser.parse_args()

    # Load the history data
    history = load_history(args.file_path)
    if history is None:
        return

    # Display the data type and keys
    print(f"Data type: {type(history)}")
    if isinstance(history, dict):
        print(f"Keys: {list(history.keys())}")

    # Plot the history
    plot_history(history)

if __name__ == "__main__":
    main()
