# CryoET OD
### Project Overview
CryoET OD is a project dedicated to performing three-dimensional (3D) object detection on Cryo Electron-Tomography (Cryo-ET) data. The goal is to combine machine learning techniques (specifically with TensorFlow) and domain-specific knowledge in Cryo-ET to detect and classify macromolecular structures in tomograms. This repository hosts the code, documentation, and resources to replicate and extend the approach.

## Table of Contents
1. Features
2. Installation
3. Usage
4. Data
5. References
6. Contributing
7. License

## Features

3D Object Detection: Customized object detection pipelines adapted for volumetric data.
Modular Codebase: Clear separation of data loading, model definition, and training scripts.
TensorFlow Integration: Implements the latest TensorFlow (see TensorFlow Docs) for high-performance deep learning.
Extensible Framework: Easily incorporate new architectures or optimization routines.
Installation
Follow these steps to install and set up the environment.

### Clone this repository:

```bash
git clone https://github.com/JCox924/atlas-machine_learning/personal_projects/CryoET_obj_id.git
cd CryoET_obj_id
```
### Create and activate a Python 3.10 environment (using conda):

```bash
conda create --name cryoet-od python=3.10 -y
conda activate cryoet-od
```
### Install dependencies:

```bash
pip install -r requirements.txt
```
### Verify installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Ensure that TensorFlow is installed correctly and that the version matches project requirements.

## Usage

### Data Preparation:

Organize your tomograms in a designated data/ folder.
Update the data paths in config.yaml or similar configuration files.
Training:

```bash
python train.py --config config.yaml
```
This command initiates the training loop for 3D object detection.
Adjust hyperparameters (learning rate, batch size) in the configuration file.
### Evaluation:

```bash
python evaluate.py --model checkpoints/best_model.h5 --data data/val
```
This evaluates the model performance on validation or test sets.
View metrics such as precision, recall, and 3D Intersection over Union (IoU).

### Inference:

```bash
python infer.py --model checkpoints/final_model.h5 --data data/test
```
Generates predictions (bounding boxes or masks) for 3D structures in tomograms.

---
## Data
* Input: Cryo-ET tomograms in a volume format (e.g., MRC, EM, or HDF5).
* Annotations: Ground-truth bounding boxes (or segmentation masks) stored in JSON or CSV.
* Output:
  * Trained model weights stored in checkpoints/.
  * Visualizations and logs in logs/.

## References

1. Cryo Electron-Tomography
   * Zhang, K., Li, S., and Li, H. (2019). Cryo-EM and cryo-ET data analysis. Biophys. Rep., 5, 123–136. doi:10.1007/s41048-019-0061-8
2. 3D Object Detection
* Chen, X., Ma, H., Wan, J., Li, B. and Xia, T. (2017). Multi-View 3D Object Detection Network for Autonomous Driving. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. TensorFlow
   * [TensorFlow Guide](https://www.tensorflow.org/guide)

## Contributing

**Branching**: Use feature branches and create pull requests against main.

**Coding Conventions**: Follow Python best practices (PEP 8).

**Testing**: Add or update unit tests in tests/ for each new feature or bug fix.

**Documentation**: Update docstrings and comments to keep the repository well-documented.

## License
Distributed under the MIT License. See LICENSE for more information.
