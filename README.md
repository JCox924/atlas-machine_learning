# Atlas Machine Learning

A curated collection of mathematical foundations, tutorial scripts, and hands‑on machine learning implementations. This repository is organized into:

- **math/**: Core mathematical concepts and visualizations
- **models/**: Pre-trained model weights for transfer learning and fine-tuning
- **personal_projects/**: Research‑oriented code for specialized projects
- **supervised_learning/**: Classification, regression, deep learning, and NLP pipelines
- **unsupervised_learning/**: Clustering, dimensionality reduction, autoencoders, and HMMs

## Repository Structure

```text
atlas-machine_learning/
├── math/                          # Mathematical modules and plotting utilities
│   ├── advanced_linear_algebra/   # Eigenvalue/vector operations, SVD, etc.
│   ├── bayesian_prob/             # Bayesian inference examples
│   ├── calculus/                  # Differentiation and integration demos
│   ├── convolutions_and_pooling/  # Convolution math and pooling illustrations
│   ├── imgs/                      # Reference figures for math tutorials
│   ├── linear_algebra/            # Vector spaces, matrix ops
│   ├── multivariate_prob/         # Joint, marginal, and conditional distributions
│   ├── plotting/                  # Utilities to render graphs and surfaces
│   └── probability/               # Discrete and continuous probability tutorials
├── models/                        # Pre-trained model weights
│   ├── bert/                      # BERT pre-trained weights and configs
│   ├── resnet/                    # ResNet pre-trained weights and configs
│   └── word2vec/                  # Word2Vec pre-trained embeddings
├── personal_projects/             # In‑depth, project‑specific codebases
│   ├── CryoET_obj_id/             # CryoET particle identification pipeline
│   │   ├── data_preprocessing/    # Data cleaning and augmentation scripts
│   │   └── models/                # 3D CNN architectures and training loops
│   └── alt_methods/               # Alternative ML methodologies under exploration
├── supervised_learning/           # Supervised algorithms and networks
│   ├── RNNs/                      # Custom recurrent cell implementations & training
│   ├── classification/            # Classical & neural classifiers
│   │   ├── data/                  # Example datasets for classification
│   │   └── tests/                 # Unit tests for model validation
│   ├── cnn/                       # Convolutional neural network examples
│   ├── deep_cnns/                 # Advanced CNN architectures (ResNet, DenseNet, etc.)
│   ├── error_analysis/            # Tools for bias/variance diagnostics
│   ├── keras/                     # Keras‑based demos and tutorials
│   ├── nlp_metrics/               # BLEU, ROUGE, and other NLP evaluation scripts
│   ├── object_detection/          # YOLO, SSD, and custom detector code
│   │   └── imgs/                  # Sample images for detection tests
│   ├── optimization/              # Learning rate schedules, Adam, RMSProp, etc.
│   ├── qa_bot/                    # End‑to‑end question‑answering chatbot
│   ├── regularization/            # L1/L2, dropout, batch normalization examples
│   ├── tensorflow/                # TensorFlow-specific utilities and examples
│   ├── transfer_learning/         # Fine‑tuning pre‑trained models
│   ├── transformer_apps/          # Transformer‑based architectures (BERT, GPT, etc.)
│   └── word_embeddings/           # Word2Vec, GloVe, and embedding visualizations
└── unsupervised_learning/         # Unsupervised methods and latent models
    ├── autoencoders/             # Vanilla, variational, and sparse autoencoders
    ├── clustering/               # K‑means, hierarchical, DBSCAN implementations
    ├── dimensionality_reduction/ # PCA, t‑SNE, UMAP examples
    ├── hmm/                      # Hidden Markov Model training & inference
    └── hyperparameter_tuning/    # Grid search, random search, and Bayesian optimization
```

## Getting Started

### Prerequisites
- **Python** 3.9 or higher
- **pip** for package management
- (Recommended) Ubuntu 20.04 LTS or equivalent Unix environment

### Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/<your-username>/atlas-machine_learning.git
   cd atlas-machine_learning
   ```
2. **Install** dependencies (modify as needed for your environment):
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

- **Run a classification demo**:
  ```bash
  python supervised_learning/classification/train_model.py --data supervised_learning/classification/data/sample.csv
  ```

- **Visualize a convolution operation**:
  ```bash
  python math/convolutions_and_pooling/demo_convolution.py
  ```

- **Train a custom LSTM cell**:
  ```bash
  python supervised_learning/RNNs/train_rnn.py --config config/lstm.yaml
  ```

- **Launch the QA chatbot**:
  ```bash
  python supervised_learning/qa_bot/app.py
  ```

## Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/<feature-name>`)
3. Commit your changes (`git commit -m 'Add <feature>'`)
4. Push the branch (`git push origin feature/<feature-name>`)
5. Open a pull request

Ensure all new code follows existing style, includes tests, and has clear documentation.
|
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Developed by Joshua Cox*

[Linkedin](https://www.linkedin.com/in/joshuacox918/)