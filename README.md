# BML-Deep-Kalman-filter-prediction

Code for the article **"Indoor thermal comfort management: A Bayesian machine-learning approach to data denoising and dynamics prediction of HVAC systems"**

## Code Versions

### TF1.x (Original version)
This is the original version of the code with the results presented in the article. A substantial part of the neural network was taken from [Illya Luznikov's](https://github.com/LuchnikovI/Deep-Kalman-filter-for-climate-control) repo.

**Contents:**
- Single thermal zone simulation including:
  - Different variations for fault detection
  - Demand response 
  - Random walk version
  - Configurable initial parameters, simulation length, noise level
  - Option for static or realistic weather simulation
- Dataset generation, saving and loading
- Custom NN structure with:
  - Training function
  - Inference functions
- Testing functions:
  - Single test and batch test functions
  - Scoring
  - Functions for testing DR, fault detection and fidelity gap method

**Implementation Notes:**
- Developed in Google Colab using dataset from this repo
- Not optimized for computer resources (most functions used only a few times)
- T4 GPU in Colab was typically under 50% utilization
- Compatible up to TF 2.17 (newer CUDA versions in TF ≥2.18 has compatibility issues)
- CUDNN GRU has better performance than current Keras GRU but only runs on GPU
- Includes all models described in the article

### TF2.x (Migrated version)
This is a migrated version of the original architecture that can run inference on CPU, with results comparable to the published article (different balance: better RMSE but worse R<sup>2</sup>).

**Key Differences:**
- More optimal computer resource usage (more object-oriented, fewer cycles)
- Currently includes only:
  - Modeling functions
  - Data generation, saving and loading
  - Migrated NN with custom training/inference functions
  - Custom testing and scoring functions
  - Only includes pretrained general model (fine-tuned models not available)

**Dataset Information:**
- Published dataset adapted for this version
- Split into multiple files (due to GitHub's 100MB file limit)
- Includes splitting and stitching function to reassemble dataset
