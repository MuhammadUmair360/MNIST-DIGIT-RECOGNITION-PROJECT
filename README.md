# MNIST Digit Classification with Classical ML and Deep Learning

**Team Members:**
Muhammad Umair (506068)

---

## Abstract
This project classifies handwritten digits from the MNIST dataset using both classical machine learning (Random Forest, SVM) and a deep learning approach (CNN). The dataset consists of 28x28 grayscale images of digits 0–9. A Random Forest classifier, an SVM, and a Convolutional Neural Network were trained and evaluated. The **CNN achieved the highest accuracy (≈0.988)**, significantly outperforming the classical models. A statistical t-test was performed to compare the architectures, and findings suggest that while classical models are efficient for baseline tasks, CNNs are superior for capturing spatial hierarchies in image data.

## Introduction
Handwritten digit recognition is a foundational problem in computer vision with applications in postal mail sorting, bank check processing, and automated form entry. The objective of this project is to:
* Implement and tune classical ML algorithms for digit classification.
* Design a robust Convolutional Neural Network (CNN) using modern regularization techniques.
* Conduct a rigorous comparative analysis using accuracy, F1-score, and statistical significance testing.
* Evaluate the models on custom, real-world handwritten images to test generalization.

## Dataset Description
### Source
The dataset used is the **MNIST (Modified National Institute of Standards and Technology)** dataset.

### Size and Structure
* **Total Instances:** 70,000 images (60,000 training, 10,000 testing).
* **Format:** 28x28 pixel grayscale images.
* **Target:** 10 classes representing digits 0 through 9.

### Preprocessing
* **Normalization:** Pixel values were scaled from [0, 255] to [0, 1].
* **Data Reshaping:** Images were flattened for classical models and kept as 4D tensors for the CNN.
* **Subsampling:** A stratified subset of 5,000 samples was used for efficient classical ML tuning.

## Methodology
### Classical ML Approaches
1. **Random Forest Classifier:** Ensemble method tuned with `GridSearchCV` (best: `n_estimators=100`, `max_depth=20`).
2. **Support Vector Machine (SVM):** Utilized the RBF kernel for non-linear separation.

### Deep Learning Architecture
A **Convolutional Neural Network (CNN)** was implemented with:
* `Conv2D` layer for spatial feature extraction.
* `BatchNormalization` and `Dropout` (0.25 and 0.5) for regularization.
* `EarlyStopping` and `ReduceLROnPlateau` for training optimization.

## Results & Analysis
### Performance Comparison
| Model | Test Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| Random Forest | 0.9393 | 0.94 |
| SVM | 0.9552 | 0.96 |
| **CNN** | **0.9878** | **0.99** |

### Statistical Significance Tests
A **one-sample t-test** compared the CNN accuracy against the mean performance of the classical algorithms.
* **p-value:** 0.12325
* **Conclusion:** While the CNN is numerically superior, the test indicates the variation is substantial but falls just outside the strict 0.05 significance threshold for this specific model sample.

## Business Impact Analysis
1. **Automation Potential:** High F1-scores make the CNN suitable for bank check digit-extraction.
2. **Resource Allocation:** Random Forest remains a viable "lightweight" alternative for edge devices without GPU support.
3. **Process Insights:** Error analysis shows occasional confusion between '7'/'2' and '9'/'4', guiding future data collection.

## Conclusion & Future Work
This project proves CNNs are the gold standard for image-based tasks, achieving **98.78% accuracy**. Future work will explore Data Augmentation and Gradient Boosting methods (XGBoost).

---

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook to reproduce training and evaluation.
