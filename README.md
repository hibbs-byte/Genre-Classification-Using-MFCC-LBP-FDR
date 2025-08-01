
# Genre Classification using MFCC, LBP, and FDR

This repository presents a machine learning approach to classifying music genres using a combination of spectral and image-based audio features. The method involves extracting MFCC (Mel-Frequency Cepstral Coefficients) features from audio signals, applying Local Binary Patterns (LBP) on spectrogram-like images, and selecting the most relevant features using Fisher Discriminant Ratio (FDR). The final classification is performed using standard supervised learning models.

## Project Overview

The objective of this project is to distinguish between music genres—specifically, rock and jazz—based on audio signal processing techniques. The pipeline combines signal analysis, image-based feature extraction, and statistical feature selection to enhance classification performance.

## Methodology

1. **MFCC Extraction**: Audio signals are processed to extract MFCCs, which are widely used for representing timbral characteristics of music.
2. **LBP Feature Computation**: MFCC feature matrices are treated as grayscale images, and Local Binary Pattern (LBP) is applied to capture texture-like features.
3. **FDR-Based Feature Selection**: Fisher Discriminant Ratio is computed to assess and rank feature importance. Top features are selected to reduce dimensionality and improve model performance.
4. **Model Training**: Supervised learning algorithms (such as SVM, Logistic Regression) are used to classify the data based on the selected features.

## Dataset

The dataset consists of two classes:
- Jazz audio samples
- Rock audio samples

Each audio sample was preprocessed into MFCC matrices and saved in CSV format before applying LBP and FDR.

## Directory Structure

```
MFCC_LBP_FDR.ipynb       # Jupyter notebook with all steps from loading data to classification
rock_lbp2.csv            # LBP features for rock genre (example path)
jazz_lbp2.csv            # LBP features for jazz genre (example path)
```

> Note: Dataset files are not included due to size. Paths should be updated to match local or Google Drive locations.

## Libraries Used

The following Python libraries were used in this project:

- `numpy`: Numerical computations
- `pandas`: Data handling and manipulation
- `matplotlib`: Data visualization
- `sklearn` (`scikit-learn`): Machine learning tools for preprocessing, feature selection, and classification
- `cv2` (`OpenCV`): Image processing (LBP computation)
- `os`: File and path handling
- `warnings`: Suppress warning messages for clean output

To install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/yourusername/Genre-Classification-Using-MFCC-LBP-FDR.git
cd Genre-Classification-Using-MFCC-LBP-FDR
```

2. Open the notebook:

- Using Jupyter Notebook:
  ```bash
  jupyter notebook MFCC_LBP_FDR.ipynb
  ```
- Or upload to [Google Colab](https://colab.research.google.com/) for cloud execution.

3. Upload your CSV feature files to the expected paths in the notebook.

4. Run all cells to reproduce the full pipeline and results.

## Results

The classification models achieved promising accuracy, particularly when using FDR for feature selection. The best-performing model showed strong separation between jazz and rock samples, indicating that the combination of MFCC + LBP with FDR is effective for genre classification.

## Potential Improvements

- Use a larger dataset with more diverse genres.
- Explore deep learning models such as CNNs on spectrograms.
- Integrate end-to-end audio preprocessing within the notebook (e.g., using `librosa`).
- Automate hyperparameter tuning and cross-validation.

## Author

This project was developed by [Your Name].  
For inquiries or collaborations, please connect via [LinkedIn](https://www.linkedin.com/in/YOURPROFILE) or email at your.email@example.com.
