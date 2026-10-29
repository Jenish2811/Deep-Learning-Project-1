# Adult Income Prediction with Artificial Neural Networks

This project uses an Artificial Neural Network (ANN) to predict whether a person's annual income is **greater than $50K** using demographic and employment-related features from `adult.csv`. The notebook explores how preprocessing choices and neural-network training settings affect classification performance, especially for the `>50K` class.

## Project Files

- `Project-2.ipynb` — end-to-end notebook containing data preparation, exploratory analysis, ANN experiments, evaluation, and visualizations.
- `adult.csv` — input dataset required by the notebook.
- `README.md` — project overview and setup instructions.

## Objective

Build a binary classifier for the income target:

- `0`: income `<=50K`
- `1`: income `>50K`

The notebook emphasizes precision, recall, and F1-score for class `1`, in addition to accuracy and ROC-AUC, because the target classes are imbalanced.

## Workflow

1. **Load and inspect the data** using pandas.
2. **Clean the data** by trimming whitespace, replacing `?` with missing values, and dropping rows containing missing values.
3. **Remove columns** `fnlwgt` and `education`.
4. **Prepare the target** by mapping `>50K` to `1` and all other income values to `0`.
5. **Explore the data** with plots of target distribution, age, hours worked per week, education level, and numeric correlations.
6. **Prepare features** using one-hot encoding for categorical variables (`drop_first=True`).
7. **Split the data** into training and test sets using an 80/20 split, `random_state=42`, and stratification.
8. **Scale numeric features only** (`age`, `educational-num`, `capital-gain`, `capital-loss`, and `hours-per-week`) using a `StandardScaler` fitted on the training set.
9. **Train and evaluate ANN models** with different configurations.
10. **Compare results** using test accuracy, class-1 precision, class-1 recall, class-1 F1, and ROC-AUC.

## Experiments Included

The notebook compares the following:

- **Baseline ANN:** two hidden layers with 128 and 64 units, ReLU activation, Glorot uniform initialization, Adam optimizer, and binary cross-entropy loss.
- **Activation functions:** ReLU, tanh, sigmoid, and ELU.
- **Weight initialization:** Glorot uniform, Glorot normal, He uniform, He normal, and zero initialization.
- **Loss functions and class imbalance:** binary cross-entropy, class-weighted binary cross-entropy training, and focal loss.
- **Batch normalization:** tests Dense → BatchNormalization → Activation ordering.
- **Optimizers:** SGD, SGD with momentum, RMSprop, Adam, and Adam with explicit hyperparameters.
- **Learning-rate sensitivity:** compares several learning rates for SGD and Adam.
- **Final combined model:** combines ReLU, He-normal initialization, batch normalization, and Adam; it uses class weighting when the weighted-BCE experiment improves class-1 F1 over the baseline.

The notebook also examines activation outputs, gradients, parameter counts, training curves, confusion matrices, and ROC curves.

## Model Architecture

The main ANN has two hidden layers:

```text
Input features
      |
Dense (128 units)
      |
Dense (64 units)
      |
Dense (1 unit, sigmoid)
      |
Predicted probability of income >50K
```

For the batch-normalized variant, each hidden block uses:

```text
Dense → BatchNormalization → Activation
```

The default training setup uses a random seed of `42`, a batch size of `64`, and a 10% validation split from the training data. The notebook's full-run settings use 50 epochs for standard experiments, 30 for shorter learning-rate experiments, and up to 80 for the final model. Set `QUICK = True` in the setup cell to use short three-epoch runs for a quicker smoke test.

## Requirements

The notebook uses Python and the following libraries:

- NumPy
- pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook or JupyterLab


## 📸Preview
![Photo](Photo.png)
![Image](Image.png)
![Img](Img.png)
![Preview](Preview.png)



## Setup and Run

1. Place `Project-2.ipynb` and `adult.csv` in the same directory.
2. Create and activate a virtual environment (recommended).
3. Install the dependencies:

   ```bash
   python -m pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow jupyter
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Project-2.ipynb` and run the cells from top to bottom.

**Important:** The notebook expects the dataset at the relative path `adult.csv`. If the file is stored elsewhere, update the `pd.read_csv('adult.csv')` path in the data-loading cell.

## Evaluation

The notebook reports:

- **Accuracy** — overall fraction of correct predictions.
- **Precision (class 1)** — fraction of predicted `>50K` cases that are actually `>50K`.
- **Recall (class 1)** — fraction of actual `>50K` cases detected.
- **F1-score (class 1)** — harmonic mean of class-1 precision and recall.
- **ROC-AUC** — ability to rank positive examples above negative examples across thresholds.

It also visualizes confusion matrices, learning curves, and ROC curves, and assembles a comparison table for selected model variants. Exact metrics are generated when the notebook runs; this README does not assume a particular result.

## Notes and Limitations

- Rows containing missing values are dropped, which reduces the dataset size.
- The scaler is fitted on the training set only to avoid leaking test-set statistics.
- The notebook uses a fixed random seed for reproducibility, but results can still vary across software versions and hardware.
- The final model's class-weighting choice is based on the notebook's single-run validation comparison. For robust conclusions, consider repeated seeds, cross-validation, and threshold tuning on validation data.
- Income prediction from demographic attributes can encode or amplify societal biases. Treat results as an educational experiment, not as a basis for consequential decisions about individuals.

