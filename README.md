# Solar Energy Output Level Prediction

## 1. Project idea

This project predicts whether a day's rooftop solar energy output will be **Low, Medium, or High** using weather-related features. This can help demonstrate how weather conditions can be used to estimate solar output.

This is a three-class classification problem. The project compares a Decision Tree classifier with a K-Nearest Neighbors (KNN) classifier.

## 2. Dataset

- **Dataset file:** [solar_data.csv](solar_data.csv)
- **Dataset generator notebook:** [generate_dataset.ipynb](generate_dataset.ipynb)
- **Number of records:** 1,500 daily records.
- **Data type:** Synthetic data generated for a 5 kW rooftop solar system at approximately 20° latitude.
- **Weather features:** Month, cloud cover, maximum and minimum temperature, humidity, rainfall, wind speed, and day length.
- **Target:** `output_level` with three classes: `Low`, `Medium`, and `High`.
- **Missing values:** Some weather readings are intentionally missing so the notebook can demonstrate missing-value handling.

The dataset is generated using a simplified solar-output formula with seasonal effects and random noise. It is intended for learning and does not represent measured real-world performance.

## 3. Files in this repository

- `solar_model.ipynb` — main notebook for data preparation, model training, evaluation, and conclusion.
- `generate_dataset.ipynb` — notebook that creates `solar_data.csv`.
- `solar_data.csv` — dataset used by the main notebook.
- `confusion_matrices.png` — confusion-matrix plot for the two models.
- `results.txt` — saved model results, if included in the repository.
- `requirements.txt` — Python package dependencies.

## 4. Setup and how to run

Install Python, then install the dependencies in a terminal:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
```

Make sure `solar_data.csv` and `solar_model.ipynb` are in the same folder. Start Jupyter:

```bash
jupyter notebook
```

Open `solar_model.ipynb` and choose **Run All**.

To regenerate the dataset, open `generate_dataset.ipynb` and run all its cells first. This overwrites `solar_data.csv` with a newly generated dataset. Then run `solar_model.ipynb`.

## 5. Method

1. Load the CSV and inspect its shape, missing values, and class balance.
2. Use weather and calendar columns as input features. `energy_kwh` is excluded from the features because it is used to derive the target label.
3. Split the data into 80% training and 20% test data using stratification.
4. Handle missing numeric values using median imputation within a scikit-learn pipeline.
5. Train and tune a Decision Tree and a KNN classifier using five-fold cross-validation. KNN features are standardized because KNN uses distances.
6. Compare test accuracy and confusion matrices.

## 6. Results

Results from the test run:

| Model | Selected setting | Test accuracy |
|---|---:|---:|
| Decision Tree | Maximum depth = 5 | **79.7%** |
| K-Nearest Neighbors (KNN) | Neighbors = 15 | 73.0% |

Confusion matrices use rows for actual labels and columns for predicted labels, in the order Low, Medium, High.

**Decision Tree**
```text
[[79 21  0]
 [13 74 13]
 [ 0 14 86]]
```

**KNN**
```text
[[77 23  0]
 [18 60 22]
 [ 0 18 82]]
```

The Decision Tree's most important features in this run were cloud cover and day length.

## 7. Conclusion

The Decision Tree performed better on this test split, achieving 79.7% accuracy compared with 73.0% for KNN. Both models found it harder to distinguish the Medium class from its neighbouring classes than to distinguish Low from High.

These results demonstrate the machine-learning workflow on a synthetic dataset. They should not be interpreted as evidence of the same accuracy on real solar installations. A useful next step would be to test the approach with real weather and solar-generation data.
