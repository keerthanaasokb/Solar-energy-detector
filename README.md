# Predicting Rooftop Solar Output from the Weather

## 1. The idea
A homeowner with solar panels wants to know tomorrow's solar output before it happens, so they can plan when to run the washing machine, charge an EV, or expect to sell power back to the grid.

This project predicts whether a day's solar output will be **Low, Medium or High** from that day's weather (cloud cover, temperature, humidity, rain, wind, day length, month). It is a 3-class classification problem. It extends the weather-to-solar forecast prototype, which used a physics formula, by learning the same relationship from data.

## 2. Dataset
- **File:** `solar_data.csv` (included, 1,500 days, 10 columns plus the target).
- **Source:** synthetic. `generate_dataset.py` builds it from sun-position geometry, a cloud-transmission curve, a heat penalty and random noise, for a 5 kW system at 20° latitude. No real dataset link is used because I had no internet access while building this.
- **Missing values:** about 3% of cloud, temperature, humidity and wind readings are blanked on purpose, to mimic faulty sensors.
- **Target:** `output_level`, made from daily energy (`energy_kwh`) using the 33rd and 67th percentiles: Low below 19 kWh, Medium 19 to 22, High above 22. The classes are balanced (about 500 each).
- **Features:** `month, cloud_cover_pct, temp_max_c, temp_min_c, humidity_pct, rain_mm, wind_kmh, day_length_h`. `energy_kwh` is left out because the label is derived from it.
- **Using real data instead:** replace `solar_data.csv` with any daily weather and generation table. Keep the same column names or edit the `features` list in `solar_model.py`. Public plant-level solar datasets (for example on Kaggle) work well for this.

## 3. Setup
```bash
pip install -r requirements.txt
python generate_dataset.py   # optional, recreates solar_data.csv
python solar_model.py        # trains, evaluates, saves confusion_matrices.png
```

## 4. Method
- 80/20 train/test split, stratified so each class appears equally in both.
- Missing values filled with the median, learned from training data only (inside a scikit-learn `Pipeline`, so no test data leaks in).
- **Decision Tree:** `max_depth` picked from 2 to 8 by 5-fold cross-validation (best: 5).
- **K-Nearest Neighbors:** features standardized first, because KNN uses distances. `k` picked from 3 to 25 by cross-validation (best: 15).

## 5. Results (300 test days)
| Model | Test accuracy |
|---|---|
| **Decision Tree (depth 5)** | **79.7%** |
| KNN (k = 15) | 73.0% |

Confusion matrices (rows = actual, columns = predicted; order Low, Medium, High):

```
Decision Tree          KNN
[[79 21  0]            [[77 23  0]
 [13 74 13]             [18 60 22]
 [ 0 14 86]]            [ 0 18 82]]
```
Plots are in `confusion_matrices.png`; full output is in `results.txt`.

Most important features for the tree: cloud cover (58%), day length (36%), rain (4%). Everything else is under 1%.

## 6. Conclusion
The **Decision Tree performed better** (79.7% vs 73.0%). What I learned:
- **Errors are between neighbouring classes.** Neither model ever confused Low with High. Mistakes are all Medium versus its neighbours, because the class boundaries are arbitrary cut-offs on a continuous number.
- **Cloud cover and day length drive the prediction,** which matches the physics: sunlight available times how much cloud blocks it.
- **Why the tree won here:** it can ignore the noisy, irrelevant features (wind, humidity, minimum temperature). KNN treats every feature as equally important, so those features add noise to its distance calculations.
- **Caveat:** the data is synthetic, so these accuracy numbers show the method works but say nothing about real-world accuracy. Next steps are real data, predicting kWh directly with regression, and using a real weather forecast as input.
