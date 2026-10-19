# Solar-energy-detector
I wanted to see if a computer could look at a day's weather and guess whether a rooftop solar panel would have a Low, Medium or High day. This project tries that with two beginner-friendly models: a Decision Tree and K-Nearest Neighbors (KNN).

It's useful because if you know tomorrow will be a strong solar day, you can plan around it, like running the washing machine or charging an EV when the panels are working hardest.

The data
`solar_data.csv` has 1,500 days of weather: cloud cover, temperatures, humidity, rain, wind, day length and month.

I should be upfront that the data is made up. I generated it with `generate_dataset.py` using simple solar physics plus some random noise, and I deliberately removed a few values to practice handling missing data. So the results show the method works, but they don't tell us how it would do on real panels.

Trying yourself
```bash
pip install -r requirements.txt
python solar_model.py
```
This trains both models, prints how they did, and saves a confusion matrix picture.

My learnings
- The Decision Tree won. My guess is that it can ignore the weather details that don't matter, while KNN treats everything as equally important.
- Cloud cover and day length mattered most, which makes sense: they decide how much sunlight actually reaches the panels.
- Neither model ever mixed up a Low day with a High day. The mistakes were almost all Medium days getting confused with the ones next to them.

My next step
Try it on real solar data, predict the actual kWh instead of just three categories, and plug in a real weather forecast.
