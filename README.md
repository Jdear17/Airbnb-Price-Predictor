# Airbnb Price Predictor

## Project Overview
This project uses machine learning techniques to predict Airbnb listing prices based on features such as room type, location, availability, and reviews. The goal is to build an interpretable and accurate pricing model by analyzing trends and applying advanced regression techniques. note making changes 21/05/25

---

## Features
- **Room Type**: One-hot encoded categories for room types (e.g., Entire home/apt, Private room).
- **Location**: Calculated distance to city center (e.g., London).
- **Availability**: Number of available days per year.
- **Reviews**: Number of reviews, reviews per month.

---

## Workflow
1. **Data Loading and Cleaning**:
   - Load the Airbnb dataset.
   - Handle missing values and remove outliers based on price quartiles.

2. **Feature Engineering**:
   - One-hot encoding for categorical variables.
   - Calculate distance to city center.
   - Normalize and scale features as needed.

3. **Modeling**:
   - Train-test split of the data.
   - Use Random Forest Regressor with hyperparameter tuning.
   - Evaluate performance with metrics like MAE, MSE, and R².

4. **Visualization**:
   - Correlation heatmaps to analyze feature relationships.
   - Scatter plots and boxplots for data distribution and outlier detection.
   - Bar plots for error analysis by room type.

---

## Performance Metrics
- **Overall Metrics**:
  - Mean Absolute Error (MAE): `xx.xx`
  - Mean Squared Error (MSE): `xx.xx`
  - R² Score: `xx.xx`

- **Room Type Metrics**:
  | Room Type         | MAE     | MSE       | R²    |
  |-------------------|---------|-----------|--------|
  | Entire home/apt   | `xx.xx` | `xx.xx`   | `xx.xx`|
  | Hotel room        | `xx.xx` | `xx.xx`   | `xx.xx`|
  | Private room      | `xx.xx` | `xx.xx`   | `xx.xx`|
  | Shared room       | `xx.xx` | `xx.xx`   | `xx.xx`|

---

## Project Structure
```
Airbnb_Price_Predictor/
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks for analysis and modeling
├── src/                 # Source code for data processing and modeling
├── visuals/             # Saved plots and visualizations
├── README.md            # Project documentation
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Airbnb_Price_Predictor.git
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks:
   ```bash
   jupyter notebook
   ```

---

## Usage
- Load and preprocess the dataset using the provided scripts.
- Train the model using the hyperparameter-tuned Random Forest Regressor.
- Visualize the predictions and analyze errors using the provided notebooks.

---

## Future Improvements
- Incorporate additional features like amenities and neighborhood demographics.
- Explore advanced models like Gradient Boosting or Neural Networks.
- Automate hyperparameter tuning with Bayesian optimization.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you'd like to improve the project.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- Dataset source: [Airbnb Open Data](https://www.kaggle.com/).
- Visualization tools: Matplotlib and Seaborn.
- Special thanks to the open-source community for supporting tools like Pandas and Scikit-learn.
