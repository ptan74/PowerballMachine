from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_url_path='/static')

# Load data from CSV
data = pd.read_csv('4d.csv')

# Preprocess the data
# You may need to convert categorical values to numerical representations
# Here, I'll focus on numerical values only
X = data.iloc[:, 1:].values  # Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_scaled, X)  # Predicting the same numbers

# Make predictions for each category
predicted_values = {}
for category in data['Price'].unique():
    # Filter data for the current category
    category_data = data[data['Price'] == category]
    X_category = category_data.iloc[:, 1:].values
    X_scaled_category = scaler.transform(X_category)

    # Predict the next numbers for the current category
    predictions = model.predict(X_scaled_category)

    # Store the predicted numbers for the current category
    predicted_values[category] = np.round(predictions).astype(int)


# Define route to render the template
@app.route('/')
def index():
    return render_template('output.html', predictions=predicted_values)


if __name__ == '__main__':
    app.run(debug=True)
