import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# Load the housing dataset into a Pandas DataFrame
df = pd.read_csv("Housing.csv")

# Basic exploration of the dataset
# Uncomment if you wanna print exploration of data
# print(df.head())          # First 5 rows
# print(df.info())          # Data types and non-null counts
# print(df.describe())      # Statistical summary of numerical columns
# print(df.shape)           # (rows, columns)
# print(df.isnull().sum())  # Check for missing values

# List of binary categorical columns containing "yes"/"no"
binary_cols=[
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

# Convert yes/no values into 1/0
for col in binary_cols:
    df[col]=df[col].map({
        "yes":1,
        "no":0
    })

# Verify the transformed dataset
print(df.head())

# One-Hot Encode the furnishingstatus column
# drop_first=True prevents multicollinearity (dummy variable trap)
df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    drop_first=True
)

# View the final column names after encoding
print(df.columns)

# Separate features (X) and target variable (Y)
X= df.drop("price",axis=1)
Y=df["price"]

# Uncomment if you want to inspect feature and target dimensions
# print(X.shape)
# print(Y.shape)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Display dataset structure after splitting
print(df.columns)
print(X_train.shape)
print(X_test.shape)

# Create and train a Linear Regression model
model=LinearRegression()
model.fit(X_train,y_train)

# Generate predictions on the test dataset
predictions=model.predict(X_test)

# Display first five predictions
print(predictions[:5])

# Calculate Mean Absolute Error (average prediction error)
mae=mean_absolute_error(
    y_test,
    predictions
)

# Calculate R² score (goodness of fit)
r2=r2_score(
    y_test,
    predictions
)

# Display evaluation metrics
print("MAE:",mae)
print("R2 Score:",r2)

# Compare actual house prices with predicted prices
comparison=pd.DataFrame({
    "Actual":y_test,
    "Predicted":predictions
})

print(comparison.head(10))

# Extract feature coefficients learned by Linear Regression
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# Sort coefficients to identify most influential features
print(
    coefficients.sort_values(
        by="Coefficient",
        ascending=False
    )
)

# Visualize Actual vs Predicted house prices
plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

# Save plot as an image file
plt.savefig("actual_vs_predicted.png")

# Display the plot
plt.show()

with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")