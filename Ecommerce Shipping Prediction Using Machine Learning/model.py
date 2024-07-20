import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Train.csv')

# Encode categorical variables
le = LabelEncoder()
df['Warehouse_block'] = le.fit_transform(df['Warehouse_block'])
df['Mode_of_Shipment'] = le.fit_transform(df['Mode_of_Shipment'])
df['Product_importance'] = le.fit_transform(df['Product_importance'])
df['Gender'] = le.fit_transform(df['Gender'])

# Scale/normalize features
scaler = StandardScaler()
columns_to_scale = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split data into training and testing sets
X = df.drop(['Reached.on.Time_Y.N'], axis=1)
y = df['Reached.on.Time_Y.N']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
params = {
    'n_estimators': [200, 250, 300],  # Increase the number of estimators
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],  # Add max_depth parameter
    'min_samples_split': [2, 5, 10],  # Add min_samples_split parameter
    'min_samples_leaf': [1, 5, 10]  # Add min_samples_leaf parameter
}

# Perform hyperparameter tuning
rf_model = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring='accuracy', cv=5)
rf_model.fit(X_train, y_train)

# Get the best-performing model and its hyperparameters
best_model = rf_model.best_estimator_
best_params = rf_model.best_params_

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Evaluate the model's performance
print("Best Parameters:", best_params)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
