import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Separate target and features
X = train_df.drop(columns=['id', 'CustomerId', 'Surname', 'Exited'])
y = train_df['Exited']
test_ids = test_df['id']
X_test_raw = test_df.drop(columns=['id', 'CustomerId', 'Surname'])

# Define column types
categorical_cols = ['Geography', 'Gender']
numeric_cols = X.columns.difference(categorical_cols)

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Pipeline with RandomForest to evaluate feature importance
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=74))
])

rf_pipeline.fit(X, y)

# Get feature names after preprocessing
encoded_cat_cols = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numeric_cols, encoded_cat_cols])
feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Get top N features
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print("Feature importances from Random Forest:")
print(importance_df)

top_features = importance_df['Feature'].head(8).tolist()  # Adjust number of top features if needed
print("Top features based on Random Forest importance:")
print(top_features)

# New preprocessor using only top features
# Mapping names back to columns (OneHotEncoder output must be matched carefully)
num_top = [f for f in top_features if f in numeric_cols]
cat_top = [f for f in top_features if f not in numeric_cols]

# Split original train into training and validation sets
X_train_full = X
y_train_full = y
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# Fit the new preprocessor on the training part only
new_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_top),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train_transformed = new_preprocessor.fit_transform(X_train)
X_val_transformed = new_preprocessor.transform(X_val)
X_test_transformed = new_preprocessor.transform(X_test_raw)

# Get encoded feature names and select top features
encoded_feature_names = new_preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_transformed_features = np.concatenate([num_top, encoded_feature_names])
top_feature_indices = [i for i, f in enumerate(all_transformed_features) if f in top_features]

X_train_selected = X_train_transformed[:, top_feature_indices]
X_val_selected = X_val_transformed[:, top_feature_indices]
X_test_selected = X_test_transformed[:, top_feature_indices]

# Train Logistic Regression on training set
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_selected, y_train)

# Evaluate on validation set
y_val_proba = log_reg.predict_proba(X_val_selected)[:, 1]
threshold = 0.5  # try 0.3, 0.6, etc. depending on what you care about
y_val_pred = (y_val_proba >= threshold).astype(int)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Predict on test set
y_test_proba = log_reg.predict_proba(X_test_selected)[:, 1]

# Save predictions
output_df = pd.DataFrame({
    'id': test_ids,
    'Exited': y_test_proba
})
output_df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")
