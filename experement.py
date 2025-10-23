# ================================================================
# Project: Deep Learning + Linear Regression Analysis
# ================================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ================================================================
# PART 1: CNN (VGG16) - Image Classification
# ================================================================

print("\n--- PART 1: CNN Model (VGG16) ---")

# Step 1: Load the VGG16 base model
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional base
for layer in vgg_base.layers:
    layer.trainable = False

# Step 2: Add classification layers
x = Flatten()(vgg_base.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # change 10 to number of your classes

vgg_complete = Model(inputs=vgg_base.input, outputs=output)

# Step 3: Compile model
vgg_complete.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Prepare your image data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Step 5: Train the model
vgg_complete.fit(train_generator, epochs=5, validation_data=test_generator)

# Step 6: Evaluate the model
loss, acc = vgg_complete.evaluate(test_generator)
print(f"✅ CNN Test Accuracy: {acc:.2f}")

# Step 7: Confusion Matrix
y_true = test_generator.classes
y_pred_probs = vgg_complete.predict(test_generator)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - VGG16 Model")
plt.show()


# ================================================================
# PART 2: Linear Regression - Sales Prediction
# ================================================================

print("\n--- PART 2: Linear Regression Model ---")

# Step 1: Load dataset
df = pd.read_csv('DATA.csv', encoding='ascii')
print(f"Data shape: {df.shape}")
print(df.head())

# Step 2: Clean data
if 'date' in df.columns:
    df.drop('date', axis=1, inplace=True)
if 'state_holiday' in df.columns:
    df = pd.concat([df, pd.get_dummies(df['state_holiday'], prefix='SH')], axis=1)
    df.drop('state_holiday', axis=1, inplace=True)

# Step 3: Correlation Matrix
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 4: Split data
X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Linear Regression Results:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Step 7: Visualization - Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.show()

# Step 8: Predict on REAL_DATA.csv
print("\nSTEP 8: Predicting on REAL_DATA.csv...")
real_df = pd.read_csv('REAL_DATA.csv', index_col=0, encoding='ascii')

if 'date' in real_df.columns:
    real_df.drop('date', axis=1, inplace=True)
if 'state_holiday' in real_df.columns:
    real_df = pd.concat([real_df, pd.get_dummies(real_df['state_holiday'], prefix='SH')], axis=1)
    real_df.drop('state_holiday', axis=1, inplace=True)

# Align columns
missing_cols = set(X.columns) - set(real_df.columns)
for col in missing_cols:
    real_df[col] = 0
real_df = real_df[X.columns]

# Scale and predict
real_df_scaled = scaler.transform(real_df)
y_real = model.predict(real_df_scaled)

# Save predictions
real_df['sales'] = y_real
real_df.to_csv('test_with_predictions.csv', index=False)
print("✅ Predictions saved to test_with_predictions.csv")
