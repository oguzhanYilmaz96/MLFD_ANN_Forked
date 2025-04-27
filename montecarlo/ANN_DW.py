
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:40:28 2025

@author: yilmaz
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt


# 1) Load or define your data matrix (19 x 14):
dataMatrix = pd.read_csv("Base.csv", header=None).values

"""
Purpose:    Loads the data matrix from a .csv file named "Base.csv".
Operation:  pd.read_csv reads the CSV file.
            header=None specifies that the file has no header row.
            .values converts the DataFrame into a NumPy array for numerical operations.
"""
# 2) Extract AoA, h/c, and cl_data
h_over_c = dataMatrix[0, 1:]  # Row 1 (cols 2..end)
AoA = dataMatrix[1:, 0]       # Col 1 (rows 2..end)
cl_data = dataMatrix[1:, 1:]  # Body (rows 2..end, cols 2..end)

nAoA = len(AoA)
nH = len(h_over_c)

"""
Purpose:    Extracts specific parts of the matrix:
            h/c: Non-dimensional height array.
            AoA: Angle of Attack array.
            cl_data: Lift coefficient data.
Operation:  dataMatrix[0, 1:]: Takes all elements in the first row except the first column.
            dataMatrix[1:, 0]: Takes the first column from the second row onward.
            dataMatrix[1:, 1:]: Takes the rest of the matrix (ignoring the first row and column).
"""

# Display experimental h/c range
print(f"Experimental h/c range: {np.min(h_over_c)*100:.2f}% to {np.max(h_over_c)*100:.2f}%")

# 3) Create (AoA, h/c) -> cl samples, skipping NaN
allX = []
allY = []

for i in range(nAoA):
    for j in range(nH):
        this_cl = cl_data[i, j]
        if not np.isnan(this_cl):
            allX.append([AoA[i], h_over_c[j]])
            allY.append(this_cl)

allX = np.array(allX)
allY = np.array(allY)

"""
Purpose:    Builds valid samples of inputs (AoA, h/c) and corresponding outputs (cl), skipping missing data (NaN values).
Operation:  Iterates through every combination of AoA and h/c.
            Checks if the lift coefficient (cl) value is NaN using np.isnan.
            Appends valid pairs to allX and their corresponding cl values to allY.
"""

# 4) Split into Training and Test Sets (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(allX, allY, test_size=0.2)

"""
Purpose:    Splits the data into training (80%) and test (20%) datasets.
Operation:  train_test_split: Randomly divides the data.
            test_size=0.2: Reserves 20% of the data for testing.
"""

# Display unique h/c values in training data
print(f"Unique h/c values in training data: {np.unique(X_train[:, 1]) * 100}")

# 5) Feature Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
Purpose:    Normalizes the data to improve ANN performance.
Operation:  StandardScaler(): Standardizes data to have a mean of 0 and a standard deviation of 1.
            scaler.fit: Computes the scaling parameters from the training data.
            scaler.transform: Applies scaling to both training and test data.
"""

# 6) Build and Train the ANN (Regression)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(2,)),
#     tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
#     tf.keras.layers.Dense(1, activation='linear')
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(8, activation='elu', input_shape=(2,)),
#     tf.keras.layers.Dense(8, activation='elu'),
#     tf.keras.layers.Dense(1, activation='linear')
# ])


model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=2000, verbose=1, validation_split=0.2)

"""
Purpose:    Constructs and trains a feedforward neural network.
Operation:  tf.keras.Sequential: Creates a sequential model.
            tf.keras.layers.Dense: Adds densely connected layers:
                First two layers have 8 neurons each, using ReLU activation.
                Output layer has 1 neuron with a linear activation for regression.
            compile: Configures the model for training with:
                optimizer='adam': Adaptive optimization.
                loss='mse': Mean Squared Error loss for regression.
            fit: Trains the model:
                epochs=150: Trains for 150 iterations.
                validation_split=0.2: Reserves 20% of training data for validation.                
"""

# 7) Predict on Test Data & Evaluate
y_pred = model.predict(X_test_scaled).flatten()
mse_val = np.mean((y_test - y_pred) ** 2)
print(f"Test MSE: {mse_val:.5f}")

"""
Purpose:    Tests the model and computes the Mean Squared Error (MSE).
Operation:  model.predict: Generates predictions for the scaled test data.
            flatten(): Converts predictions to a 1D array.
            np.mean: Calculates MSE between actual (y_test) and predicted values.
"""

# 8) Predict Cl for Specific (AoA, h/c) Pairs
grid_AoA = np.linspace(np.min(AoA), np.max(AoA), 17)
grid_h_to_c = np.linspace(18.5, 107, 13)
legend_h_over_c = np.unique(grid_h_to_c)

# Prepare grid for predictions
gridX, gridY = np.meshgrid(grid_AoA, legend_h_over_c)
grid_points = np.column_stack((gridX.ravel(), gridY.ravel()))

# Scale the grid points
grid_points_scaled = scaler.transform(grid_points)

# Predict Cl values for the entire grid
grid_cl = model.predict(grid_points_scaled).reshape(gridX.shape)

"""
Purpose:    Generates a grid of predictions over a range of AoA and h/c.
Operation:  np.linspace: Creates a range of evenly spaced values.
            np.meshgrid: Forms a 2D grid of these values.
            np.column_stack: Combines grid coordinates into an array.
            scaler.transform: Scales the grid for predictions.
            model.predict: Predicts cl values for the grid.
"""

# 9) Plot the Predicted Cl vs AoA for Each h/c
plt.figure()
colors = plt.cm.jet(np.linspace(0, 1, len(legend_h_over_c)))
for i, color in enumerate(colors):
    plt.plot(grid_AoA, grid_cl[i, :], color=color, label=f"h/c = {legend_h_over_c[i]:.1f}%")
plt.xlabel("\u03B1 (degrees)")
plt.ylabel("C_L")
plt.legend()
plt.grid(True)
plt.title("Predicted C_L vs \u03B1 for Various h/c")
plt.show()

"""
Purpose:    Visualizes the predicted Cl values for varying AoA and fixed h/c.
Operation:  Loops through each row of the grid and plots Cl values for the corresponding h/c.
"""

# 10) 3D Surface Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(gridX, gridY, grid_cl, cmap='viridis')
plt.colorbar(surf)
ax.set_xlabel("\u03B1 (degrees)")
ax.set_ylabel("h/c")
ax.set_zlabel("C_L")
plt.title("Predicted C_L(\u03B1, h/c)")
plt.show()

"""
Purpose:    Plots a 3D surface of predicted Cl as a function of AoA and h/c.
"""

# 11) Predict C_L for Specific Experimental Points
sample_point = np.array([[10, 50]])
sample_point_scaled = scaler.transform(sample_point)
predicted_CL = model.predict(sample_point_scaled).flatten()[0]
print(f"Predicted C_L at \u03B1 = {sample_point[0, 0]:.1f}, h/c = {sample_point[0, 1]:.1f}%: {predicted_CL:.5f}")

"""
Purpose:    Predicts Cl for a specific combination of AoA and h/c.
"""

# Save ANN weights and biases
weights_and_biases = {layer.name: {'weights': layer.get_weights()[0], 'biases': layer.get_weights()[1]} for layer in model.layers}
np.save("ANN_Weights.npy", weights_and_biases)

"""
Purpose:    Saves the weights and biases of the trained ANN to a file for future use.
"""

# Load experimental data from 'Base.csv'
experimental_data = pd.read_csv('Base.csv')

# Prepare experimental data for plotting
AoA_exp = experimental_data.iloc[1:, 0].astype(float).values  # Extract AoA values (skipping header)
h_over_c_exp = experimental_data.columns[1:].astype(float)    # Extract h/c values
cl_exp = experimental_data.iloc[1:, 1:].astype(float).values  # Extract Cl data (skipping header)

# Determine the number of unique h/c values from both predicted and experimental data
num_h_over_c = max(len(legend_h_over_c), len(h_over_c_exp))

# Generate colors based on the larger dataset
colors = plt.cm.jet(np.linspace(0, 1, num_h_over_c))

# Plot predicted data
plt.figure()
for i, color in enumerate(colors[:len(legend_h_over_c)]):
    # plt.plot(grid_AoA, grid_cl[i, :], color=color, label=f"Predicted h/c = {legend_h_over_c[i]:.1f}%")
    plt.plot(grid_AoA, grid_cl[i, :], color=color)


# Plot experimental data
for i, (h_c, color) in enumerate(zip(h_over_c_exp, colors[:len(h_over_c_exp)])):
    plt.scatter(AoA_exp, cl_exp[:, i], color=color, edgecolor='k', s=25, label=f"Exp h/c = {h_c:.1f}%")

# Labels, legend, and grid
plt.xlabel("\u03B1 (degrees)")
plt.ylabel("C_L")
plt.legend(
    loc='lower right',
    ncol=2,
    fontsize='small',
    bbox_to_anchor=(1, 0)
)
plt.grid(True)
plt.title("Predicted and Experimental C_L vs \u03B1 for Various h/c")
plt.show()



