import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Przygotowanie danych treningowych
def generate_data(num_samples, radius):
    points = []
    labels = []
    for _ in range(num_samples):
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        distance = np.sqrt(x**2 + y**2)
        label = 1 if distance <= radius else 0
        points.append([x, y])
        labels.append(label)
    return np.array(points), np.array(labels)


# Zdefiniowanie struktury sieci neuronowej
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Przygotowanie danych treningowych
num_samples = 1000
radius = 5.0
points, labels = generate_data(num_samples, radius)

# Trenowanie sieci neuronowej
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(points, labels, epochs=10, batch_size=32)

# Testowanie sieci neuronowej
test_points, test_labels = generate_data(100, radius)
predictions = model.predict(test_points)
rounded_predictions = np.round(predictions).flatten()

accuracy = np.sum(rounded_predictions == test_labels) / len(test_labels)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

# Obliczanie powierzchni decyzyjnej
x_min, x_max = -radius - 1, radius + 1
y_min, y_max = -radius - 1, radius + 1
step = 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid_points)
grid_labels = np.round(grid_predictions).flatten()

plt.figure()
plt.title("Powierzchnia decyzyjna")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(grid_points[:, 0], grid_points[:, 1], c=grid_labels, cmap='coolwarm', alpha=0.6)
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='coolwarm', edgecolors='black')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
