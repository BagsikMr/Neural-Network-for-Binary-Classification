import numpy as np
import tensorflow as tf


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
