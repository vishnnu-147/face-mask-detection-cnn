# Face Mask Detection using CNN (Demo Version)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate synthetic images
def generate_images(samples=100, img_size=64):
    X = []
    y = []
    for i in range(samples):
        # No mask
        img0 = np.random.rand(img_size, img_size, 3)
        X.append(img0)
        y.append(0)

        # Mask
        img1 = np.random.rand(img_size, img_size, 3)
        img1[24:40, 16:48, :] += 0.5
        img1 = np.clip(img1, 0, 1)
        X.append(img1)
        y.append(1)

    return np.array(X), np.array(y)

X, y = generate_images(120)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
