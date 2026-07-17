import os
import io
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# Create graph folder
os.makedirs("static/graphs", exist_ok=True)

# Configuration
batch_size = 128
epochs = 1
num_classes = 10

# Load dataset
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print("Training data shape :", train_X.shape)
print("Testing data shape  :", test_X.shape)

# Preprocessing
train_X = train_X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_X = test_X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encoding
train_Y = to_categorical(train_Y, num_classes)
test_Y = to_categorical(test_Y, num_classes)

print("\nExample Label after One-Hot Encoding:")
print(train_Y[0])

# Train Validation Split
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X,
    train_Y,
    test_size=0.2,
    random_state=13
)

# CNN Model
fashion_model = Sequential()

fashion_model.add(
    Conv2D(
        32,
        kernel_size=(3,3),
        activation='linear',
        input_shape=(28,28,1),
        padding='same'
    )
)

fashion_model.add(LeakyReLU(negative_slope=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

fashion_model.add(
    Conv2D(
        64,
        kernel_size=(3,3),
        activation='linear',
        padding='same'
    )
)

fashion_model.add(LeakyReLU(negative_slope=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

fashion_model.add(
    Conv2D(
        128,
        kernel_size=(3,3),
        activation='linear',
        padding='same'
    )
)

fashion_model.add(LeakyReLU(negative_slope=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(negative_slope=0.1))

fashion_model.add(Dense(num_classes, activation='softmax'))

# Compile
fashion_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Summary")

# Print model summary to console
fashion_model.summary()

# Train model
history = fashion_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

# Evaluate
test_loss, test_accuracy = fashion_model.evaluate(
    test_X,
    test_Y,
    verbose=0
)

print("\nTest Loss :", test_loss)
print("Test Accuracy :", test_accuracy)

# Plot Accuracy
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("static/graphs/cnn_accuracy.png")
plt.close()