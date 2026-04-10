from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -----------------------------
# Simple CNN Model (Dummy Model)
# -----------------------------
model = Sequential()

# Convolution layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sleep / Awake

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Save Model
# -----------------------------
model.save("CNN_model.h5")

print("✅ CNN_model.h5 created successfully!")
