from tensorflow.keras.models import load_model

model = load_model("eco_rgb_model.h5", compile=False)

print("Model loaded successfully!")
model.summary()