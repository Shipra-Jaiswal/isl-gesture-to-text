from tensorflow.keras.models import load_model

model = load_model("GRU_Transformer.keras")

print("INPUT SHAPE:")
print(model.input_shape)

print("\nOUTPUT SHAPE:")
print(model.output_shape)