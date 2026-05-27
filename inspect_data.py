import numpy as np

X = np.load("X_balanced.npy")

print("Dataset Shape:", X.shape)

sample = X[0]

print("\nSingle Sample Shape:")
print(sample.shape)

print("\nFirst Frame Shape:")
print(sample[0].shape)

print("\nFirst 20 values:")
print(sample[0][:20])

print("\nMin value:", np.min(sample))
print("Max value:", np.max(sample))