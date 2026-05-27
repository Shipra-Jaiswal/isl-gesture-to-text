import numpy as np

labels = np.load("labels_2.npy", allow_pickle=True)

print(labels)
print(type(labels))
print(len(labels))