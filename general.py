import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

confusion_matrix = np.array([[3, 66], [4, 13]])
labels = ["Non-Herring", "Herring"]
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Herring Classification Confusion Matrix")
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.savefig("confusion_matrix.png")