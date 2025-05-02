
import matplotlib.pyplot as plt
import numpy as np

# --- Replace with actual metric values ---
metrics_to_plot = ["Accuracy", "F1", "Precision", "Recall", "AUC"]

# Example metrics based on user-provided values:
# Snorkel: 79.38	92.56	75.34	92.49	63.56
# WeaSEL:  83.97	90.74	84.65	80.55	89.22
# WeaSEL++:84.77	91.31	84.80	83.91	85.71

snorkel = [0.7938, 0.7534, 0.9249, 0.6356, 0.9256]
weasel = [0.8397, 0.8465, 0.8055, 0.8922, 0.9074]
weaselpp = [0.8477, 0.8480, 0.8391, 0.8571, 0.9131]

x = np.arange(len(metrics_to_plot))
width = 0.25

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, snorkel, width, label="Snorkel", color='lightgreen')
bars2 = plt.bar(x, weasel, width, label="WeaSEL", color='orange')
bars3 = plt.bar(x + width, weaselpp, width, label="WeaSEL++", color='skyblue')

# Annotate values
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f"{yval:.2f}", 
                     xy=(bar.get_x() + bar.get_width()/2, yval), 
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom')

plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Comparison of Snorkel vs WeaSEL vs WeaSEL++")
plt.xticks(ticks=x, labels=metrics_to_plot)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
