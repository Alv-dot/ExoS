import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("performance_log.csv")

df['GroundTruth'] = [0, 1, 2, 3, 4, 5] * (len(df) // 6)

df['Correct'] = (df['Prediction'] == df['GroundTruth']).astype(int)
accuracy = df['Correct'].mean() * 100

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Correct'].cumsum() / (df.index + 1), label="Cumulative Accuracy")
plt.axhline(y=accuracy / 100, color='r', linestyle='--', label=f"Final Accuracy: {accuracy:.2f}%")
plt.xlabel("Time (samples)")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time")
plt.legend()
plt.show()
