import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('classification_scores.csv')

# Emotions
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Convert emotion columns to numeric data type
for emotion in emotions:
    df[emotion] = pd.to_numeric(df[emotion], errors='coerce')

# Create an empty confusion matrix
confusion_matrix = pd.DataFrame(0, index=emotions, columns=emotions)

for _, row in df.iterrows():
    # Get the true emotion label (assuming it's in the 'filepath' column)
    true_label = row['filepath'].split('_')[-1].split('.')[0]
    
    # Get the predicted emotion label (the emotion with the highest score)
    predicted_label = row[emotions].astype(float).idxmax()
    
    # Increment the count in the confusion matrix
    confusion_matrix.loc[true_label, predicted_label] += 1

# Normalize the confusion matrix by dividing each row by its sum
confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

# Plot the normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_normalized, annot=True, cmap='Purples', fmt='.2f', 
            xticklabels=labels, yticklabels=labels, annot_kws={'size': 14},
           vmin=0.1, vmax=1.0)

plt.xlabel('Predicted Label', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, rotation=22.5)
plt.ylabel('True Label', fontsize=20, fontweight='bold')
plt.yticks(fontsize=16, rotation=66.5)
plt.title('Confusion Matrix', fontsize=24, fontweight='bold')
plt.savefig('Confusion_Matrix.pdf', dpi=800, bbox_inches='tight', pad_inches=0.1)
plt.show()