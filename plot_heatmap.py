import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('classification_scores_valid.csv')

# Get the emotion labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

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

confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_normalized, annot=True, cmap='Oranges', fmt='.2f', 
            xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('validation.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
plt.show()
