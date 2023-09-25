# Import necessary libraries
import pandas as pd     # For data handling and manipulation
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # To build a random forest classifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,  # For model evaluation
    roc_curve, roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For creating heatmaps

# Replace with the path to your CSV file
csv_file_path = (r"C:\Users\valle\OneDrive\Desktop\Assignment-2\employee attrition.csv")

# Load the dataset
data = pd.read_csv(csv_file_path)

# Handle missing values (if any)
data.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['sales'] = label_encoder.fit_transform(data['sales'])
data['salary'] = label_encoder.fit_transform(data['salary'])

# Split the data into features (X) and target (y) columns
X = data.drop('left', axis=1)  # Features (information about employees)
y = data['left']  # Target (whether an employee left or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Scaling features for training data
X_test = scaler.transform(X_test)  # Scaling features for testing data

# Build and train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)  # Creating a machine learning model
model.fit(X_train, y_train)  # Training the model on the training data

# Make predictions
y_pred = model.predict(X_test)  # Using the trained model to make predictions on the test data

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculating the accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)  # Creating a confusion matrix
class_report = classification_report(y_test, y_pred)  # Generating a classification report

# Print the accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)

# Generate ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]  # Calculating class probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_probs)  # Creating ROC curve data
roc_auc = roc_auc_score(y_test, y_probs)  # Calculating AUC-ROC

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Generate Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)  # Creating Precision-Recall curve data
auc_pr = auc(recall, precision)  # Calculating AUC-PR

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Generate Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
