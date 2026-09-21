import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Create a DataFrame
data = pd.DataFrame(data_dict)

# Define features (X) and target (y)
X = data.drop(columns=['yes creator', 'person'])
y = data['yes creator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# Function to add new data and update the model
def update_model(new_data, new_label):
    global data, clf, X_train, X_test, y_train, y_test
    # Append the new data to the original DataFrame
    new_entry = pd.DataFrame([new_data], columns=data.columns.drop(['yes creator', 'person']))
    new_entry['yes creator'] = new_label
    new_entry['person'] = 'New Entry'  # placeholder for person name
    data = pd.concat([data, new_entry], ignore_index=True)

    # Update X and y with the new data
    X = data.drop(columns=['yes creator', 'person'])
    y = data['yes creator']

    # Split the data again
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrain the model
    clf.fit(X_train, y_train)
    print("Model updated with new data.")


# Example: Update model with new data
new_data = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]  # Example new input features
new_label = 1  # Example new label
update_model(new_data, new_label)

# Predict on the test set and evaluate
y_pred = clf.predict(X_test)
print("Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Creator', 'Yes Creator']))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Creator', 'Yes Creator'],
            yticklabels=['No Creator', 'Yes Creator'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
