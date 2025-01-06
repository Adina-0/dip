import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA


def analyze_feature_performance(data):
    # # Convert nested dictionary into DataFrame
    # flat_data = []
    # labels = []
    # for class_name, descriptors in data.items():
    #     for i in range(len(next(iter(descriptors.values())))):  # Assuming all descriptor lists are of same length
    #         row = {'Class': class_name}
    #         for descriptor_name, values in descriptors.items():
    #             row[descriptor_name] = values[i]
    #         flat_data.append(row)
    #         labels.append(class_name)
    #
    # df = pd.DataFrame(flat_data)

    df = data.copy()

    # Step 2: Encode Class Labels
    le = LabelEncoder()
    df['Class_encoded'] = le.fit_transform(data['Class'])

    # Step 3: Correlation Matrix
    correlation_matrix = df.drop(columns=['Class', 'Class_encoded']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()

    # Step 4: Train a RandomForestClassifier to Evaluate Feature Importance
    X = df.drop(columns=['Class', 'Class_encoded'])
    y = df['Class_encoded']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(X.shape[1]), importances[indices], align="center")
    plt.yticks(range(X.shape[1]), X.columns[indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

    # Step 5: PCA for Dimensionality Reduction and Visualization
    pca = PCA(n_components=2)

    # drop NaN values
    nan_indexes = X.index[X.isnull().any(axis=1)]
    X = X.dropna()
    df = df.drop(nan_indexes)

    X_pca = pca.fit_transform(X)
    print(X_pca)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Class'], palette='viridis')

    plt.title('PCA of Feature Space')
    plt.tight_layout()
    plt.show()

    # Step 6: Classification Report and Confusion Matrix
    y_pred = rf.predict(X_test)

    # Classification report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# #path to csv
# path = "./Results/allFeatures_new.csv"
# data = pd.read_csv(path)
# data.set_index('Image', inplace=True)
# data = data[['Class', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Gabor Mean', 'Fourier Mean', 'Entropy Channel B', 'Entropy Channel S', 'Entropy Channel V', 'LBP Mean', 'LBP Median', 'LBP Mode', 'LBP Standard Deviation', 'LBP IQR', 'LBP Skewness', 'LBP Kurtosis', 'LBP Minimum', 'LBP Maximum', 'LBP Entropy']]
# data = data[['Class', 'Mean Intensity', 'Intensity Variation', 'Mean R', 'Mean G', 'Mean B', 'Mean Brightness', 'Brightness Variation', 'Mean Density', 'Density Variation', 'Mean Saturation', 'Typical Hue', 'Hue Variation']]
# data = data['Area', 'Extent', 'Aspect Ratio', 'Major Axis Length', 'Minor Axis Length', 'Eccentricity', 'Convex Area', 'Equivalent Diameter', 'Solidity', 'Perimeter', 'Circularity', 'Thickness']]
# analyze_feature_performance(data)