import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, StackingClassifier


# ===Read and show data info===
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Combine dataframes vertically (stacking rows)
df = pd.concat([df_train, df_test], ignore_index=True)

# Display the first few rows of the dataframe
df.head()

# Function to display data information and basic statistics
def get_df_info(df):
    print("\n\033[1mShape of DataFrame:\033[0m ", df.shape)
    print("\n\033[1mColumns in DataFrame:\033[0m ", df.columns.to_list())
    print("\n\033[1mData types of columns:\033[0m\n", df.dtypes)

    print("\n\033[1mInformation about DataFrame:\033[0m")
    df.info()

    print("\n\033[1mNumber of unique values in each column:\033[0m")
    for col in df.columns:
        print(f"\033[1m{col}\033[0m: {df[col].nunique()}")

    print("\n\033[1mNumber of null values in each column:\033[0m\n", df.isnull().sum())

    print("\n\033[1mNumber of duplicate rows:\033[0m ", df.duplicated().sum())

    print("\n\033[1mDescriptive statistics of DataFrame:\033[0m\n", df.describe().transpose())

    # Plotting missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    # Plotting the distribution of target variable
    plt.figure(figsize=(8, 4))
    sns.countplot(x='Attrition', data=df)
    plt.title('Class Distribution')
    plt.show()

# Call the function
get_df_info(df)

# ===Data Preprocessing for Machine Learning===
# Divide the dataframe into features (X) and target (y)
X = df.drop(['Employee ID', 'Attrition'], axis=1)
y = df['Attrition']

def encode_categorical_columns(df):
    """
    One-hot encodes categorical columns in a dataframe.

    Args:
        df: The pandas dataframe to encode.

    Returns:
        A new pandas dataframe with categorical columns one-hot encoded.
    """
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

# Assuming df is your dataframe
X = encode_categorical_columns(X)

# Visualize feature distributions after encoding
plt.figure(figsize=(16, 12))
X.hist(bins=30, figsize=(20, 15), layout=(5, 5))
plt.suptitle('Feature Distributions')
plt.show()

# ===Train models===
def apply_models(X, y):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit the encoder on the entire dataset
    y = le.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for class imbalance
    class_counts = np.bincount(y_train)
    print("\n\033[1mClass distribution in training set:\033[0m", class_counts)

    # Apply SMOTE (class imbalance)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Visualize the new class distribution after SMOTE
    class_counts_after_smote = np.bincount(y_train)
    print("\n\033[1mClass distribution after applying SMOTE:\033[0m", class_counts_after_smote)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=np.arange(len(class_counts_after_smote)), y=class_counts_after_smote)
    plt.title('Class Distribution After SMOTE')
    plt.show()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the models
    models = {
        'LogisticRegression': OneVsRestClassifier(LogisticRegression()),
        'SVC': OneVsRestClassifier(SVC()),
        'DecisionTree': OneVsRestClassifier(DecisionTreeClassifier()),
        'RandomForest': OneVsRestClassifier(RandomForestClassifier()),
        'ExtraTrees': OneVsRestClassifier(ExtraTreesClassifier()),
        'AdaBoost': OneVsRestClassifier(AdaBoostClassifier()),
        'GradientBoost': OneVsRestClassifier(GradientBoostingClassifier()),
        'XGBoost': OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        'LightGBM': OneVsRestClassifier(LGBMClassifier(verbose=-1)),
        'CatBoost': OneVsRestClassifier(CatBoostClassifier(verbose=0))
    }

    # Initialize a dictionary to hold the performance of each model
    model_performance = {}

    # Apply each model
    for model_name, model in models.items():
        print(f"\n\033[1mClassification with {model_name}:\033[0m\n{'-' * 30}")

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Convert the numerical predictions back to the original categorical names
        y_test_orig = le.inverse_transform(y_test)
        y_pred_orig = le.inverse_transform(y_pred)

        # Calculate the accuracy and f1 score
        accuracy = accuracy_score(y_test_orig, y_pred_orig)
        f1 = f1_score(y_test_orig, y_pred_orig, average='weighted')

        # Store the performance in the dictionary
        model_performance[model_name] = (accuracy, f1)

        # Print the accuracy score
        print("\033[1m**Accuracy**:\033[0m\n", accuracy)

        # Print the confusion matrix
        cm = confusion_matrix(y_test_orig, y_pred_orig)
        print("\n\033[1m**Confusion Matrix**:\033[0m\n", cm)

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Print the classification report
        print("\n\033[1m**Classification Report**:\033[0m\n", classification_report(y_test_orig, y_pred_orig))

    # Sort the models based on f1 score and pick the top 3
    top_3_models = sorted(model_performance.items(), key=lambda x: x[1][1], reverse=True)[:3]
    print("\n\033[1mTop 3 Models based on Accuracy & F1 Score:\033[0m\n", top_3_models)

    # Extract the model names and classifiers for the top 3 models
    top_3_model_names = [model[0] for model in top_3_models]
    top_3_classifiers = [models[model_name] for model_name in top_3_model_names]

    # Create a Voting Classifier with the top 3 models
    print("\n\033[1mInitializing Voting Classifier with top 3 models...\033[0m\n")
    voting_clf = VotingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)), voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print("\n\033[1m**Voting Classifier Evaluation**:\033[0m\n")
    print("\033[1m**Accuracy**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Confusion Matrix**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Classification Report**:\033[0m\n", classification_report(y_test, y_pred))

    # Create a Stacking Classifier with the top 3 models
    print("\n\033[1mInitializing Stacking Classifier with top 3 models...\033[0m\n")
    stacking_clf = StackingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)))
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    print("\n\033[1m**Stacking Classifier Evaluation**:\033[0m\n")
    print("\033[1m**Accuracy**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Confusion Matrix**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Classification Report**:\033[0m\n", classification_report(y_test, y_pred))

apply_models(X, y)
