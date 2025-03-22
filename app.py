# IMPORTANT LIBRARIES
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score  # For model evaluation
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # For decision tree models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor  # For ensemble models
from sklearn.linear_model import LogisticRegression, LinearRegression  # For linear models
from sklearn.neural_network import MLPClassifier  # For neural networks
from sklearn.svm import SVC, SVR  # For support vector machines
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced visualizations
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.impute import SimpleImputer  # For handling missing values
from xgboost import XGBClassifier, XGBRegressor  # For XGBoost models


# Function to train and evaluate the model
def train_and_evaluate(model, X_train, X_test, y_train, y_test, feature_names=None, is_regression=False):
    """
    Trains the model and evaluates its performance using various metrics and visualizations.
    
    Parameters:
        model: The machine learning model to train.
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
        feature_names: Names of the features (for visualization purposes).
        is_regression: Boolean indicating whether the task is regression or classification.
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    if is_regression:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # R-squared Score
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        # Scatter plot of actual vs predicted values
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)  # Plot actual vs predicted values
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")  # Diagonal line for reference
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()

    else:
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        print(f"Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Plot confusion matrix
        plt.title("Confusion Matrix")
        plt.show()

        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:  # Check if the problem is binary classification
            y_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
            fpr, tpr, _ = roc_curve(y_test, y_prob)  # Compute ROC curve
            roc_auc = auc(fpr, tpr)  # Compute AUC score
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")  # Plot ROC curve
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal line for reference
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.show()

    # Feature Importance (for tree-based models)
    if hasattr(model, "feature_importances_"):  # Check if the model has feature importance attribute
        feature_importances = model.feature_importances_  # Get feature importances
        if feature_names is not None:
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})  # Create a DataFrame
            importance_df = importance_df.sort_values(by="Importance", ascending=False)  # Sort by importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df)  # Plot feature importance
            plt.title("Feature Importance")
            plt.show()

    # Histogram of Feature Distributions
    if feature_names is not None:
        X_train_df = pd.DataFrame(X_train, columns=feature_names)  # Convert training data to DataFrame
        X_train_df.hist(figsize=(12, 10), bins=20)  # Plot histograms of features
        plt.suptitle("Histograms of Feature Distributions")
        plt.show()

    # Correlation Heatmap
    if feature_names is not None:
        X_train_df = pd.DataFrame(X_train, columns=feature_names)  # Convert training data to DataFrame
        X_train_df["Target"] = y_train  # Add target column
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")  # Plot correlation heatmap
        plt.title("Correlation Heatmap")
        plt.show()

    return y_pred


# Main function
def main():
    """
    Main function to handle user input, data preprocessing, model training, and evaluation.
    """
    # Step 1: Upload Dataset
    file_path = input("Enter the path to your dataset (CSV file): ")  # Get file path from user
    data = pd.read_csv(file_path)  # Load dataset
    print("Dataset Columns:", data.columns.tolist())  # Display column names

    # Step 2: Select Algorithm
    print("Select Algorithm:")
    print("1. Decision Tree (Classification)")
    print("2. Random Forest (Classification)")
    print("3. Logistic Regression (Classification)")
    print("4. Artificial Neural Network (ANN) (Classification)")
    print("5. Linear Regression (Regression)")
    print("6. Support Vector Machine (SVM) (Classification)")
    print("7. Gradient Boosting Regression (Regression)")
    print("8. XGBoost (Classification)")
    print("9. XGBoost (Regression)")
    algo_choice = int(input("Enter the number corresponding to your choice: "))  # Get algorithm choice

    # Step 3: Select X and Y Columns
    X_columns = input("Enter the names of the feature columns (comma-separated): ").split(",")  # Get feature columns
    X_columns = [col.strip() for col in X_columns]  # Remove extra spaces
    X_columns = [col if col in data.columns else col.title() for col in X_columns]  # Handle case sensitivity
    not_found_cols = [col for col in X_columns if col not in data.columns]  # Check for invalid columns
    if not_found_cols:
        print(f"Error: The following columns were not found in the dataset: {not_found_cols}")
        print("Please check the spelling and try again.")
        return

    y_column = input("Enter the name of the target column: ")  # Get target column
    if y_column not in data.columns:
        y_column = y_column.title()  # Handle case sensitivity
        if y_column not in data.columns:
            print(f"Error: Target column '{y_column}' not found in the dataset.")
            print("Please check the spelling and try again.")
            return

    X = data[X_columns]  # Select feature columns
    y = data[y_column]  # Select target column

    # Drop rows with NaNs in the target variable (y) before discretization
    data = data.dropna(subset=[y_column])
    X = data[X_columns]
    y = data[y_column]

    # Convert categorical features to numerical using Label Encoding
    for column in X.columns:
        if X[column].dtype == "object":  # Check if the column is categorical
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])  # Encode categorical values

    # Check if target variable is continuous and convert to discrete if necessary
    is_regression = algo_choice in [5, 7, 9]  # Check if the task is regression
    if not is_regression and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        print("Target variable appears to be continuous. Converting to discrete using binning/thresholding.")
        try:
            y = pd.qcut(y, q=2, labels=[0, 1])  # Convert continuous target to binary
        except ValueError:
            print("Warning: Could not create unique bins using qcut. The target variable might have many similar values.")
            print("Consider using a different discretization method or adjusting the 'q' parameter.")

    # Impute missing values in X using SimpleImputer
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with the mean
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the Model
    if algo_choice == 1:
        model = DecisionTreeClassifier()  # Decision Tree for classification
    elif algo_choice == 2:
        model = RandomForestClassifier()  # Random Forest for classification
    elif algo_choice == 3:
        model = LogisticRegression()  # Logistic Regression for classification
    elif algo_choice == 4:
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)  # Neural Network for classification
    elif algo_choice == 5:
        model = LinearRegression()  # Linear Regression
    elif algo_choice == 6:
        model = SVC(probability=True)  # Support Vector Machine for classification
    elif algo_choice == 7:
        model = GradientBoostingRegressor()  # Gradient Boosting for regression
    elif algo_choice == 8:
        model = XGBClassifier()  # XGBoost for classification
    elif algo_choice == 9:
        model = XGBRegressor()  # XGBoost for regression
    else:
        print("Invalid choice!")
        return

    # Step 5: Evaluate and Plot
    train_and_evaluate(model, X_train, X_test, y_train, y_test, feature_names=X_columns, is_regression=is_regression)


if __name__ == "__main__":
    main()  # Run the main function