import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_model(X, Y, test_size=0.2):
    """
    Trains a logistic regression model using provided data.

    Args:
        X (list of tuples): Feature dataset [(int, int, int), ...].
        y (list of bool): Binary target labels [True, False, ...].
        test_size (float): Proportion of data to use for testing.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
    """
    # Convert inputs to NumPy arrays
    X = np.array(X)
    Y = np.array(Y, dtype=int)  # Convert boolean to int (True → 1, False → 0)

    # Split into training and test sets
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Evaluate the model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy:.2f}")

    return model


def predict(model, values_array):
    """
    Predicts True or False for an array of 3-element tuples.

    Args:
        model: Trained logistic regression model.
        values_array (list of tuples): Each tuple contains three integers.

    Returns:
        list: List of boolean values (True for 1, False for 0).
    """
    values_array = np.array(values_array)  # Convert to NumPy array
    predictions = model.predict(values_array)  # Predict labels
    return [bool(pred) for pred in predictions][0]  # Convert to True/False

