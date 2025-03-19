import json
import torch
from sentence_transformers import CrossEncoder
from joblib import dump, load
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



def main():
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    semantic_model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-base",
        trust_remote_code=True,
        device=device
    )

    # open the dataset
    with open("./data/dataset_1.json") as f:
        dataset = json.load(f)

        output = []
        for data in dataset:
            
            results = semantic_model.predict([(data["premise"], data["hypothesis"])])
            values = results[0].tolist()
            
            output.append({
                "values": values,
                "coherence": data["coherence"]
            })

        # save the result to json file
        with open("./out/result_1.json", "w") as f:
            json.dump(output, f, indent=4)                
              
              
    # open the result file
    with open("./out/result_1.json") as f:
        results = json.load(f)
        
        X = []
        Y = []
        for result in results:
            X.append(result["values"])
            Y.append(result["coherence"])
            
        logistic_model = train_logistic_model(X, Y, 0.2)
        dump(logistic_model, "./out/model.joblib")
        
        print("Model trained and saved successfully.")

        # premise: what is cs3 what is are they hypthesis: what are they
        example_inputs = [(-3.4040327072143555, -3.5754246711730957, 5.186447620391846)]
        results = predict(logistic_model, example_inputs)

        print(f"Predictions for {example_inputs}: {results}")


if __name__ == "__main__":
    main()