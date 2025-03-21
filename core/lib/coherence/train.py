
import json
import torch
from sentence_transformers import CrossEncoder
from joblib import dump
from core.lib.coherence.utils import train_logistic_model, predict

def main():
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    semantic_model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-base",
        trust_remote_code=True,
        device=device
    )

    # open the dataset
    with open("core/lib/coherence/data/coherence.json") as f:
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
        with open("core/lib/coherence/out/results.json", "w") as f:
            json.dump(output, f, indent=4)                
              
              
    # open the result file
    with open("core/lib/coherence/out/results.json") as f:
        results = json.load(f)
        
        X = []
        Y = []
        for result in results:
            X.append(result["values"])
            Y.append(result["coherence"])
            
        logistic_model = train_logistic_model(X, Y, 0.2)
        dump(logistic_model, "core/lib/coherence/out/model.joblib")
        
        print("Model trained and saved successfully.")

        # premise: what is cs3 what is are they hypthesis: what are they
        example_inputs = [(-3.4040327072143555, -3.5754246711730957, 5.186447620391846)]
        results = predict(logistic_model, example_inputs)

        print(f"Predictions for {example_inputs}: {results}")


if __name__ == "__main__":
    main()