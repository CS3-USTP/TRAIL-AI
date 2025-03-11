from sentence_transformers import CrossEncoder

# Load the model
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

assistant = """
tell me more about that
"""

while True:
    user_message = input("Enter your message: ")
    
    if user_message == "exit":
        break
    
    score = model.predict([(assistant, user_message)])
    
    print("Score: ", score) 
    
    
    
    
    
    
    
    
    
    
    
    