import pickle
with open("model_eqs.pkl", "rb") as f:
        eqs_model = pickle.load(f)

print("EQS Prediction:", eqs_model.predict([[0, 2, 100]]))