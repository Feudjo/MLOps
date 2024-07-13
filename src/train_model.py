# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml import model
# Add code to load in the data.
data = pd.read_csv("data/census.csv")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
                                    training=False,
                                    encoder=encoder,
                                    lb=lb,
                                    label="salary")



#Train and save a model.

rfc =  model.train_model(X_train, y_train)

model.save_model("starter/model/rf.pkl", rfc)
model.save_model("starter/model/encoder.pkl", encoder)

# Inference model
pred = model.inference(rfc, X_test)
_,_,f1_score = model.compute_model_metrics(y_test, pred)

# Compute accuracy
print(f"F1_Score: {f1_score}")

model.compute_slice(rfc, test, "sex", encoder, lb)



