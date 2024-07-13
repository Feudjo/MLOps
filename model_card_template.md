# Model Card

## Model Details
Model Name: Random Forest Classifier
Model Type: Supervised Classification
Algorithm: Random Forest
Framework: scikit-learn
Version: 1.2.2
The model is a random forest classifier that uses the default hyperparameters
in scikik-learn 1.2.2.
## Intended Use
This model is intended for classification tasks where the goal is to
determine weather the income of an individual is greather than or less than 50k.
## Training Data
The data was obtained from the uci machine learning repository,
https://archive.ics.uci.edu/dataset/20/census+income .

For training purposes, the categorical features were transformed using a One Hot Encoder,
and the labels were processed wih a label binarizer.

The original dataset consisted of 32561 rows, and a 80/20 split was used to break
this into train and test set where 80% was used for training and the rest for testing.
## Evaluation Data
This data consist of 20% of the original dataset, and the features were  processed
in the same way as the training.
## Metrics
Although three metrics were computed (f1 score, precision, recall), the model
was evaluated using the f1 score. The present value is 0.82.
## Ethical Considerations
The model may exhibit biases present in the training data. Furthermore, feature
importance score can provide insights into how the model makes decisions.

## Caveats and Recommendations
- It is crucial to constantly evaluate the model for fairness.
- The performance of the model might degrade on data significantly different from the
  training data. Regular training and evaluation on new data is recommended.