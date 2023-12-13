from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Preparing train data
train_data = pd.read_csv("../data/train_set.csv")
train_data = train_data.sample(frac=0.1)
train_data = pd.DataFrame({'text': train_data['text'],
                           'labels': train_data['label']})

# Preparing eval data
eval_data = pd.read_csv("../data/test_set.csv")
eval_data = pd.DataFrame({'text': eval_data['text'],
                           'labels': eval_data['label']})



# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_data)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_data)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])