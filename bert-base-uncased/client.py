import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer

# URL of the Triton server
url = "localhost:8000"
model_name = "bert-base-uncased"

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Sample input text with a masked token
texts = ["Paris is the capital of  [MASK].", "The olympics is a [MASK] festival."]

# Tokenize the input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np", max_length=128)

# Convert inputs to int32
input_ids = inputs['input_ids'].astype(np.int32)
attention_mask = inputs['attention_mask'].astype(np.int32)
token_type_ids = inputs['token_type_ids'].astype(np.int32)

# Create Triton client
client = httpclient.InferenceServerClient(url=url)

# Prepare input tensors
input_ids_infer = httpclient.InferInput("input_ids", input_ids.shape, "INT32")
input_ids_infer.set_data_from_numpy(input_ids)

attention_mask_infer = httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
attention_mask_infer.set_data_from_numpy(attention_mask)

token_type_ids_infer = httpclient.InferInput("token_type_ids", token_type_ids.shape, "INT32")
token_type_ids_infer.set_data_from_numpy(token_type_ids)

# Prepare output tensor
output = httpclient.InferRequestedOutput("logits")

# Make the request to the Triton server
results = client.infer(model_name, inputs=[input_ids_infer, attention_mask_infer, token_type_ids_infer], outputs=[output])

# Get the results
logits = results.as_numpy("logits")

# Debug: Print the shape of the logits array
print("Logits shape:", logits.shape)

# Find the positions of the [MASK] token in the input
mask_token_index = np.where(input_ids == tokenizer.mask_token_id)

# Convert logits to token predictions for the masked positions
predicted_token_ids = np.argmax(logits, axis=-1)

# Extract the predicted tokens for the [MASK] positions
masked_predictions = [predicted_token_ids[batch_idx, mask_pos] for batch_idx, mask_pos in zip(*mask_token_index)]

# Decode the predicted token ids back to tokens
decoded_predictions = [tokenizer.decode([pred]) for pred in masked_predictions]

# Print the results
for text, masked_prediction in zip(texts, decoded_predictions):
    print(f"Text: {text} -> Predicted token: {masked_prediction}")
