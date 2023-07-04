import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import csv
import pandas as pd

# Define model names
generation_model_name = r'd:/oobabooga_windows/oobabooga_windows/text-generation-webui/models/ehartford_Wizard-Vicuna-13B-Uncensored'
sentiment_model_name = 'cardiffnlp/twitter-roberta-base-sentiment'

# Initialize tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
sentiment_model = pipeline('sentiment-analysis', model=sentiment_model_name)

# Define your inputs and expected outputs
data = {
    "user_input": [
        "I'm really angry about this situation.", 
        "Your services are terrible.", 
        "I need a solution right now!",
        # add more dialogues here
    ], 
    "expected_model_response": [
        "[sighs] I understand your frustration, and I'm here to help.", 
        "[clears throat] I'm sorry to hear that. How can we improve?",
        "[excited] I'm on it! Let's see what we can do.", 
        # add more responses here
    ]
}

df = pd.DataFrame(data)

# save to csv file
df.to_csv("training_data.csv", index=False)

# Generate responses and predict sentiment
for idx, row in df.iterrows():
    response, sentiment = generate_response_and_sentiment(row['user_input'])
    df.loc[idx, 'generated_model_response'] = response
    df.loc[idx, 'sentiment'] = sentiment

# Save the updated DataFrame
df.to_csv("training_data_with_responses.csv", index=False)
