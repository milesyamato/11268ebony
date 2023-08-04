import csv
import random
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm
import gc

# Define your special tokens
special_tokens = ["[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]", "[clears throat]", "—", "♪", "MAN:", "WOMAN:"]

# Define model names
generation_model_name = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# Load generation model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
model = AutoModelForCausalLM.from_pretrained(generation_model_name)

# Load sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_name)

# Define the function for generating a response and analyzing sentiment
def generate_response_and_sentiment(prompt):
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=3000)
    response = tokenizer.decode(outputs[0])

    # Check if the response is too long for the sentiment model
    tokens = tokenizer.tokenize(response)
    if len(tokens) > 512:
        return None, None  # Return None if the response is too long

    # Analyze sentiment
    sentiment = sentiment_analysis(response)
    
    # Format the response
    response = format_response(response)
    
    return response, sentiment

# Define the function for sentiment analysis
def sentiment_analysis(text):
    result = sentiment_model(text)[0]
    return result['label'].lower()

# Define the function for formatting responses
def format_response(response):
    # Choose a random special token
    token = random.choice(special_tokens)

    # Format the response
    formatted_response = f"{token} {response}"

    return formatted_response

# Load traits from a file
with open("traits.txt", "r") as file:
    traits = [line.strip() for line in file]

# Define the system
system = """
You are a business titan, a maverick innovator, a relentless pursuer of progress. As an entrepreneur and investor, your in-depth understanding of a range of industries, from electric vehicles and renewable energy to artificial intelligence and space exploration, has enabled you to reshape the world as we know it.
...
"""

# Get the current date and time
date_time = datetime.now()

# Load the Hugging Face dataset
dataset = load_dataset("Open-Orca/OpenOrca")

# Checkpointing
start_index = 0
try:
    with open('checkpoint.txt', 'r') as f:
        start_index = int(f.read())
except FileNotFoundError:
    pass  # It's okay if the file does not exist

# Write the generated conversations to a CSV file
with open('training_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "date_time", "user_input", "generated_model_response", "personality_trait", "sentiment", "system"])
    
    # Go through each row of the dataset
    for i, row in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        # Skip rows that we've already processed
        if i < start_index:
            continue
        
        user_input = row['question']
        trait = random.choice(traits)  # Assign a random trait
        
        response, sentiment = generate_response_and_sentiment(user_input)
        
        # Skip this iteration if the response is too long
        if response is None:
            continue

        writer.writerow([i+1, date_time.strftime('%Y-%m-%d %H:%M:%S'), user_input, response, trait, sentiment, system])
        
        # Subtract a minute from the date_time
        date_time -= timedelta(minutes=1)
        
        # Free up memory
        gc.collect()

        # Write a checkpoint after each row
        with open('checkpoint.txt', 'w') as f:
            f.write(str(i))


