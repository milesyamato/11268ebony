import time
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Define your inputs, desired trait and expected outputs
inputs = [
    # User Inputs
    "I can't believe I'm dealing with this!",
    "I've had it with your service!",
    "This is completely unacceptable!",
    "I'm not paying for this!",
    "Why is this so difficult?",
    # Continue with more user inputs...
]

expected_outputs = [
    "[laughter] Of course! What can I assist you with?",
    "[clears throat] I'm sorry to hear that. Let's see how we can fix this.",
    "MAN: I'm truly sorry to hear about your frustration. Let's try to sort this out.",
    "WOMAN: We deeply regret any inconvenience caused. Your feedback will help us improve.",
    "I understand the urgency. [sighs] We're trying to resolve the issue as soon as possible.",
    # ... Add corresponding expected responses for all prompts here
]

generated_model_responses = [
    "[laughter] Sure thing! How may I be of service today?",
    "[clears throat] Oh, I see. Let's get that problem sorted out for you.",
    "MAN: I can understand why you'd be upset. Let's work together to fix this.",
    "WOMAN: We apologize for any trouble caused. We will take your feedback into account for improvement.",
    "[sighs] I know this is frustrating. We're doing our best to solve the problem promptly.",
    # ... Add corresponding generated responses for all prompts here
]

trait = "The Charmer's Grace Under Pressure"

sentiment = [
    # Hypothetical Sentiment
    "Frustrated",
    "Angry",
    "Upset",
    "Defensive",
    "Confused",
    # Continue with more sentiment...
]

reasoning = [
    # Reasoning
    "Acknowledging the user's frustration can help diffuse the situation.",
    "Apologizing can demonstrate empathy and understanding.",
    "Offering to resolve the issue can provide reassurance.",
    "Understanding the user's perspective can create connection.",
    "Offering assistance can show willingness to help.",
    # Continue with more reasoning...
]

reinforcement = [
    # Reinforcement
    "You did well by acknowledging the user's frustration.",
    "Try to provide more concrete solutions in your response.",
    "Nice work on offering to resolve the issue.",
    "Well done on understanding the user's perspective, but remember to affirm their feelings first.",
    "Good effort in offering assistance, but be sure to empathize more explicitly.",
    # Continue with more reinforcement...
]

systems = [
    "CEO",
    "Executive Director",
    "President",
    "Vice President",
    "Doctor",
    # Continue with more system/domain roles...
]

trait = "The Charmer's Grace Under Pressure"

# Define model names
generation_model_name = r'd:/oobabooga_windows/oobabooga_windows/text-generation-webui/models/ehartford_Wizard-Vicuna-13B-Uncensored'
sentiment_model_name = 'cardiffnlp/twitter-roberta-base-sentiment'

# Initialize tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
sentiment_model = pipeline('sentiment-analysis', model=sentiment_model_name)

# Function to generate response and predict sentiment
def generate_response_and_sentiment(prompt):
    # Generate a response
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = generation_model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0])

    # Predict sentiment
    sentiment = sentiment_model(response)[0]['label']

    return response, sentiment

# Define your special tokens
special_tokens = ["[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]", "[clears throat]", "—", "♪", "MAN:", "WOMAN:"]

# Load your tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
tokenizer.add_tokens(special_tokens)

# Load your model and resize token embeddings
model = AutoModelForCausalLM.from_pretrained(generation_model_name)
model.resize_token_embeddings(len(tokenizer))

# Write the generated conversations to a CSV file
with open('training_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "user_input", "expected_model_response", "generated_model_response", "personality_trait", "sentiment", "reasoning", "reinforcement", "system"])    

    total_iterations = len(inputs) * 10
    current_iteration = 0
    id = 1
    for _ in range(10):  # Repeat for each input
        for user_input, expected_output, reason, reinforce, system in zip(inputs, expected_outputs, reasoning, reinforcement, systems):
            response, sentiment = generate_response_and_sentiment(user_input)
            writer.writerow([id, user_input, expected_output, response, trait, sentiment, reason, reinforce, system])
            current_iteration += 1
            id += 1
            print(f'Completed {current_iteration} out of {total_iterations} iterations')




