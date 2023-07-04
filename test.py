from transformers import AutoConfig

# Load the configuration from Hugging Face's model hub
config = AutoConfig.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ")

# Print the config to see the details
print(config)
