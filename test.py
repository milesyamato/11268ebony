from transformers import AutoConfig

# Load the configuration from Hugging Face's model hub
config = AutoConfig.from_pretrained("TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ")

# Print the config to see the details
print(config)
