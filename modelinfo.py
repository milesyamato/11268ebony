from huggingface_hub import hf_api

# Specify the model name
model_name = 'psmathur/orca_mini_v2_13b'

# Get model information
model_info = hf_api.model_info(model_name)

# Get and print all file names in the model repository
for file in model_info.siblings:
    print(f"File name: {file.rfilename}")

# Get and print model configuration
print(model_info.config)
