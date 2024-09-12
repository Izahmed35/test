import os
import replicate

# Fetch the REPLICATE_API_TOKEN from the environment variables
replicate_token = os.getenv("REPLICATE_API_TOKEN")

if not replicate_token:
    raise ValueError("Please set the REPLICATE_API_TOKEN environment variable")

# Run the model using Replicate's API
output = replicate.run(
    "izahmed35/izmodel:0e9f080f7e29f6800cf3ba745587fdf21824d33184c8f5264976fddfa02d135c",
    input={
        "model": "dev",
        "lora_scale": 1,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "guidance_scale": 3.5,
        "output_quality": 90,
        "prompt_strength": 0.8,
        "extra_lora_scale": 1,
        "num_inference_steps": 28
    }
)
print(output)
