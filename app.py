import os
import replicate
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Home route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission and generate the image
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    model_version = "izahmed35/izmodel:0e9f080f7e29f6800cf3ba745587fdf21824d33184c8f5264976fddfa02d135c"

    try:
        # Fetch the REPLICATE_API_TOKEN from environment variables
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            return "API token is not set. Please set the REPLICATE_API_TOKEN environment variable."

        # Run the model
        output = replicate.run(
            model_version,
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
        # Get the image URL from output
        image_url = output[0]

        # Redirect to display the image
        return redirect(url_for('show_image', image_url=image_url))

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Route to display the generated image
@app.route('/image')
def show_image():
    image_url = request.args.get('image_url')
    return f'<h1>Generated Image</h1><img src="{image_url}" alt="Generated Image"/>'

if __name__ == '__main__':
    app.run(debug=True)
