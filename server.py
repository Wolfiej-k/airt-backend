from flask import Flask, request
from flask_tunnel import FlaskTunnel
from flask_cors import CORS

from io import BytesIO
from PIL import Image
import base64

from generate_image import ImageGenerator
generator = ImageGenerator()

app = Flask(__name__)
FlaskTunnel(app)
CORS(app)

def encode(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image = base64.b64encode(buffered.getvalue())
    return image.decode('utf-8')

def decode(image):
    return Image.open(BytesIO(
        base64.b64decode(image)))

@app.route('/diffusion', methods=['POST'])
def diffusion():
    data = request.get_json(force=True)
    if "prompt" in data:
        prompt = data['prompt']
        guidance, steps, count = 7, 20, 4
        images = generator.get_diffusion_output(
            prompt, guidance=guidance, steps=steps, count=count)
        
        parsed = [encode(x) for x in images]        
        return { "output": parsed }
    return { "output": "invalid input" }

@app.route('/merge', methods=['POST'])
def merge():
    data = request.get_json(force=True)
    if "input" in data and "style" in data:
        image = decode(data['input'])
        style = decode(data['style'])
        merged = generator.get_merged_output(image, style)
        
        return { "output": encode(merged) }
    return {"output": "invalid input"}

app.run()