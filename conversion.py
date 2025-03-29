from PIL import Image
import base64
from io import BytesIO
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Gemini API settings
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Load from environment variable if set

# Load the image
image_path = "/Users/mohulshukla/Desktop/coco/saved_drawings/drawing_20250329_005231.png"
image = Image.open(image_path)

# Convert the image to bytes and then to base64
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Set up the client with your API key
client = genai.Client(api_key=GEMINI_API_KEY)

# Prepare the prompt and image data
prompt = "Enhance this sketch into a detailed image."
contents = [
    {"text": prompt},
    {"inlineData": {
        "mimeType": "image/jpeg",
        "data": img_str
    }}
]

# Set the model and configuration
model = "gemini-2.0-flash-exp-image-generation"
config = types.GenerateContentConfig(response_modalities=['Text', 'Image'])

# Generate the enhanced image
response = client.models.generate_content(
    model=model,
    contents=contents,
    config=config
)

print("RESPONSE: ", response)
# Process the response
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print("FUCK")
        print(part.text)
    elif part.inline_data is not None:
        # Save the generated image
        image = Image.open(BytesIO((part.inline_data.data)))
        image.save('gemini-native-image.png')
        image.show()
