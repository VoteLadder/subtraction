import os
from flask import Flask, render_template, request, send_file
import torch
from torchvision import transforms
from PIL import Image
import io
from model import UNetGenerator  # Assuming UNetGenerator is defined in model.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load model once when the app starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetGenerator().to(device)
model.load_state_dict(torch.load('9_14_best_final.pth', map_location=device))
model.eval()

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def save_output(output):
    # Convert the output tensor to a PIL Image
    output_image = transforms.ToPILImage()(output.squeeze().cpu())
    
    # Save to BytesIO object
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the image
            input_image = load_and_preprocess_image(file_path).to(device)
            with torch.no_grad():
                output = model(input_image)
            
            # Save and return the processed image
            output_bytes = save_output(output)
            return send_file(output_bytes, mimetype='image/png')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)