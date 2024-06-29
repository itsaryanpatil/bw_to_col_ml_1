from flask import Flask, request, render_template, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image

if __name__ == '__main__':
    from copy_of_colorize import ColorizationNet, rgb_to_gray  # Import your model and necessary functions
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    app = Flask(__name__, template_folder='templates', static_folder='static')  # Adjusted to include 'static'

    # Set device to use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your pre-trained model
    model = ColorizationNet().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    # Define transformations for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match your model's expected sizing
        transforms.ToTensor()
    ])

    # Function to preprocess the uploaded image
    def preprocess_image(image):
        # Convert image to grayscale
        gray_img = image.convert('L')
        # Apply transformations
        img_tensor = transform(gray_img).unsqueeze(0)  # Add batch dimension
        return img_tensor.to(device)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return render_template('index.html', message='No file part')
            
            file = request.files['file']
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                return render_template('index.html', message='No selected file')
            
            # If a file is uploaded
            if file:
                file.save('static/normal_img.jpg')
                try:
                    # Open and preprocess the image
                    img = Image.open(file)
                    img_tensor = preprocess_image(img)
                    
                    # Perform colorization
                    with torch.no_grad():
                        colorized_tensor = model(img_tensor)
                    
                    # Convert tensor to image
                    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
                    
                    # Save the colorized image (optional)
                    colorized_img.save('static/colorized_image.jpg')  # Save to 'static' directory
                    
                    # Display the colorized image
             
                    return render_template('index.html', 
                                       bw_image=file,  # Path to black-and-white image
                                       colorized_image='static/colorized_image.jpg')  # Path to colorized image
                
                except Exception as e:
                    return render_template('index.html', message='Error processing image: {}'.format(str(e)))
        
        return render_template('index.html')

    app.run(debug=True)