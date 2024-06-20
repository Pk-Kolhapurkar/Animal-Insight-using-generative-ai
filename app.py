from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import warnings
import requests
import gradio as gr

warnings.filterwarnings('ignore')

# Load the pre-trained Vision Transformer model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# API key for the animal information
api_key = '+Rf6S+6HhBwWSlq+rC5wzw==uNM884hcXekpp4tk'  # Replace with your actual API key

def identify_image(image_path):
    """Identify the animal in the image."""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    animal_name = predicted_label.split(',')[0]
    return animal_name

def get_animal_info(animal_name):
    """Get the animal information from the API."""
    api_url = f'https://api.api-ninjas.com/v1/animals?name={animal_name}'
    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    if response.status_code == requests.codes.ok:
        animal_info = response.json()
    else:
        animal_info = {"Error": response.status_code, "Message": response.text}
    return animal_info

def format_animal_info(animal_info):
    """Format the animal information into an HTML table."""
    if "Error" in animal_info:
        return f"Error: {animal_info['Error']} - {animal_info['Message']}"

    if len(animal_info) == 0:
        return "No animal information found."

    animal_data = animal_info[0]
    taxonomy = animal_data.get('taxonomy', {})
    locations = ', '.join(animal_data.get('locations', []))
    characteristics = animal_data.get('characteristics', {})

    table = f"""
    <table border="1" style="width: 100%; border-collapse: collapse;">
        <tr><th colspan="4" style="text-align: center;"><b>Animal Information</b></th></tr>
        <tr><td colspan="4" style="text-align: center;"><b>Animal Name: {animal_data['name']}</b></td></tr>
        <tr><th colspan="4" style="text-align: center;"><b>Taxonomy</b></th></tr>
        <tr>
            <td style="text-align: left;"><b>Kingdom</b></td><td style="text-align: right;">{taxonomy.get('kingdom', 'N/A')}</td>
            <td style="text-align: left;"><b>Phylum</b></td><td style="text-align: right;">{taxonomy.get('phylum', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Class</b></td><td style="text-align: right;">{taxonomy.get('class', 'N/A')}</td>
            <td style="text-align: left;"><b>Order</b></td><td style="text-align: right;">{taxonomy.get('order', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Family</b></td><td style="text-align: right;">{taxonomy.get('family', 'N/A')}</td>
            <td style="text-align: left;"><b>Genus</b></td><td style="text-align: right;">{taxonomy.get('genus', 'N/A')}</td>
        </tr>
        <tr><th colspan="4" style="text-align: center;"><b>Locations</b></th></tr>
        <tr>
            <td colspan="4" style="text-align: center;">{locations}</td>
        </tr>
        <tr><th colspan="4" style="text-align: center;"><b>Characteristics</b></th></tr>
        <tr>
            <td style="text-align: left;"><b>Main Prey</b></td><td style="text-align: right;">{characteristics.get('main_prey', 'N/A')}</td>
            <td style="text-align: left;"><b>Distinctive Feature</b></td><td style="text-align: right;">{characteristics.get('distinctive_feature', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Wingspan</b></td><td style="text-align: right;">{characteristics.get('wingspan', 'N/A')}</td>
            <td style="text-align: left;"><b>Incubation Period</b></td><td style="text-align: right;">{characteristics.get('incubation_period', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Habitat</b></td><td style="text-align: right;">{characteristics.get('habitat', 'N/A')}</td>
            <td style="text-align: left;"><b>Predators</b></td><td style="text-align: right;">{characteristics.get('predators', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Diet</b></td><td style="text-align: right;">{characteristics.get('diet', 'N/A')}</td>
            <td style="text-align: left;"><b>Lifestyle</b></td><td style="text-align: right;">{characteristics.get('lifestyle', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Favorite Food</b></td><td style="text-align: right;">{characteristics.get('favorite_food', 'N/A')}</td>
            <td style="text-align: left;"><b>Type</b></td><td style="text-align: right;">{characteristics.get('type', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Average Clutch Size</b></td><td style="text-align: right;">{characteristics.get('average_clutch_size', 'N/A')}</td>
            <td style="text-align: left;"><b>Slogan</b></td><td style="text-align: right;">{characteristics.get('slogan', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Nesting Location</b></td><td style="text-align: right;">{characteristics.get('nesting_location', 'N/A')}</td>
            <td style="text-align: left;"><b>Age of Molting</b></td><td style="text-align: right;">{characteristics.get('age_of_molting', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Color</b></td><td style="text-align: right;">{characteristics.get('color', 'N/A')}</td>
            <td style="text-align: left;"><b>Skin Type</b></td><td style="text-align: right;">{characteristics.get('skin_type', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Top Speed</b></td><td style="text-align: right;">{characteristics.get('top_speed', 'N/A')}</td>
            <td style="text-align: left;"><b>Lifespan</b></td><td style="text-align: right;">{characteristics.get('lifespan', 'N/A')}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Weight</b></td><td style="text-align: right;">{characteristics.get('weight', 'N/A')}</td>
            <td style="text-align: left;"><b>Length</b></td><td style="text-align: right;">{characteristics.get('length', 'N/A')}</td>
        </tr>
    </table>
    """
    return table

def main_process(image_path):
    """Identify the animal and fetch its information."""
    animal_name = identify_image(image_path)
    animal_info = get_animal_info(animal_name)
    formatted_animal_info = format_animal_info(animal_info)
    return formatted_animal_info

# Define the Gradio interface
def gradio_interface(image):
    formatted_animal_info = main_process(image)
    return formatted_animal_info

# Create the Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath"),
    outputs="html",
    title="Animal Identification and Information",
    description="Upload an image of an animal to get detailed information.",
    allow_flagging="never"  # Disable flagging
)

# Launch the Gradio app
iface.launch()
