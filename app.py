import gradio as gr
import torch
from torchvision import transforms
from model import WeatherCNN
from PIL import Image
import os

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_model.pth")
EXAMPLES_PATH = os.path.join(BASE_DIR, "examples")

CLASSES = {
    0: 'Bulutlu ☁️', 
    1: 'Yağmurlu 🌧️', 
    2: 'Güneşli ☀️', 
    3: 'Gündoğumu 🌅'
}

DEVICE = torch.device("cpu")

# 1. Modeli Yükle
model = WeatherCNN(num_classes=4) 
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
else:
    print(f"UYARI: Model bulunamadı: {MODEL_PATH}")

# 2. Tahmin Fonksiyonu
def predict_weather(image):
    if image is None: return None
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

# 3. Örnek Resimleri Bul
example_images = []
if os.path.exists(EXAMPLES_PATH) and os.path.isdir(EXAMPLES_PATH):
    files = os.listdir(EXAMPLES_PATH)
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(EXAMPLES_PATH, file)
            example_images.append([full_path])
else:
    try:
        os.makedirs(EXAMPLES_PATH, exist_ok=True)
    except:
        pass

# 4. Arayüz 
interface = gr.Interface(
    fn=predict_weather,
    inputs=gr.Image(type="pil", label="Fotoğraf Yükle"),
    outputs=gr.Label(num_top_classes=4, label="Sonuçlar"),
    examples=example_images if example_images else None,
    title="🌤️ Hava Durumu Analiz Sistemi",
)

if __name__ == "__main__":
    interface.launch()