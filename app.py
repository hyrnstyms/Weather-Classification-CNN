import gradio as gr
import torch
from torchvision import transforms
from model import WeatherCNN
from PIL import Image
import os

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_model.pth")
GRAPH_PATH = os.path.join(BASE_DIR, "models", "loss_graph.png") # Grafik yolu
EXAMPLES_PATH = os.path.join(BASE_DIR, "examples")

CLASSES = {0: 'Bulutlu ☁️', 1: 'Yağmurlu 🌧️', 2: 'Güneşli ☀️', 3: 'Gündoğumu 🌅'}
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
if os.path.exists(EXAMPLES_PATH):
    files = [os.path.join(EXAMPLES_PATH, f) for f in os.listdir(EXAMPLES_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    example_images = [[f] for f in files]

# 4. Gradio BLOCKS Arayüzü (Sekmeli Yapı)
with gr.Blocks(title="🌤️ Hava Durumu Analiz Sistemi") as demo:
    gr.Markdown("# 🌤️ Hava Durumu Analiz ve Eğitim Raporu")
    
    with gr.Tabs():
        # --- SEKME 1: TAHMİN ---
        with gr.TabItem("📸 Hava Durumu Tahmini"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Fotoğraf Yükle")
                    predict_btn = gr.Button("Analiz Et", variant="primary")
                with gr.Column():
                    lbl_output = gr.Label(num_top_classes=4, label="Sonuçlar")
            
            # Buton aksiyonu
            predict_btn.click(predict_weather, inputs=img_input, outputs=lbl_output)
            
            # Örnekler
            if example_images:
                gr.Examples(examples=example_images, inputs=img_input)

        # --- SEKME 2: EĞİTİM GRAFİĞİ ---
        with gr.TabItem("📈 Eğitim Grafiği (Loss)"):
            gr.Markdown("Modelin eğitim sürecindeki hata (loss) değişim grafiği aşağıdadır.")
            if os.path.exists(GRAPH_PATH):
                # Grafiği göster
                gr.Image(GRAPH_PATH, label="Loss Grafiği", interactive=False)
            else:
                gr.Markdown("⚠️ **Grafik bulunamadı.** Lütfen önce `train.py` dosyasını çalıştırarak modeli eğitin.")

if __name__ == "__main__":
    demo.launch()