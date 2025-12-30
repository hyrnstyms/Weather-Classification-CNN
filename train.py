import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import WeatherCNN
import os
import matplotlib.pyplot as plt # Grafik çizimi için ekledik

# --- AYARLAR ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10  
DATA_PATH = "./dataset" 
MODEL_SAVE_PATH = "./models/weather_model.pth"
GRAPH_SAVE_PATH = "./models/loss_graph.png" # Grafik kayıt yolu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Veri Ön İşleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Veriyi Yükleme
    if not os.path.exists(DATA_PATH):
        print(f"HATA: '{DATA_PATH}' klasörü bulunamadı!")
        return

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Eğitim Cihazı: {DEVICE}")

    # 3. Modeli Başlatma
    model = WeatherCNN(num_classes=len(full_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- LOSS TAKİBİ İÇİN LİSTE ---
    loss_history = [] 

    # 4. Eğitim Döngüsü
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Ortalama kaybı hesapla ve listeye ekle
        epoch_loss = running_loss/len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp (Loss): {epoch_loss:.4f}")

    # 5. Modeli ve Grafiği Kaydetme
    if not os.path.exists("./models"):
        os.makedirs("./models")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model kaydedildi: {MODEL_SAVE_PATH}")

    # --- GRAFİK ÇİZME VE KAYDETME ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), loss_history, label='Eğitim Kaybı (Training Loss)', color='red', marker='o')
    plt.title('Model Eğitim Kaybı Grafiği')
    plt.xlabel('Epoch Sayısı')
    plt.ylabel('Loss (Kayıp)')
    plt.grid(True)
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"Grafik kaydedildi: {GRAPH_SAVE_PATH}")
    plt.close()

if __name__ == "__main__":
    train()