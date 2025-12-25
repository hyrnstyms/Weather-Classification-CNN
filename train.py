import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import WeatherCNN
import os

# --- AYARLAR ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10  
DATA_PATH = "./dataset" # Resimlerin olduğu klasör
MODEL_SAVE_PATH = "./models/weather_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Veri Ön İşleme (Resimleri modele uygun hale getirme)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 224x224 ister
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Veriyi Yükleme
    if not os.path.exists(DATA_PATH):
        print(f"HATA: '{DATA_PATH}' klasörü bulunamadı! Lütfen resimleri ekleyin.")
        return

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    
    # Veriyi %80 Eğitim, %20 Test olarak ayırıyoruz
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Eğitim Cihazı: {DEVICE}")
    print(f"Toplam Resim: {len(full_dataset)} | Eğitim: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Sınıflar: {full_dataset.classes}")

    # 3. Modeli Başlatma
    model = WeatherCNN(num_classes=len(full_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Eğitim Döngüsü
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad() # Gradyanları sıfırla
            outputs = model(images) # Tahmin yap
            loss = criterion(outputs, labels) # Hatayı hesapla
            loss.backward() # Geriye yayılım (Backpropagation)
            optimizer.step() # Ağırlıkları güncelle
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp (Loss): {running_loss/len(train_loader):.4f}")

    # 5. Modeli Kaydetme
    if not os.path.exists("./models"):
        os.makedirs("./models")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model başarıyla kaydedildi: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()