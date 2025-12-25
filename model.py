import torch
import torch.nn as nn
import torch.nn.functional as F

class WeatherCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(WeatherCNN, self).__init__()
        
        # --- KATMAN 1 (Gözler) ---
        # Girdi: 3 kanal (RGB), Çıktı: 32 özellik haritası
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Boyutu yarıya düşürür (224 -> 112)
        
        # --- KATMAN 2 (Burun/Kulak) ---
        # Girdi: 32, Çıktı: 64 özellik
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # pool tekrar çalışacak (112 -> 56)
        
        # --- KATMAN 3 (Yüz/Şekil) ---
        # Girdi: 64, Çıktı: 128 özellik
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # pool tekrar çalışacak (56 -> 28)
        
        # --- KARAR VERME KATMANI (Beyin) ---
        # Flatten: 128 tane 28x28'lik haritayı düz bir vektöre çevirir
        self.fc1 = nn.Linear(128 * 28 * 28, 512) # 512 nöronlu gizli katman
        self.dropout = nn.Dropout(0.5) # Ezberlemeyi önlemek için nöronların yarısını kapat
        self.fc2 = nn.Linear(512, num_classes) # Sonuç katmanı (4 sınıf)

    def forward(self, x):
        # 1. Katman: Evrişim -> Aktivasyon (ReLU) -> Havuzlama (Pool)
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2. Katman
        x = self.pool(F.relu(self.conv2(x)))
        
        # 3. Katman
        x = self.pool(F.relu(self.conv3(x)))
        
        # Düzleştirme (Matrix -> Vektör)
        x = x.view(-1, 128 * 28 * 28)
        
        # Sınıflandırma
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x