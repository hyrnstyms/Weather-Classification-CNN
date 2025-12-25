🌤️ Weather-Classification-CNN
Bu proje, görüntü işleme (Computer Vision) teknikleri kullanılarak dış ortam görüntülerinden anlık hava durumunu (Güneşli, Bulutlu, Yağmurlu, Gündoğumu) tespit eden bir Derin Öğrenme (Deep Learning) uygulamasıdır.

Projede hazır modeller (Transfer Learning) yerine, mimariyi tam olarak kontrol edebilmek ve öğrenme sürecini analiz etmek amacıyla Özgün (Custom) CNN Mimarisi tasarlanmış ve PyTorch ile geliştirilmiştir.

📋 İçindekiler
Proje Hakkında

Veri Seti

Kullanılan Yöntem ve Mimari

Kurulum

Kullanım

Dosya Yapısı

1. Proje Hakkında
Problem: Geleneksel hava durumu tahminleri (radar ve uydu) geniş ölçekli tahminler yapar ancak yerel (mikro-iklim) durumları anlık olarak görselleştiremez. Pahalı sensörler olmadan, sadece görsel veri ile hava durumunu anlamak IoT ve Akıllı Şehirler için kritik bir ihtiyaçtır.

Amaç: Kamera görüntülerini analiz ederek hava durumunu sınıflandıran, yüksek doğruluk oranına sahip ve kaynak dostu bir yapay zeka modeli geliştirmektir.

Uygulama Alanları:

Otonom Sistemler: Sürücüsüz araçların yol ve hava durumunu algılaması.

Akıllı Tarım: Bölgesel güneşlenme süresi ve yağış takibi.

Meteoroloji: Yeryüzü tabanlı gökyüzü görüntüleme sistemleri (Sky Imaging).

2. Veri Seti
Projede Kaggle Multi-class Weather Dataset kullanılmıştır. Veri seti 4 temel sınıftan oluşmaktadır:

☁️ Cloudy (Bulutlu)

🌧️ Rain (Yağmurlu)

☀️ Shine (Güneşli)

🌅 Sunrise (Gündoğumu)

Veri Ön İşleme (Preprocessing): Modelin daha verimli öğrenmesi için aşağıdaki işlemler uygulanmıştır:

Yeniden Boyutlandırma: Tüm görüntüler 224x224 piksel boyutuna getirilmiştir.

Normalizasyon: RGB kanalları standart ImageNet ortalamalarına göre normalize edilmiştir.

Veri Ayrımı: Veri seti %80 Eğitim (Train) ve %20 Test (Validation) olarak ayrılmıştır.

3. Kullanılan Yöntem ve Mimari
Bu projede Evrişimli Sinir Ağları (Convolutional Neural Networks - CNN) tercih edilmiştir. Hazır bir model (ResNet vb.) kullanmak yerine, problemin doğasına uygun 3 katmanlı özgün bir CNN tasarlanmıştır.

Neden Custom CNN?
Eğitim Amaçlı: Derin öğrenme katmanlarının (Conv2d, MaxPool, Linear) mantığını kavramak.

Hafif Sıklet (Lightweight): Gereksiz milyonlarca parametre yerine, sadece bu problem için özelleşmiş, CPU üzerinde bile hızlı çalışabilen bir yapı oluşturmak.

Overfitting Kontrolü: Küçük veri setlerinde çok derin ağlar veriyi ezberleyebilir (overfitting). Tasarlanan model Dropout katmanları ile bu riski minimize eder.

Model Mimarisi
Giriş Katmanı: 224x224 RGB Görüntü.

Konvolüsyon Blokları: 3 adet Conv2d + ReLU + MaxPool2d bloğu ile öznitelik çıkarımı.

Düzleştirme (Flatten): Matris verisinin vektöre dönüştürülmesi.

Sınıflandırma (Fully Connected): 512 nöronlu gizli katman ve Dropout sonrası 4 sınıflı çıkış katmanı.

4. Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

Gereksinimler:

Python 3.8 veya üzeri

Gerekli kütüphaneler: torch, torchvision, gradio, pillow

Adımlar:

Projeyi klonlayın:

Bash

git clone https://github.com/KULLANICI_ADIN/Weather-Classification-CNN.git
cd Weather-Classification-CNN
Gerekli kütüphaneleri yükleyin:

Bash

pip install -r requirements.txt
Veri setini hazırlayın: Kaggle veri setini indirin ve dataset klasörü içine sınıf isimleriyle (Cloudy, Rain, Shine, Sunrise) yerleştirin.

5. Kullanım
Modeli Eğitmek
Modeli sıfırdan eğitmek için terminalde şu komutu çalıştırın:

Bash

python train.py
Bu işlem eğitim sürecini başlatır, her epoch sonunda hata oranını (Loss) gösterir ve eğitimi tamamladığında models/weather_model.pth dosyasını kaydeder.

Arayüzü Başlatmak (Web Demo)
Eğitilmiş modeli test etmek ve kullanıcı arayüzünü açmak için:

Bash

python app.py
Gradio arayüzü tarayıcınızda açılacaktır. İster bilgisayarınızdan fotoğraf yükleyebilir, isterseniz de alt kısımdaki örnek butonlarını kullanarak test edebilirsiniz.

6. Dosya Yapısı
Plaintext

Weather-Classification-CNN/
│
├── dataset/                # Eğitim verileri (Kullanıcı tarafından eklenir)
│   ├── Cloudy/
│   ├── Rain/
│   ├── Shine/
│   └── Sunrise/
│
├── examples/               # Arayüz testleri için örnek görseller
├── models/                 # Eğitilen model dosyası (.pth) burada saklanır
│
├── model.py                # Özgün CNN model mimarisi
├── train.py                # Model eğitim kodları
├── app.py                  # Gradio web arayüzü kodları
├── requirements.txt        # Proje bağımlılıkları
└── README.md               # Proje dokümantasyonu