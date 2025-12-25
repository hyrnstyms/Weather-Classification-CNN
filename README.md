# GÖRÜNTÜ İŞLEME VE DERİN ÖĞRENME İLE HAVA DURUMU SINIFLANDIRMA PROJESİ

## 1. Proje Konusu ve Seçilme Gerekçesi

**Projenin Tanımı:**
Bu proje, dijital görüntü işleme ve derin öğrenme teknikleri kullanılarak, dış ortam görüntülerinden anlık hava durumunun (Güneşli, Yağmurlu, Bulutlu, Gündoğumu vb.) otomatik olarak tespit edilmesini ve sınıflandırılmasını amaçlamaktadır.

**Seçilme Gerekçesi ve İlgili Alanın Önemi:**
Hava durumu takibi, geleneksel olarak pahalı sensörler ve meteorolojik istasyonlar aracılığıyla yapılmaktadır. Ancak günümüzde akıllı şehir konseptinin yaygınlaşmasıyla birlikte, "görsel veri" üzerinden anlık ve lokal hava durumu tespiti kritik bir önem kazanmıştır.
* **Otonom Sistemler:** Sürücüsüz araçların yol tutuşunu ayarlaması ve görüş mesafesini analiz etmesi için görsel hava durumu verisi hayati öneme sahiptir.
* **Akıllı Trafik Yönetimi:** Mevcut şehir kameraları (CCTV) kullanılarak, ekstra sensör maliyetine katlanmadan hava koşullarına göre trafik sinyalizasyonunun optimize edilmesi mümkündür.

**Literatür Özeti:**
Literatürde yapılan çalışmalar incelendiğinde, geçmişte renk histogramları ve kenar belirleme gibi geleneksel yöntemlerin kullanıldığı, ancak bu yöntemlerin karmaşık arka planlarda yetersiz kaldığı görülmüştür. Güncel çalışmalarda ise derin öğrenme tabanlı modellerin (özellikle CNN mimarilerinin), öznitelik çıkarımındaki başarısı nedeniyle standart haline geldiği ve %90 üzeri doğruluk oranlarına ulaştığı gözlemlenmiştir.

---

## 2. Veri Setinin Belirlenmesi

**Veri Kaynağı ve İçeriği:**
Modelin eğitimi için Kaggle platformunda bulunan ve akademik çalışmalarda yaygın olarak referans alınan **"Multi-class Weather Dataset"** tercih edilmiştir. Veri seti, farklı atmosfer koşullarını ve ışık seviyelerini içeren etiketli görüntülerden oluşmaktadır.

**Veri Seti Özellikleri:**
* **Sınıflar:** Cloudy (Bulutlu), Rain (Yağmurlu), Shine (Güneşli), Sunrise (Gündoğumu).
* **Veri Dağılımı:** Veri seti dengeli bir yapı gözetilerek hazırlanmış, her sınıf için modelin genelleme yapabilmesine yetecek çeşitlilikte görüntü toplanmıştır.

**Ön İşleme (Preprocessing) Adımları:**
Ham verilerin modele verilmeden önce optimize edilmesi sağlanmıştır:
1.  **Yeniden Boyutlandırma:** Hesaplama maliyetini düşürmek ve standart bir giriş sağlamak amacıyla tüm görüntüler 224x224 piksel boyutuna getirilmiştir.
2.  **Normalizasyon:** Piksel değerleri 0-255 aralığından 0-1 aralığına çekilerek modelin öğrenme hızı (convergence) artırılmıştır.
3.  **Veri Ayırma:** Veri seti; modelin eğitimi için %80 Eğitim (Training), parametre optimizasyonu için %10 Doğrulama (Validation) ve nihai başarım ölçümü için %10 Test seti olarak ayrılmıştır.

---

## 3. Yöntem ve Algoritma Seçimi

**Uygulanan Yöntem: Konvolüsyonel Sinir Ağları (CNN)**

**Seçim Gerekçesi ve Karşılaştırmalı Analiz:**
Görüntü sınıflandırma problemi için literatürdeki yöntemler karşılaştırıldığında en uygun yaklaşımın CNN olduğu belirlenmiştir:

* **Geleneksel Makine Öğrenmesi (SVM/KNN):** Bu yöntemler, görüntüdeki özellikleri (kenar, köşe vb.) elle çıkarmayı gerektirir (Hand-crafted features). Bu durum hem zaman alıcıdır hem de görüntüdeki karmaşık desenleri yakalamakta yetersiz kalır.
* **Yapay Sinir Ağları (ANN):** Görüntüyü tek boyutlu bir vektöre dönüştürdüğü için pikselin komşuluk ilişkilerini (mekansal bilgiyi) kaybeder.
* **CNN (Seçilen Yöntem):** CNN mimarisi, filtreler aracılığıyla görüntüyü tarayarak hiyerarşik özellikleri (önce kenarları, sonra şekilleri, en son nesneleri) otomatik olarak öğrenir. Ayrıca "Translation Invariance" özelliği sayesinde, bulutun veya güneşin görüntünün hangi köşesinde olduğundan bağımsız olarak doğru tespiti yapabilir.

---

## 4. Model Eğitimi ve Değerlendirilmesi

**Model Mimarisi:**
Model, sıralı (Sequential) bir katman yapısı üzerine inşa edilmiştir. Temel bloklar şunlardır:
* **Konvolüsyon Katmanları (Conv2D):** Görüntüden öznitelik haritalarını çıkarır.
* **Havuzlama Katmanları (MaxPooling):** Önemsiz detayları atarak veriyi özetler ve işlem yükünü azaltır.
* **Dropout:** Rastgele nöronları kapatarak modelin ezberlemesini (overfitting) engeller.
* **Tam Bağlı Katman (Dense) ve Softmax:** Çıkarılan özellikleri yorumlayarak 4 sınıf için olasılık değerlerini üretir.

**Eğitim Sonuçları:**
Model eğitimi sonucunda elde edilen Başarım (Accuracy) ve Kayıp (Loss) grafikleri incelendiğinde şu sonuçlar çıkarılmıştır:
1.  **Doğruluk Analizi:** Eğitim doğruluğu ile doğrulama doğruluğunun birbirine paralel ve yükselen bir trend izlediği görülmüştür. Bu durum, modelin sadece eğitim verisini ezberlemediğini, yeni gördüğü verilerde de başarılı olduğunu kanıtlar.
2.  **Kayıp (Loss) Analizi:** İterasyon (epoch) sayısı arttıkça hata oranının istikrarlı bir şekilde sıfıra yaklaştığı gözlemlenmiştir.
3.  **Genel Başarım:** Test seti üzerinde yapılan denemelerde modelin, özellikle "Yağmurlu" ve "Güneşli" gibi görsel olarak zıt sınıfları %90'ın üzerinde bir başarıyla ayırabildiği tespit edilmiştir.

---

## 5. Kurulum ve Kullanım 

Proje dosyaları GitHub üzerinde düzenli bir yapıda tutulmuştur.

### Gereksinimler
* Python 3.8+
* Kütüphaneler: `torch`, `torchvision`, `gradio`, `pillow`

 Projeyi İndirme
```bash
git clone [https://github.com/hyrnstyms/Weather-Classification-CNN.git](https://github.com/hyrnstyms/Weather-Classification-CNN.git)
cd Weather-Classification-CNN
pip install -r requirements.txt

**Modeli Eğitme**
Modeli sıfırdan eğitmek ve `models/weather_model.pth` dosyasını oluşturmak için:

```bash
python train.py

**Arayüzü Başlatma**
Eğitilen modeli kullanıcı dostu web arayüzünde test etmek için:

```bash
python app.py

Gradio arayüzü tarayıcıda açılacaktır. Örnek resimler veya kendi yüklediğiniz fotoğraflarla test yapabilirsiniz.
