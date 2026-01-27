# Beyin Tümörü Sınıflandırma Projesi

Bu proje, MR görüntülerinden **4 sınıf** için beyin tümörü sınıflandırması yapar:

- Glioma
- Meningioma
- No Tumor
- Pituitary

Projede hem **eğitim** (notebook) hem de **inference/test** (Gradio web arayüzü) akışı vardır.

---

## Proje İçeriği / Dosya Yapısı

- `Test.py`  
  Gradio tabanlı web arayüzü ile seçili modeli yükleyip tek görüntü üzerinde tahmin yapar.
- `Train.ipynb`  
  Modellerin eğitimi, doğrulama, test değerlendirme, metrik/ROC-PR grafikleri ve TorchScript export.
- `EDA.ipynb`  
  Veri seti keşfi: sınıf dağılımı, boyut analizi, piksel istatistikleri, edge/texture analizleri.
- `chart.ipynb`  
  Modellerin metriklerini CSV’den okuyup karşılaştırmalı grafikler üretir.
- `Untitled 1.csv`  
  Modellerin performans metrik tablosu (chart notebook bunu kullanır).
- `models/`  
  Eğitilmiş / export edilmiş modeller (TorchScript `.pt`) ve değerlendirme çıktıları.

> Not: Modellerin tamamı repo içinde olmayabilir (boyut kısıtı). Gerekirse Drive linkini kullanın.

---

## Kurulum

### 1) Ortam (önerilen)
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Bağımlılıklar
Projede `requirements.txt` varsa:
```bash
pip install -r requirements.txt
```

Eğer yoksa en azından şunlar gerekir (ortama göre değişebilir):
- `torch`, `torchvision`
- `gradio`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- (EDA için) `opencv-python`, `scikit-image`, `scipy`, `scikit-learn`

---

## Çalıştırma (Web Arayüzü)

```bash
python Test.py
```

Ardından tarayıcıdan:
- http://127.0.0.1:7860

Adımlar:
1. Model seçin
2. MR görüntüsü yükleyin
3. “Tahmin Yap” butonuna basın
4. Sınıf ve olasılıkları görün

---

## Modeller (TorchScript)

`Test.py` içinde `MODELS` sözlüğü, beklenen dosya yollarını tanımlar. Örnek yapı:

- `models/<MODEL_ADI>_<ACC>/`
  - `<MODEL_ADI>_jit.pt`

Eğer model dosyaları eksikse, proje içindeki açıklamadaki Drive linkinden indirip aynı klasör yapısına yerleştirin.

---

## Eğitim (Train.ipynb)

`Train.ipynb` genel akışı:
- Dataset yükleme + train/val split
- Augmentation ve normalization
- Model seçimi (dropdown)
- Eğitim + early stopping + LR scheduler
- Test değerlendirme (Accuracy/Precision/Recall/F1/Kappa/AUC/Specificity)
- Grafiklerin ve raporların kaydı
- TorchScript export (`*_jit.pt`)

> Notebook içinde dataset yolu Colab/Drive örneğiyle tanımlı. Yerelde çalıştıracaksanız `DATASET_PATH` değerini kendi dizininize göre güncelleyin.

---

## EDA (EDA.ipynb)

Bu notebook veri setini anlamak için:
- Sınıf dağılımı (train/test)
- Görüntü boyutları ve kanal analizi
- Piksel yoğunluğu/kontrast istatistikleri
- Edge detection ve texture özellikleri (LBP, GLCM vb.)
- Çeşitli görsellerin dosyaya kaydı

---

## Grafikler (chart.ipynb)

`chart.ipynb`, `Untitled 1.csv` içindeki metriklerden:
- Çoklu metrik bar chart
- Accuracy sıralaması
- Heatmap
- Radar chart (top-5)
- Scatter/violin/parallel coordinates vb.
grafikleri üretir ve PNG olarak kaydeder.

---

## Notlar / Troubleshooting

- **Dataset path hatası**: `Train.ipynb` ve `EDA.ipynb` içindeki `DATASET_PATH / DATA_PATH` değerlerini kontrol edin.
- **CPU/GPU**: CUDA yoksa otomatik CPU’ya düşer; eğitim süresi artar.
- **Normalization farkı**: `Train.ipynb` içinde kullanılan normalize değerleri ile `Test.py` normalize değerlerinin aynı olması önerilir (tutarlılık için).

---

## Lisans / Atıf

Bu repo eğitim/ödev amaçlı hazırlanmıştır. Kullanılan veri seti ve model ağırlıkları için ilgili kaynakların lisans koşullarını kontrol edin.
