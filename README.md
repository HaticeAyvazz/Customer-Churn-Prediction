# Customer Churn Prediction

Telco müşteri verisi üzerinde uçtan uca bir makine öğrenmesi projesi: veri temizleme ve EDA,
denetimli churn sınıflandırması, hiperparametre optimizasyonu, SHAP ile model yorumlanabilirliği,
müşteri segmentasyonu (clustering) ve tahminleri canlı test edebileceğin bir Streamlit uygulaması.

## Proje Akışı

1. **Veri Temizleme & EDA** — eksik/boşluk değerler, tip dönüşümleri, korelasyon ve dağılım analizleri
2. **Encoding** — ikili kategorik kolonlar için `LabelEncoder`, çok sınıflı nominal kolonlar için `OneHotEncoder`
3. **Scaling** — `StandardScaler`
4. **Model Karşılaştırması** — 7 farklı sınıflandırıcı, varsayılan parametrelerle
5. **Hiperparametre Tuning** — `GridSearchCV` ile en güçlü 3 aday üzerinde ince ayar
6. **En İyi Model Seçimi**
7. **SHAP Analizi** — global ve tekil müşteri bazında model açıklanabilirliği
8. **Müşteri Segmentasyonu** — PCA + KMeans/DBSCAN/HDBSCAN ile kümeleme ve küme profilleme
9. **Streamlit Uygulaması** — churn tahmini + SHAP açıklaması

## Veri Ön İşleme Detayları

- `TotalCharges` kolonu string olarak gelen boşluk (`" "`) değerler içeriyordu → `pd.to_numeric(errors="coerce")` ile sayısala çevrildi, oluşan `NaN`'lar `0` ile dolduruldu (bu satırlar `tenure = 0` olan, yani henüz hiç fatura almamış yeni müşterilere karşılık geliyordu).
- `customerID` modelden çıkarıldı (tahmine katkısı olmayan bir kimlik kolonu).
- İkili kolonlar (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) → `LabelEncoder`.
- Çok sınıflı nominal kolonlar (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`) → `OneHotEncoder(drop="first")` (dummy variable tuzağını önlemek için ilk sınıf düşürüldü).
- Tüm sayısal öznitelikler `StandardScaler` ile ölçeklendi (özellikle Logistic Regression ve SVC gibi mesafe/gradyan tabanlı modeller için gerekli).
- `train_test_split` işlemi `stratify=y` ile yapıldı — churn oranı (%26.5 Yes / %73.5 No) train ve test setinde korunuyor, aksi halde azınlık sınıfı test setinde yetersiz temsil edilebilirdi.

## Model Karşılaştırması (varsayılan parametrelerle)

| Model | Test F1 | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.609 | **0.807** |
| Gradient Boosting | 0.570 | 0.798 |
| LightGBM | 0.569 | 0.791 |
| XGBoost | 0.553 | 0.776 |
| Random Forest | 0.550 | 0.787 (train'de 0.996 → aşırı overfit) |
| KNeighbors | 0.512 | 0.747 |

Bu ilk turda en güçlü 3 aday (**Logistic Regression, Gradient Boosting, SVC**) `GridSearchCV`
ile ayrı ayrı tune edildi (`cv=5`, `scoring='f1'`):

| Model | Tuned Test F1 | Tuned Test Accuracy | Taranan parametreler |
|---|---|---|---|
| **Logistic Regression** | **0.619** | 0.784 | `C`, `penalty` (`l1`/`l2`), `solver='liblinear'` |
| Gradient Boosting | 0.581 | 0.803 | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| SVC | 0.557 | 0.779 | `C`, `gamma`, `kernel` (`rbf`/`linear`) |

## Neden Logistic Regression Seçildi?

- **Metrik seçimi**: Churn dağılımı dengesiz (%73.5 kalan / %26.5 churn eden), bu yüzden
  `accuracy` yanıltıcı — her müşteriyi "kalacak" tahmin eden bir model bile ~%73 accuracy alır.
  Asıl önemli olan **F1 skoru**: hem churn eden müşteriyi kaçırmamak (recall) hem de gereksiz
  yere "churn edecek" damgası vurmamak (precision) arasındaki denge.
- Tuning F1'e göre optimize edildiğinde (`scoring='f1'`) **Logistic Regression tüm modeller
  arasında en yüksek Test F1'i (0.619) verdi** — hem base hem tuned karşılaştırmalarında.
- Tuning sonrası accuracy hafifçe düştü (0.807 → 0.784) ama bu beklenen bir trade-off: model
  "No" sınıfına daha az yanlı hale gelip churn eden müşterileri daha iyi yakalamaya başladı —
  churn tahmininde iş hedefi bu.
- Ağaç tabanlı modeller (Random Forest, XGBoost) train setinde çok yüksek skorlar verse de test
  setinde düşüyor → **overfitting**. Logistic Regression'ın basit/doğrusal yapısı bu veri
  büyüklüğü ve özellik sayısı için daha iyi genelleme sağladı.
- Ekstra fayda: Logistic Regression **doğrusal ve yorumlanabilir** bir model — SHAP ile
  katsayı yönleri kolayca iş mantığına (tenure, contract type, internet service vb.)
  bağlanabiliyor.

## SHAP Analizi

Seçilen model üzerinde iki seviyede yorumlanabilirlik sağlandı:

- **Global**: bar plot + beeswarm plot ile hangi özelliklerin genel olarak churn tahminini en
  çok etkilediği (`tenure`, `Contract`, `InternetService`, `OnlineSecurity`, `TechSupport` öne
  çıkan özellikler arasında).
- **Tekil (local)**: waterfall plot ile herhangi bir müşteri için "bu tahmin neden verildi"
  sorusunun cevabı — hangi özellik olasılığı kaç puan artırdı/azalttı.

## Müşteri Segmentasyonu

Churn modelinden bağımsız olarak, PCA ile boyut indirgeme sonrası KMeans / DBSCAN / HDBSCAN
karşılaştırılarak müşteriler davranışsal kümelere ayrıldı ve her kümenin churn oranı, ortalama
harcama gibi metriklerle profili çıkarıldı (`cluster_profile`).

## Streamlit Uygulaması

`app.py`, notebook'ta kaydedilen artefaktları (`models/*.pkl`) kullanarak canlı tahmin yapar:
kullanıcı formdan müşteri bilgilerini girer, model churn olasılığını verir ve SHAP waterfall
grafiğiyle bu tahminin gerekçesini gösterir.

## Kurulum & Çalıştırma

```bash
pip install -r requirements.txt
```

1. `Telco-Customer-Churn.csv` dosyasını proje kök dizinine ekle (repoya dahil değil).
2. Notebook'u baştan sona çalıştır — bu işlem `models/` klasörünü ve tüm `.pkl` artefaktlarını
   üretir:
   ```bash
   jupyter notebook prediction.ipynb
   ```
3. Streamlit uygulamasını başlat:
   ```bash
   streamlit run app.py
   ```

## Proje Yapısı

```
.
├── prediction.ipynb   # Ana notebook: EDA, modelleme, tuning, SHAP, clustering
├── app.py              # Streamlit tahmin uygulaması
├── requirements.txt
├── .gitignore
├── models/             # best_model, preprocessor, scaler, encoders, shap_explainer (.pkl)
├── reports/            # notebook'tan üretilen grafik görselleri
└── README.md 
```

## Notlar

-Ham veri seti (.csv) boyut ve lisans nedeniyle repoya dahil edilmemiştir — Kaggle üzerinden indirilebilir.
