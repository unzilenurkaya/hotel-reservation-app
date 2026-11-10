# Otel rezervasyonu projesi teknik denetimi

10 Kasım 2026 · Temel sürüm `e29db68` · İnceleme dalı `review/ml-reliability`

## Önce ve sonra

| Alan | Eski sürümde doğrulanan durum | İnceleme dalı |
| --- | --- | --- |
| Örneklem | 36.275 satıra tüm veri üzerinde ardışık IQR uygulandı, 17.158 satır kaldı. | 36.275 satırın tamamı kullanıldı. Aynı model girdileri grup olarak bölündü. |
| Etiket | `Canceled=0`, `Not_Canceled=1`. Streamlit, `1` değerini yüksek iptal uyarısı olarak gösteriyordu. | Açık hedef sözlüğüyle `Canceled=1`, `Not_Canceled=0`. Skor, sınıf sırası ne olursa olsun iptal sınıfına ait sütundan okunuyor. |
| Form | Altı alanın yalnızca `lead_time` değeri saklanan 15 özelliğe geçiyordu. Beş alanın değiştirilmesi model girdisini değiştirmedi. | Form ve eğitim 12 gerçek ham alanı paylaşıyor. Fiyat `avg_price_per_room`, istek sayısı `no_of_special_requests`, otopark `required_car_parking_space`, segment `market_segment_type` üzerinden aktarılıyor. Veri setinde bulunmayan depozito alanı yok. |
| Model | Tarihsel `model.pkl`, XGBoost `XGBClassifier` içeriyor. Döngüden kalan son `model` kaydedilmişti. | Eğitim içi ortalama iptal F1 puanıyla seçim yapıldı. XGBoost kazandı ve ön işleme ile birlikte kaydedildi. Tarihsel dosya korundu. |
| Bağımlılıklar | Orijinal listeyle temiz ortamda Streamlit `No module named 'xgboost'` ile durdu. Listede scikit-learn da eksikti. | Doğrudan bağımlılıklar ve SciPy sürümü sabitlendi. macOS OpenMP gereksinimi belgelendi. |
| Tekrarlanabilirlik | Tarihsel defter, depoda bulunmayan `archive.zip` dosyasını açmaya çalışıyor. Model rastgelelik ayarları sabit değildi. | Ham CSV ile çalışan `train.py` ve bağımsız `audit_legacy.py` var. Bölme ve modeller için tohumlar sabit. |

## 36.275 satır neden 17.158 oldu?

Eski kod, sayısal sütunları sırayla dolaşıyor ve her çeyreklik aralığı önceki filtrelerden geçmiş örneklemde yeniden hesaplıyor. IQR sıfır olduğunda alt ve üst sınır aynı değere eşitleniyor. Örneğin ilk sütun tüm rezervasyonları **tam iki yetişkin** ile sınırlandırdı. Çocuk sayısı, otopark talebi ve varış yılı için de geçerli gözlemler bu şekilde elendi.

| Adım | Elenen satır | Kalan satır |
| --- | ---: | ---: |
| Yetişkin sayısı | 10.167 | 26.108 |
| Çocuk sayısı | 2.389 | 23.719 |
| Hafta sonu ve hafta içi gece sayısı | 171 | 23.548 |
| Otopark talebi | 650 | 22.898 |
| Rezervasyon ile varış arasındaki gün | 623 | 22.275 |
| Varış yılı | 4.356 | 17.919 |
| Tekrar gelen misafir | 106 | 17.813 |
| Gecelik ortalama oda fiyatı | 354 | 17.459 |
| Özel istek sayısı | 301 | 17.158 |

Toplam **19.117 satır (%52,70)** elendi. Ham veride 11.885 iptal ve 24.390 iptal edilmeyen rezervasyon vardı. Filtrelenmiş veride bu sayılar 6.651 ve 10.507 oldu. İptal oranı **%32,76’dan %38,76’ya** çıktı. Sıfır IQR’li sayım, yıl ve ikili alanlarda bu yöntem, veri hatası yerine meşru örnekleri ayıkladı.

Ham CSV’de boş değer bulunmadı ve 36.275 rezervasyon kimliğinin tamamı farklıydı. Kimlik ve hedef hariç 17 ham alanda ilk örnekten sonra tekrar eden 10.408 girdi satırı var. Ayrıca 37 satırda 29 Şubat 2018 tarihi bulunuyor. Yeni model ay bilgisini kullanıyor ve yılı, günü ve zamanlama durumu doğrulanamayan geçmiş müşteri alanlarını girdi dışında bırakıyor.

## Sızıntı ve değerlendirme

- **Bölme öncesi veri kullanımı.** Eski IQR eşikleri, kategori frekansları, kategorik kodlama ve sabit sütun seçimi, test ayrılmadan önce tüm ilgili örneklem üzerinden hesaplandı.
- **Eş girdi örtüşmesi.** Eski 3.432 test kaydının 1.210’u (%35,26), 15 model özelliğinde eğitimdeki en az bir kayda eşit. 714 farklı girdi deseni iki tarafta da var. Bu örtüşme ayrı rezervasyonları aynı kişiye bağladığımız anlamına gelmez.
- **Hedef.** Eski modelin özellikleri arasında `booking_status` yok. Hedef, nadir kategori işlevinin kapsamına girmiş olsa da iki sınıf da %1’in üzerinde olduğundan bu CSV’de etiketler birleşmedi. Eski pozitif sınıf `1` iptal edilmeyen rezervasyondu.
- **Yeni değerlendirme.** 25.478 farklı girdi grubu, eğitim ve test arasında paylaşılmadı. Eğitim 29.164, test 7.111 satırdan oluştu. Ayrı test oranı %19,60 ve testte 2.246 iptal var. Ön işleme her eğitim bölümünde ayrı öğrenildi. Model seçimi, test kullanılmadan dört katlı eğitim içi doğrulamayla yapıldı.
- **Sınırlar.** Gruplar müşteri kimliğine göre değil girdi değerlerine göre oluşturuldu. Rezervasyon ve iptal zaman damgaları olmadığı için ileri tarihli bağımsız bir dönemde performans veya gerçek zamanlı kullanım doğrulanmadı. Model skoru kalibre edilmedi.

| Model | Eski doğruluk, 3.432 satır | Yeni doğruluk, 7.111 satır | Yeni iptal F1 |
| --- | ---: | ---: | ---: |
| Random Forest | %89,13 | %84,32 | 0,7273 |
| XGBoost | %87,79 | %85,68 | 0,7552 |

Yeni XGBoost’un eğitim içi iptal F1 ortalaması 0,7679, Random Forest’ın 0,7395. Seçim bu ortalamalara dayanıyor. Seçilen model ayrı testte 1.570 iptali yakaladı, 676 iptali kaçırdı ve iptal edilmeyen 342 kayda iptal dedi. Değişen örneklem, girdi seti ve bölme nedeniyle eski ve yeni doğruluk farkı tek başına model performansındaki artış veya düşüşü ölçmez.

## Arayüz ve test kanıtı

- Eski `model.pkl` dosyasının sınıfı ve 15 beklenen özelliği doğrulandı. XGBoost ve OpenMP yüklendikten sonra gerçek eski modelle iki çıktı da çalıştırıldı. Eski sınıf 0 iptal olduğu halde «iptal ihtimali düşük», sınıf 1 iptal edilmediği halde «iptal ihtimali yüksek» mesajı gösterdi.
- Eski formda fiyat, özel istek, otopark, depozito ve segment değerleri birlikte değiştirildi. Modele gönderilen 15 sütunlu satır birebir aynı kaldı. Boş sütunlara sıfır atamak, bu beş değerin modele ulaşmasını sağlamadı.
- Ayrı Python 3.12.14 ortamına `requirements.txt` kuruldu ve 47 paketin bağımlılık uyumu kontrol edildi. macOS arm64 üzerinde Homebrew OpenMP 23.1.5 çalışma zamanı yerel test dizininden kullanıldı.
- Temiz ortamda eğitim iki kez çalıştırıldı ve iki değerlendirme JSON’u birebir aynı çıktı. Son çalıştırmada sayısal uyarılar hata kabul edildi.
- **6 otomatik test geçti.** Gerçek modelle hem iptal hem iptal edilmeme mesajı, skor sütunu, tüm form alanlarının aktarılması, grup ayrımı, seçilen model ve eksik/bozuk dosya davranışı kontrol edildi.
- Yerel Chrome oturumunda formdan tahmin alındı. Ekran görüntüsündeki örnek «İptal edilmez» ve %4,6 iptal model skoru üretti.

## GitHub ve özgeçmiş

**Depo açıklaması önerisi.** Hotel booking cancellation prediction with grouped validation and a reproducible Streamlit demo.

**Konu etiketleri önerisi.** `python`, `machine-learning`, `streamlit`, `scikit-learn`, `xgboost`, `classification`, `reproducibility`.

README, ekran görüntüsü, eğitim komutu ve test sonuçları inceleme dalında hazır. Yerel demo doğrulandı. Barındırılan demo için modelin kurulum aşamasında üretilmesi veya doğrulanmış eğitim çıktısının paketlenmesi gerekir. Bir sunucuda yayınlamadan önce veri setinin lisansı ve o ortamın çalıştırma testi kontrol edilmeli.

Özgeçmişteki %89,13 ve %87,79 değerleri eski deney için doğru. Açıklama bunları güncel arayüzün doğrulanmış sonuçları gibi gösterebildiği için, inceleme dalı GitHub’da erişilebilir olduğunda güncel değerlendirme metniyle değiştirilmesi uygun.

> 36.275 otel rezervasyonunda dört sınıflandırma modelini karşılaştırdım. Aynı özelliklere sahip kayıtları gruplandırarak eğitim ve testi ayırdım. Seçilen XGBoost modeli, 7.111 kayıtlık ayrı testte %85,68 doğruluk ve iptal sınıfında 0,755 F1 puanı elde etti. Ortak ön işleme hattını Streamlit arayüzüne bağladım.

## Kaynaklar

- [Orijinal GitHub deposu ve tarihsel not defteri](https://github.com/unzilenurkaya/hotel-reservation-app)
- [Hotel Reservations Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)
- [scikit-learn veri sızıntısı ve ön işleme kılavuzu](https://scikit-learn.org/stable/common_pitfalls.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [XGBoost kurulum kılavuzu](https://xgboost.readthedocs.io/en/stable/install.html)
