# TRiP(cast) skorunun dış validasyonu – analiz kodu
# External validation of the TRiP(cast) score – analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23455209.svg)](https://doi.org/10.5281/zenodo.23455209)

Bu depo, aşağıdaki çalışmanın tüm istatistiksel analizlerini, tablolarını ve şekillerini yeniden üreten kodu içerir:

> Baykal T, Bulut M, Çetin E. *Alt ekstremite immobilizasyonunda tromboprofilaksi gerektirmeyen düşük venöz tromboembolizm riskli hastaların belirlenmesinde TRiP(cast) skorunun değeri: dış validasyon çalışması.* [Dergi, yıl, makale DOI eklenecek]

This repository contains the code that reproduces all statistical analyses, tables and figures of the study above.

---

## İçerik / Contents

| Dosya / File | Açıklama / Description |
|---|---|
| `trip_cast_analysis.py` | Tüm analizleri tek seferde çalıştıran betik / single script running all analyses |
| `requirements.txt` | Kullanılan Python kütüphaneleri ve sürümleri / library versions |
| `ornek_veri/sahte_ornek_veri.xlsx` | **Sahte** örnek veri (gerçek hasta verisi değildir) / **synthetic** example data, not real patients |
| `LICENSE` | MIT lisansı / MIT licence |

**Gerçek hasta verisi bu depoda yer almaz.** Veriler, etik kurul onayı ve kişisel verilerin korunması mevzuatı (KVKK) gereği yalnızca makul talep üzerine, sorumlu yazar aracılığıyla paylaşılabilir.

**Real patient data are not included.** Data are available from the corresponding author on reasonable request, subject to ethics approval and Turkish data protection law (KVKK).

---

## Kurulum / Installation

Python 3.11 veya üzeri gereklidir (analizler Python 3.13 ile yapılmıştır).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım / Usage

```bash
# Sahte örnek veriyle deneme / test run with synthetic data
python trip_cast_analysis.py --data ornek_veri/sahte_ornek_veri.xlsx --out sonuclar_ornek

# Gerçek veriyle / with the study data
python trip_cast_analysis.py --data veri/trip_cast_veri.xlsx --out sonuclar
```

Betik ekrana özet sonuçları yazar ve `--out` klasörüne şunları kaydeder:

| Çıktı / Output | Makaledeki karşılığı / In manuscript |
|---|---|
| `tablo2_temel_ozellikler.csv` | Tablo 2 (ek olarak <35 yaş satırı) |
| `tablo3_kesim_degerleri.csv` | Tablo 3 (yaş kuralları dahil) |
| `tablo4_bilesenler.csv` | Tablo 4 |
| `tablo5_duyarlilik_altgrup.csv` | Tablo 5 |
| `tablo6_takip_kaybi_senaryo.csv` | Tablo 6 |
| `kalibrasyon_gruplari.csv`, `karar_egrisi.csv` | Şekil 4 ve Şekil 5 verileri |
| `ozet_sonuclar.json` | Metinde geçen tüm özet istatistikler (AUC, DeLong, kalibrasyon, McNemar, net fayda, devrilme noktası) |
| `Sekil1_AkisSemasi.png` … `Sekil5_KararEgrisi.png` | Şekil 1–5 (300 dpi) |

Bootstrap tohum değeri sabittir (`SEED = 2026`); aynı veriyle her çalıştırmada aynı sonuç elde edilir.

---

## Veri biçimi / Data format

Excel dosyası, aşağıdaki sırayla **25 sütun** içermelidir (sütun başlıkları önemli değildir, sıra önemlidir):

| # | Değişken | Kodlama |
|---|---|---|
| 1 | Hasta sıra no | tam sayı |
| 2 | Cinsiyet | 1 = erkek, 2 = kadın |
| 3 | Yaş | yıl |
| 4 | Travma riski | 1 = düşük, 2 = orta, 3 = yüksek |
| 5 | İmmobilizasyon tipi | 1 = ayak/parmak, 2 = diz altı, 3 = uzun bacak |
| 6 | Boy | metre |
| 7 | BKİ | kg/m² |
| 8 | Boy | cm |
| 9 | Kilo | kg |
| 10 | BKİ kategorisi | 0 = <25, 1 = 25–35, 2 = ≥35 |
| 11 | Ailede VTE öyküsü | 0/1 |
| 12 | Kişisel VTE öyküsü | 0/1 |
| 13 | Kanser (son 5 yıl) | 0/1 |
| 14 | Son 3 ayda hastane yatışı | 0/1 |
| 15 | Variköz ven / kronik venöz yetmezlik | 0/1 |
| 16 | Son 3 ayda >6 saat yolculuk | 0/1 |
| 17 | Kalp yetmezliği | 0/1 |
| 18 | Romatoid artrit | 0/1 |
| 19 | Kronik böbrek hastalığı | 0/1 |
| 20 | KOAH | 0/1 |
| 21 | Oral kontraseptif / hormon tedavisi | 0/1 |
| 22 | 90 günde semptomatik VTE | 0/1 (sonlanım) |
| 23 | Profilaktik antikoagülan reçetesi | 0/1 |
| 24 | TRiP(cast) puanı | tam sayı |
| 25 | Yüksek risk grubu | 1 = ≥7, 0 = <7 |

Betik, 24. sütundaki puanı bileşenlerden yeniden hesaplayarak kontrol eder ve uyumsuz satır sayısını ekrana yazar.

## Yöntem notları / Methods notes

- **Ayırt edicilik:** AUC; ana analizde 2000 tekrarlı yüzdelik bootstrap GA, alt gruplarda DeLong GA. Skor ile sürekli yaş ve yaş puanı çıkarılmış skor, bağımlı ROC eğrileri için DeLong testiyle karşılaştırılır.
- **Tanısal performans:** Duyarlılık, özgüllük, PPD, NPD (Wilson GA); LR+/LR− (logaritmik GA); ≥7 eşiği ile ≥55 yaş kuralı arasında eşleştirilmiş McNemar kesin testi.
- **Kalibrasyon:** Orijinal POT-CAST denklemi, logit(p) = −6,677015 + 0,3332203 × puan (Nemeth ve ark., EClinicalMedicine 2020; Ek Tablo A2). Gözlenen/beklenen oranı (Poisson kesin GA), kalibrasyon kesişimi (ofsetli lojistik model), kalibrasyon eğimi, Spiegelhalter z testi, Brier skoru.
- **Klinik yararlılık:** %0,5–15 eşik olasılıklarında net fayda (karar eğrisi analizi).
- **Takip kaybı:** Kayıp hastaların puanları veri setinde bulunmadığından, risk sınıfı ve VTE sıklığına ilişkin senaryolar ve düşük risk grubunda başarısızlık oranını %1'in üzerine çıkaran en az olay sayısı (devrilme noktası). Kayıp hasta sayısı ve akış şeması sayıları betiğin başındaki `FLOW` sözlüğünde tanımlıdır.

## Atıf / Citation

Bu kodu kullanırsanız lütfen makaleye ve bu deponun Zenodo DOI'sine atıf yapın: [https://doi.org/10.5281/zenodo.23455209](https://doi.org/10.5281/zenodo.23455209)

## İletişim / Contact

Taygun Baykal – sorumlu yazar / corresponding author – [e-posta]
