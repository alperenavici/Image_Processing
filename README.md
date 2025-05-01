# Image Processing Application

Bu uygulama, çeşitli görüntü işleme algoritmaları uygulayabileceğiniz bir web uygulamasıdır. Görüntüleri yükleyebilir ve onlara farklı işlemler uygulayabilirsiniz.

## Özellikler

- Gri Dönüşüm
- Binary Dönüşüm
- Görüntü Döndürme
- Görüntü Kırpma
- Görüntü Yaklaştırma/Uzaklaştırma
- Renk Uzayı Dönüşümleri
- Histogram Germe/Genişletme
- İki Resim Arasında Aritmetik İşlemler (Ekleme, Bölme)
- Kontrast Artırma
- Konvolüsyon İşlemi (Mean)
- Eşikleme İşlemleri (Tek Eşikleme)
- Kenar Bulma Algoritmaları (Prewitt)
- Görüntüye Gürültü Ekleme (Salt & Pepper)
- Gürültü Temizleme Filtreleri (Mean, Median)
- Görüntü Keskinleştirme (Unsharp)
- Morfolojik İşlemler (Genişleme, Aşınma, Açma, Kapama)

## Kurulum

### Gereksinimler

- Node.js (v14 veya üzeri)
- Python (v3.7 veya üzeri)
- pip (Python paket yöneticisi)

### Backend kurulumu

```bash
cd backend
pip install -r requirements.txt
```

### Frontend kurulumu

```bash
cd frontend
npm install
```

## Çalıştırma

### Backend'i çalıştırma

```bash
cd backend
python api/server.py
```

Bu, backend API'sini 5000 numaralı portta çalıştıracaktır.

### Frontend'i çalıştırma

```bash
cd frontend
npm start
```

Bu, frontend uygulamasını genellikle 1234 numaralı portta çalıştıracaktır.

Tarayıcınızda `http://localhost:1234` adresine giderek uygulamayı görebilirsiniz.

## Kullanım

1. Ana sayfada "Upload Image" alanına tıklayarak bir görüntü yükleyin.
2. Soldaki işlemler listesinden uygulamak istediğiniz işlemi seçin.
3. İşlem için gereken parametreleri ayarlayın.
4. İki görüntü gerektiren işlemler için ikinci bir görüntü yükleyin.
5. "Process Image" butonuna basarak işlemi uygulayın.
6. Sonuçları sağ tarafta görebilirsiniz. Orijinal ve işlenmiş görüntüler yan yana gösterilecektir.
7. Histogram verileri varsa, görüntünün altında histogram grafiği görüntülenecektir.

## Teknik Detaylar

Bu proje, hazır görüntü işleme kütüphaneleri kullanmadan, temel NumPy işlemleriyle görüntü işleme algoritmalarını uygular. Frontend React ve TypeScript kullanılarak oluşturulmuştur, backend ise Python/Flask ile geliştirilmiştir. 