# Hand Blur Camera

Aplikasi deteksi gesture tangan menggunakan MediaPipe. Saat gesture peace sign terdeteksi, seluruh frame video akan di-blur.

## Fitur

- Deteksi real-time gesture tangan menggunakan MediaPipe
- Blur otomatis pada frame saat peace sign terdeteksi
- Konfigurasi resolusi dan blur kernel via command line
- Test suite untuk validasi deteksi gesture

## Requirements

- Python 3.8+
- cv2 (OpenCV)
- mediapipe
- numpy

## Instalasi

```bash
pip install opencv-python mediapipe numpy
```

## Penggunaan

### 1. Jalankan Aplikasi Utama

```bash
python main.py
```

**Options:**
- `--resolution WIDTH HEIGHT` - Set resolusi kamera (default: 640 480)
- `--blur-kernel WIDTH HEIGHT` - Set kernel blur (default: 71 71)

**Contoh:**
```bash
python main.py --resolution 1280 720 --blur-kernel 51 51
```

**Kontrol:**
- Tekan `ESC` untuk exit

### 2. Jalankan Test

```bash
python test_gesture.py
```

Test menjalankan 3 skenario:
- Peace sign (2 jari ke atas) → expect blur
- Index only (1 jari) → expect no blur
- Fist (0 jari) → expect no blur

## Struktur Kode

### main.py

**HandBlurCamera class**
- `__init__(resolution, blur_kernel)` - Inisialisasi kamera dan MediaPipe
- `_is_peace_sign(lm)` - Deteksi peace sign dari landmarks
  - Index dan middle finger extended (y < dari joint sebelumnya)
  - Ring dan pinky finger folded (y > dari joint sebelumnya)
- `run()` - Loop utama untuk capture, deteksi, dan blur

### test_gesture.py

**Fungsi helper:**
- `make_landmark(x, y, w, h)` - Buat mock landmark object
- `build_landmarks(fingers, hand_center, w, h)` - Buat mock hand landmarks untuk testing
- `test()` - Jalankan test cases dan report hasil

## Troubleshooting

**Kamera tidak terbuka**
- Check kamera device tersedia
- Coba akses kamera di aplikasi lain dulu

**Blur tidak terdeteksi**
- Pastikan pencahayaan cukup
- Posisi peace sign jelas dan stabil
- Cek detection confidence di Hands config

**Performance lambat**
- Reduce resolusi: `--resolution 480 360`
- Increase blur kernel secara hati-hati
