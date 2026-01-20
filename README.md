# ASVspoof2019 LA: STFT + LightCNN (spoof detection)

Модель:
- Фронтенд: log-power STFT (n_fft=1724, hop=128, win=1724)
- Вход: 1 × 867 × 600
- Бэкэнд: LightCNN (MFM), классификация bonafide vs spoof
- Метрика: EER на dev
- Логирование: W&B (опционально)

## Установка
```bash
pip install -r requirements.txt
