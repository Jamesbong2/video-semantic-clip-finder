# Video Semantic Clip Finder

Ищет в фильмах моменты по текстовому запросу (CLIP), режет клипы и сохраняет в `clips/<query>`, плюс пишет CSV с колонками:
`outfile,start_sec,end_sec,query`.

## Требования
- Python 3.9–3.12
- FFmpeg в PATH
- PyTorch (ставится вручную под вашу ОС/CUDA)

## Установка

1. Установите PyTorch под вашу систему (CPU или CUDA):
   👉 https://pytorch.org/get-started/locally/

2. Установите остальные зависимости:
   ```bash
   pip install -r requirements.txt
