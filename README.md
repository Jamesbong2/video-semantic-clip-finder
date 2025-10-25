# video_search (CLIP-based snippet finder)

Ищет по запросу (например, `fight`) релевантные моменты в фильмах и режет короткие клипы (5 сек).
На выходе: папка `clips/<query>/...mp4` и CSV `clips/results_<query>.csv` с колонками:
`outfile,start_sec,end_sec,query`.

## Требования

- **Python 3.10+**
- **FFmpeg** в PATH (проверьте `ffmpeg -version`)
- **PyTorch** (GPU желательно, но есть CPU-режим)
- Остальные библиотеки из `requirements.txt`

> ⚠️ Для PyTorch подберите сборку под вашу CUDA/OS:  
> https://pytorch.org/get-started/locally/  

> Примеры:
> - CUDA 12.x: `pip3 install torch --index-url https://download.pytorch.org/whl/cu121`
> - CPU only: `pip3 install torch --index-url https://download.pytorch.org/whl/cpu`

## Установка


# 1) создать виртуалку (по желанию)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 2) поставить torch (какой подойдёт вам на вашу систему)(https://pytorch.org/)
Пример : pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3) поставить остальные библиотеки
pip install -r requirements.txt
Структура данных

.
├── search_once.py    #скрипт
├── movies/           # сложите сюда видео (*.mp4, *.mkv, *.avi, *.mov)
└── clips/            # сюда упадут нарезанные клипы и CSV (создастся автоматически)


## Запуск

python search_once.py "fight" 
