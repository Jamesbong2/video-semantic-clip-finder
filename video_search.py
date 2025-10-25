
import os, sys, subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


# --------- НАСТРОЙКИ ---------
MOVIES_DIR    = "movies"
OUT_DIR       = "clips"
SAMPLE_FPS    = 2          # сколько кадров/сек анализировать
CLIP_SEC      = 5.0      # длительность клипа
SMOOTH_SEC    = 2.0        # сглаживание sim по времени (сек)
MIN_GAP_SEC   = 12.0       # минимальный разрыв между клипами
THRESHOLD     = 0.25       # не используется в пороге, оставлено для совместимости
TOP_PER_MOV   = None       # ограничение числа клипов на фильм (или None)
BATCH         = 256        # batch для CLIP

FFMPEG_LOGLVL = "error"


ANTI_BLACK_MIN_NB = 0.08   # доля не-чёрных пикселей в окне (0..1)
ANTI_BLACK_MIN_ED = 0.008  # плотность «краёв» (текстуры) в окне


FAST_ACCEPT_MARGIN = 0.010  

SOFT_KEEP_MIN = 4
SOFT_KEEP_MAX = 8


THR_FLOOR_MIN  = 0.05
THR_FLOOR_KSIG = 2.0
THR_ALPHA      = 1.2


#классификатор CLIP 
class ClipEncoder:
    def __init__(self, model_name="ViT-L-14", pretrained="openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        torch.backends.cudnn.benchmark = True
        if self.device == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tok = self.tokenizer([text]).to(self.device)
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            t = self.model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
        return t.detach().cpu().numpy()[0].astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        if self.device == "cuda":
            batch = batch.half()
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            x = self.model.encode_image(batch)
        x = x / x.norm(dim=-1, keepdim=True)
        return x.detach().cpu().numpy().astype(np.float32)


# --------- УТИЛИТЫ ---------
def bgr_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def get_video_meta(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None
    cap.release()
    return {
        "native_fps": float(fps),
        "native_frame_count": int(frame_count),
        "movie_duration_sec": float(duration) if duration is not None else None
    }

def iter_sampled_frames(video_path: str, sample_fps: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(native / sample_fps)), 1)

    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            ts = frame_idx / native
            yield frame, ts, frame_idx
        frame_idx += 1
    cap.release()

def ffmpeg_cut(src: str, dst: str, start: float, duration: float):
   
    from shutil import which
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = which("ffmpeg") or "ffmpeg"

    back = 1.0
    rough = max(0.0, start - back)
    precise = start - rough

    base_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", FFMPEG_LOGLVL,
        "-ss", f"{rough:.3f}", "-i", src,
        "-ss", f"{precise:.3f}", "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", "format=yuv420p",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-shortest", "-sn",
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-y", dst
    ]
    try:
        subprocess.run(base_cmd, check=True)
    except subprocess.CalledProcessError:
        # фоллбек на libx264
        cmd_fb = base_cmd[:]
        for i in range(len(cmd_fb) - 1):
            if cmd_fb[i] == "-c:v":
                cmd_fb[i+1] = "libx264"
        subprocess.run(cmd_fb, check=True)

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, ker, mode="same")

def cheap_metrics(frame_bgr, down_w=160) -> Tuple[float, float]:
    """Быстрые метрики на кадр: (non_black_ratio, edge_density)."""
    h, w = frame_bgr.shape[:2]
    new_h = int(h * (down_w / max(w, 1)))
    small = cv2.resize(frame_bgr, (down_w, max(1, new_h)), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    nb_ratio = float((gray > 18).mean())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    return nb_ratio, edge_density

def window_indices(center_i: int, half_w: int, n: int) -> Tuple[int, int]:
    l = max(0, center_i - half_w)
    r = min(n, center_i + half_w + 1)
    return l, r

def topk_mean_window(arr: np.ndarray, l: int, r: int, k: int = 5) -> float:
    size = r - l
    if size <= 0:
        return 0.0
    k = min(k, size)
    part = np.partition(arr[l:r], size - k)
    return float(np.mean(part[size - k:]))


# ПОСЛЕДОВАТЕЛЬНАЯ НАРЕЗКА
def cut_clips(tasks: List[Tuple[str, str, float, float]]) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    print(f"[CUT] Режу клипы: {len(tasks)} шт...")
    results: List[Dict[str, Any]] = []
    for (src, dst, start, duration) in tqdm(tasks, desc="Cut"):
        try:
            ffmpeg_cut(src, dst, start, duration)
            results.append({"outfile": dst, "ok": True, "error": None,
                            "src": src, "dst": dst, "start": start, "duration": duration})
        except subprocess.CalledProcessError as e:
            results.append({"outfile": dst, "ok": False, "error": f"ffmpeg error: {e}",
                            "src": src, "dst": dst, "start": start, "duration": duration})
        except Exception as e:
            results.append({"outfile": dst, "ok": False, "error": str(e),
                            "src": src, "dst": dst, "start": start, "duration": duration})
    ok_cnt = sum(1 for r in results if r["ok"])
    print(f"[CUT] Готово: ok={ok_cnt}/{len(results)}, fail={len(results) - ok_cnt}")
    return results


#ОСНОВНАЯ ЛОГИКА 
def search_once(query: str,
                movies_dir=MOVIES_DIR, out_dir=OUT_DIR,
                sample_fps=SAMPLE_FPS, clip_sec=CLIP_SEC,
                smooth_sec=SMOOTH_SEC, min_gap_sec=MIN_GAP_SEC,
                top_per_movie=TOP_PER_MOV, batch=BATCH):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    class_dir = Path(out_dir) / query
    class_dir.mkdir(parents=True, exist_ok=True)

    enc = ClipEncoder()
    print(f"[INFO] device = {enc.device}")
    qvec = enc.encode_text(query)

   
    csv_rows: List[Dict[str, Any]] = []


    movie_paths: List[Path] = []
    for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov"):
        movie_paths += list(Path(movies_dir).rglob(ext))
    movie_paths = sorted(movie_paths)

    if not movie_paths:
        print(f"[WARN] Не нашёл фильмов в {movies_dir}")
        return pd.DataFrame()

    try:
        cv2.setNumThreads(4)
    except Exception:
        pass

    for mpath in movie_paths:
        mpath = str(mpath)
        mname = os.path.basename(mpath)
        print(f"\n[SCAN] {mname}")

        meta = get_video_meta(mpath)
        native_fps = meta["native_fps"]
      
        frames_pil: List[Image.Image] = []
        tss: List[float] = []
        idxs: List[int] = []
        frame_sims: List[float] = []
        nb_list: List[float] = []
        ed_list: List[float] = []

        step = max(int(round(native_fps / sample_fps)), 1)

     
        for frame_bgr, ts, idx in tqdm(iter_sampled_frames(mpath, sample_fps), desc="Read frames"):
            frames_pil.append(bgr_to_pil(frame_bgr))
            tss.append(ts)
            idxs.append(idx)
            nb, ed = cheap_metrics(frame_bgr)
            nb_list.append(nb)
            ed_list.append(ed)

            if len(frames_pil) >= batch:
                feats = enc.encode_images(frames_pil)
                sims = feats @ qvec
                frame_sims.extend(sims.tolist())
                frames_pil.clear()

        if frames_pil:
            feats = enc.encode_images(frames_pil)
            sims = feats @ qvec
            frame_sims.extend(sims.tolist())
            frames_pil.clear()

        if not frame_sims:
            print("[WARN] Нет кадров")
            continue


        tss_np  = np.array(tss, dtype=np.float32)
        sims_np = np.array(frame_sims, dtype=np.float32)
        nb_np   = np.array(nb_list, dtype=np.float32)
        ed_np   = np.array(ed_list, dtype=np.float32)

        k = max(int(round(smooth_sec * sample_fps)), 1)
        sims_sm = smooth_1d(sims_np, k)

    
        nb_p25 = float(np.percentile(nb_np, 25))
        ed_p25 = float(np.percentile(ed_np, 25))
        nb_thr_adapt = max(ANTI_BLACK_MIN_NB, nb_p25)
        ed_thr_adapt = max(ANTI_BLACK_MIN_ED, ed_p25)

        thr_mean  = float(np.mean(sims_sm))
        thr_std   = float(np.std(sims_sm))
        thr_floor = max(THR_FLOOR_MIN, thr_mean + THR_FLOOR_KSIG * thr_std)
        auto_thr  = max(thr_floor,   thr_mean + THR_ALPHA * thr_std)

        print(f"[INFO] auto threshold = {auto_thr:.3f} | mean={thr_mean:.3f}, std={thr_std:.3f}")


        order = np.argsort(-sims_sm)

    
        taken = np.zeros_like(sims_sm, dtype=bool)
        min_gap_frames = max(int(round(min_gap_sec * sample_fps)), 1)
        peaks_candidates: List[int] = []
        for idx in order:
            if sims_sm[idx] < auto_thr:
                continue
            if taken[idx]:
                continue
            l = max(0, idx - min_gap_frames)
            r = min(len(sims_sm), idx + min_gap_frames + 1)
            taken[l:r] = True
            peaks_candidates.append(idx)
            if top_per_movie is not None and len(peaks_candidates) >= int(top_per_movie):
                break


        half_w = int(round((clip_sec / 2.0) * sample_fps))
        filtered_peaks: List[int] = []
        for pi in peaks_candidates:
            l, r = window_indices(pi, half_w, len(sims_sm))
            nb_ratio_win = float(np.mean(nb_np[l:r]))
            ed_ratio_win = float(np.mean(ed_np[l:r]))
            avg_sim_topk = topk_mean_window(sims_sm, l, r, k=5)
            pass_fast = (avg_sim_topk >= auto_thr + FAST_ACCEPT_MARGIN) and (nb_ratio_win >= 0.30)
            pass_adpt = (nb_ratio_win >= nb_thr_adapt) and (ed_ratio_win >= ed_thr_adapt) and (avg_sim_topk >= auto_thr - 0.002)
            if pass_fast or pass_adpt:
                filtered_peaks.append(pi)


        if len(filtered_peaks) < SOFT_KEEP_MIN:
            need = min(SOFT_KEEP_MAX, SOFT_KEEP_MIN) - len(filtered_peaks)
            if need > 0:
                used_set = set(filtered_peaks)
                remaining: List[Tuple[int, float]] = []
                for pi in peaks_candidates:
                    if pi in used_set:
                        continue
                    l, r = window_indices(pi, half_w, len(sims_sm))
                    nb_ratio_win = float(np.mean(nb_np[l:r]))
                    if nb_ratio_win < 0.35:
                        continue
                    avg_sim_topk = topk_mean_window(sims_sm, l, r, k=5)
                    remaining.append((pi, avg_sim_topk))
                    if len(remaining) >= need * 2:
                        break
                remaining.sort(key=lambda x: -x[1])
                for pi, _ in remaining[:need]:
                    filtered_peaks.append(pi)
                filtered_peaks = list(dict.fromkeys(filtered_peaks))

      
        filtered_peaks.sort(key=lambda i: -sims_sm[i])  # стабильный порядок
        cut_tasks: List[Tuple[str, str, float, float]] = []
        clip_bounds: List[Tuple[float, float]] = []  # (start, end) для CSV

        for i, pi in enumerate(filtered_peaks, 1):
            t_center = float(tss_np[pi])
            start = max(0.0, t_center - clip_sec / 2)
            end = start + clip_sec
            out_name = f"{Path(mpath).stem}_{int(start)}_{int(end)}_{query}_{i:03d}.mp4"
            out_path = str((Path(out_dir) / query / out_name))
            cut_tasks.append((mpath, out_path, start, clip_sec))
            clip_bounds.append((start, end))

        cut_results = cut_clips(cut_tasks)
        for (start, end), res in zip(clip_bounds, cut_results):
            if not res["ok"]:
                continue
            csv_rows.append({
                "outfile": res["outfile"],
                "start_sec": float(start),
                "end_sec": float(end),
                "query": query
            })

   
    df = pd.DataFrame(csv_rows, columns=["outfile", "start_sec", "end_sec", "query"])
    csv_path = Path(out_dir) / f"results_{query}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[OK] Сохранён CSV: {csv_path}  | Клип(ов): {len(df)}")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование:  python search_once.py "<query>"\nПример:    python search_once.py "heavy rain"')
        sys.exit(0)
    query = sys.argv[1]
    search_once(query)
