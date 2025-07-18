import librosa
import numpy as np
import os
import json

# 假設你的音樂文件路徑
MUSIC_DIR = "C:\\Users\\user\\Downloads\\Merry_mid\\Merry_mid\\level_Universal"
CONFIG_PATH = os.path.join(MUSIC_DIR, "config.json")

def analyze_music_beats(music_file_path):
    y, sr = librosa.load(music_file_path)
    # 偵測節拍，默認使用 Beat Tracking Algorithm (DBN)
    # 可以調整 hop_length, onset_detector 等參數來優化效果
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times.tolist() # 轉換為列表方便存儲

def preprocess_beats():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: {CONFIG_PATH} not found.")
        return

    for level_name, level_data in config.items():
        if "music" in level_data:
            music_relative_path = level_data["music"]
            music_full_path = os.path.join(MUSIC_DIR, music_relative_path)

            if os.path.exists(music_full_path):
                print(f"Analyzing beats for {music_full_path}...")
                beat_times = analyze_music_beats(music_full_path)
                level_data["beat_times"] = beat_times # 將節拍時間存入配置

                # 為了方便，也可以將 highlight_interval 改為動態計算
                # 例如，根據前幾個節拍的平均間隔來設定
                if len(beat_times) > 1:
                    avg_interval = np.mean(np.diff(beat_times[:min(5, len(beat_times))]))
                    level_data["highlight_interval_from_beats"] = float(avg_interval)
                    print(f"  Detected average beat interval: {avg_interval:.3f}s")

            else:
                print(f"Warning: Music file not found: {music_full_path}")

    # 將更新後的配置寫回 config.json
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print("Beat analysis complete and config.json updated.")

if __name__ == '__main__':
    preprocess_beats()