import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont

# 關卡順序定義
LEVEL_ORDER = ["level_1", "level_2", "level_3"]

# 畫面設定
SCREEN_WIDTH, SCREEN_HEIGHT = 1440, 960

# --- 輔助函式 ---

def desaturate_image(img):
    """將圖片去飽和度以表示禁用狀態"""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    bgr = img[:, :, :3]
    alpha = img[:, :, 3] if img.shape[2] == 4 else None
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    if alpha is not None:
        return cv2.merge([gray, gray, gray, alpha])
    else:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def load_config(level_name):
    """讀取指定關卡的配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if level_name not in config:
                raise ValueError(f"未找到關卡配置：{level_name}")
            return config[level_name]
    except Exception as e:
        raise Exception(f"配置文件读取失败：{str(e)}")

def show_loading_screen(screen):
    """顯示載入畫面"""
    screen.fill((0, 0, 0))  # 黑屏
    try:
        font = ImageFont.truetype("NotoSansTC-Black.ttf", 60)
    except IOError:
        font = ImageFont.load_default()

    # 建立一個 PIL Image 來繪製文字
    pil_img = Image.new('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    
    text = "Loading..."
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (SCREEN_WIDTH - text_width) // 2
    y = (SCREEN_HEIGHT - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # 將 PIL Image 轉換為 Pygame Surface 並顯示
    pygame_surface = pygame.image.frombuffer(pil_img.tobytes(), pil_img.size, 'RGB')
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()
    pygame.time.wait(500) # 至少顯示 0.5 秒

def run_level(screen, cap, level_name):
    """執行單一關卡的遊戲主迴圈"""
    
    # --- 關卡初始化 ---
    current_level_index = LEVEL_ORDER.index(level_name)
    config = load_config(level_name)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 資源路徑
    bg_path = os.path.join(current_dir, config["background"])
    music_path = os.path.join(current_dir, config["music"])
    choose_frame_path = os.path.join(current_dir, config["choose_frame"])

    # 讀取背景
    bg_image = cv2.imread(bg_path)
    if bg_image is None:
        raise FileNotFoundError(f"❌ 背景圖片讀取失敗: {bg_path}")
    bg_image_resized = cv2.resize(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # 攝影機畫面設定
    cam_x, cam_y = 40, 210
    cam_w, cam_h = 700, 610

    # 音效初始化
    pygame.mixer.quit() # 先退出之前的 mixer
    pygame.mixer.init() # 重新初始化
    pygame.mixer.music.load(music_path)

    # Mediapipe Pose 初始化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 讀取所有圖片資源...
    action_resources = {}
    for action_name, action_config in config["actions"].items():
        sound_file = action_config.get("sound")
        image_file = action_config.get("image")
        word_image_file = action_config.get("word_image")
        action_resources[action_name] = {
            "sound": pygame.mixer.Sound(os.path.join(current_dir, sound_file)) if sound_file else None,
            "image": cv2.imread(os.path.join(current_dir, image_file), cv2.IMREAD_UNCHANGED) if image_file else None,
            "word_image": cv2.imread(os.path.join(current_dir, word_image_file), cv2.IMREAD_UNCHANGED) if word_image_file else None,
            "positions": action_config.get("positions", [])
        }
        
    img_right = cv2.imread(os.path.join(current_dir, "02.png"), cv2.IMREAD_UNCHANGED)
    img_left = cv2.imread(os.path.join(current_dir, "03.png"), cv2.IMREAD_UNCHANGED)
    img_head = cv2.imread(os.path.join(current_dir, "04.png"), cv2.IMREAD_UNCHANGED)
    img_open = cv2.imread(os.path.join(current_dir, "05.png"), cv2.IMREAD_UNCHANGED)
    word_right = cv2.imread(os.path.join(current_dir, "word2.png"), cv2.IMREAD_UNCHANGED)
    word_left = cv2.imread(os.path.join(current_dir, "word3.png"), cv2.IMREAD_UNCHANGED)
    word_head = cv2.imread(os.path.join(current_dir, "word4.png"), cv2.IMREAD_UNCHANGED)
    word_open = cv2.imread(os.path.join(current_dir, "word5.png"), cv2.IMREAD_UNCHANGED)
    choose_bg = cv2.imread(choose_frame_path, cv2.IMREAD_UNCHANGED)
    
    finish_popup = cv2.imread(os.path.join(current_dir, "finish.png"), cv2.IMREAD_UNCHANGED)
    btn1_img = cv2.imread(os.path.join(current_dir, "light1.png"), cv2.IMREAD_UNCHANGED)
    btn2_img = cv2.imread(os.path.join(current_dir, "light2.png"), cv2.IMREAD_UNCHANGED)
    btn3_img = cv2.imread(os.path.join(current_dir, "light3.png"), cv2.IMREAD_UNCHANGED)
    btn4_img = cv2.imread(os.path.join(current_dir, "light4.png"), cv2.IMREAD_UNCHANGED)
    
    end_btn = cv2.imread(os.path.join(current_dir, "end.png"), cv2.IMREAD_UNCHANGED)
    next_btn = cv2.imread(os.path.join(current_dir, "next.png"), cv2.IMREAD_UNCHANGED)
    next2_btn = cv2.imread(os.path.join(current_dir, "next2.png"), cv2.IMREAD_UNCHANGED)
    end_btn_disabled = desaturate_image(end_btn.copy())
    next_btn_disabled = desaturate_image(next_btn.copy())
    
    # 圖片及按鈕位置...
    img_right_x, img_right_y, img_right_w, img_right_h = 978, 445, 236, 334 
    img_left_x, img_left_y, img_left_w, img_left_h = 1200, 445, 236, 334
    img_head_x, img_head_y, img_head_w, img_head_h = 764, 445, 236, 334
    img_open_x, img_open_y, img_open_w, img_open_h = 542, 445, 234, 334
    img_open2_x, img_open2_y, img_open2_w, img_open2_h = 978, 445, 236, 334
    img_head2_x, img_head2_y, img_head2_w, img_head2_h = 1200, 445, 236, 334
    img_word2_x, img_word2_y, img_word2_w, img_word2_h = 1063, 800, 75, 22
    img_word3_x, img_word3_y, img_word3_w, img_word3_h = 1280, 800, 75, 22
    img_word4_x, img_word4_y, img_word4_w, img_word4_h = 830, 800, 100, 22
    img_word5_x, img_word5_y, img_word5_w, img_word5_h = 610, 800, 100, 22
    img_word5_2_x, img_word5_2_y, img_word5_2_w, img_word5_2_h = 1063, 800, 75, 22
    img_word4_2_x, img_word4_2_y, img_word4_2_w, img_word4_2_h = 1280, 800, 75, 22
    choose_positions = [(540, 335), (760, 335), (980, 335), (1200, 335)]
    choose_width, choose_height = 235, 530
    end_btn_x, end_btn_y, end_btn_w, end_btn_h = 853, 65, 210, 50
    next_btn_x, next_btn_y, next_btn_w, next_btn_h = 1116, 65, 210, 50
    next2_btn_x, next2_btn_y, next2_btn_w, next2_btn_h = 610, 565, 210, 50
    
    btn1_x, btn1_y, btn1_w, btn1_h = config["button_positions"]["btn1"]["x"], config["button_positions"]["btn1"]["y"], config["button_positions"]["btn1"]["w"], config["button_positions"]["btn1"]["h"]
    btn2_x, btn2_y, btn2_w, btn2_h = config["button_positions"]["btn2"]["x"], config["button_positions"]["btn2"]["y"], config["button_positions"]["btn2"]["w"], config["button_positions"]["btn2"]["h"]
    btn3_x, btn3_y, btn3_w, btn3_h = config["button_positions"]["btn3"]["x"], config["button_positions"]["btn3"]["y"], config["button_positions"]["btn3"]["w"], config["button_positions"]["btn3"]["h"]
    btn4_x, btn4_y, btn4_w, btn4_h = config["button_positions"]["btn4"]["x"], config["button_positions"]["btn4"]["y"], config["button_positions"]["btn4"]["w"], config["button_positions"]["btn4"]["h"]

    # 遊戲狀態變數初始化
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    clock = pygame.time.Clock()
    
    prev_right, prev_left, prev_head, prev_open = False, False, False, False
    lamp1_alpha, lamp2_alpha, lamp3_alpha, lamp4_alpha = 0.1, 0.1, 0.1, 0.1
    sound_triggered_in_current_window = False
    countdown_started, countdown_completed = False, False
    hold_start_time, post_countdown_start_time = None, None
    show_finish_popup, finish_triggered = False, False
    score_total, combo_count = 0, 0
    music_start_time = None
    last_sound_time = time.time()
    last_switch_time = time.time()
    highlight_index, next_highlight_index = 0, 1
    last_motion_switch_time = time.time()
    
    # 從 config 讀取參數
    countdown_time = config["countdown_time"]
    motion_switch_interval = config["motion_switch_interval"]
    lamp_duration = config["lamp_duration"]
    sound_interval = config["sound_interval"]
    action_window = config["action_window"]
    highlight_interval = 0.75 if level_name == "level_3" else 1.5
    motion_combinations = config["motion_combinations"]
    current_motion_index = 0

    # --- 函式定義 (部分移入 run_level) ---
    def overlay_image(background, overlay, x, y, width, height):
        """疊圖函式"""
        overlay_resized = cv2.resize(overlay, (width, height))
        h, w = overlay_resized.shape[:2]
        bg_h, bg_w = background.shape[:2]
        if x + w > bg_w or y + h > bg_h or x < 0 or y < 0: return background
        if overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] +
                                               alpha_l * background[y:y+h, x:x+w, c])
        else:
            background[y:y+h, x:x+w] = overlay_resized
        return background

    def overlay_image_with_alpha(background, overlay, x, y, w, h, alpha):
        """帶透明度的叠加函数"""
        overlay_resized = cv2.resize(overlay, (w, h))
        if overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0 * alpha
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] +
                                            alpha_l * background[y:y+h, x:x+w, c])
        return background
        
    def create_rounded_mask(width, height, radius):
        """數位圓角遮罩函式"""
        mask = np.zeros((height, width), dtype=np.uint8)
        rect = (0, 0, width, height)
        submask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(submask)
        draw.rounded_rectangle(rect, radius=radius, fill=255)
        return np.array(submask)

    def get_image_position(action_name):
        """從配置中读取图片位置"""
        positions = config.get("image_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None

    def get_word_position(action_name):
        """從配置中读取单词位置"""
        positions = config.get("word_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None
        
    def is_full_body_visible(landmarks):
        """是否全身可見（含 debug）"""
        visible_parts = [0, 11, 12, 15, 16, 23, 24, 27, 28]
        for i in visible_parts:
            if landmarks[i].visibility < 0.2:
                # print(f"🔴 visibility 太低：id={i}, visibility={landmarks[i].visibility:.2f}")
                return False
        y_coords = [landmarks[i].y for i in visible_parts]
        if min(y_coords) <= -0.1 or max(y_coords) >= 1.1:
            # print(f"🔴 y 超出範圍：min_y={min(y_coords):.2f}, max_y={max(y_coords):.2f}")
            return False
        return True

    def are_legs_together(landmarks):
        """雙腳併攏判斷（含 debug）"""
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        ankle_dist = abs(l_ankle.x - r_ankle.x)
        if ankle_dist < 0.3:
            return True
        else:
            # print(f"🔴 腳未併攏，距離={ankle_dist:.3f}")
            return False

    def detect_pose_action(landmarks):
        """動作判斷"""
        l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        l_wrist = np.array([landmarks[15].x, landmarks[15].y])
        r_wrist = np.array([landmarks[16].x, landmarks[16].y])
        l_ear = np.array([landmarks[7].x, landmarks[7].y])
        r_ear = np.array([landmarks[8].x, landmarks[8].y])

        actions = {"open": False, "head": False, "left": False, "right": False}

        height_similar = abs(l_wrist[1] - r_wrist[1]) < 0.05
        near_shoulder = abs(l_wrist[1] - l_shoulder[1]) < 0.05 and abs(r_wrist[1] - r_shoulder[1]) < 0.05
        hands_apart = abs(l_wrist[0] - r_wrist[0]) > 0.7
        too_high = l_wrist[1] < l_ear[1] and r_wrist[1] < r_ear[1]
        actions["open"] = height_similar and near_shoulder and hands_apart and not too_high

        both_above_shoulder = l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1]
        near_ear = abs(l_wrist[0] - l_ear[0]) < 0.15 and abs(r_wrist[0] - r_ear[0]) < 0.15
        height_diff_ok = abs(l_wrist[1] - r_wrist[1]) < 0.15
        actions["head"] = both_above_shoulder and near_ear and height_diff_ok

        actions["left"] = l_wrist[1] < l_shoulder[1] - 0.1 and r_wrist[1] > r_shoulder[1] + 0.1
        actions["right"] = r_wrist[1] < r_shoulder[1] - 0.1 and l_wrist[1] > l_shoulder[1] + 0.1
        
        return actions

    # --- 遊戲主迴圈 ---
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = bg_image_resized.copy()
        frame = cv2.flip(frame, 1)

        # 攝影機畫面處理...
        orig_h, orig_w = frame.shape[:2]
        scale_factor = max(cam_w / orig_w, cam_h / orig_h)
        resized_w = int(orig_w * scale_factor)
        resized_h = int(orig_h * scale_factor)
        resized_frame = cv2.resize(frame, (resized_w, resized_h))
        x_start = (resized_w - cam_w) // 2
        y_start = (resized_h - cam_h) // 2
        cropped_frame = resized_frame[y_start:y_start + cam_h, x_start:x_start + cam_w]
        
        # 處理整個畫面（不再分左右）
        result = pose.process(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

        # if result.pose_landmarks:
        #     # print("🟢 偵測到骨架")
        #     mp_drawing.draw_landmarks(
        #         cropped_frame,
        #         # result.pose_landmarks,
        #         mp_pose.POSE_CONNECTIONS,
        #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #     )
        # else:
            # print("🔴 沒偵測到骨架")

        # 初始化動作的狀態變數
        right_detected = False
        left_detected = False
        head_detected = False 
        open_detected = False
        full_body_ready = False

        # 預設狀態：尚未準備、沒有任何動作被偵測
        ready = False
        open_detected = left_detected = right_detected = head_detected = False
        full_body_ready_now = False

        if result.pose_landmarks:
            landmarks = result.pose_landmarks
            full_body_ready = is_full_body_visible(landmarks) and are_legs_together(landmarks)

            if full_body_ready and not countdown_started and not countdown_completed:
                hold_start_time = hold_start_time or time.time()
                elapsed = time.time() - hold_start_time
                if elapsed >= 3:
                    countdown_started = True  # 開始倒數
                    countdown_start_time = time.time()
            else:
                hold_start_time = None

            if countdown_started and not countdown_completed:
                countdown_elapsed = time.time() - countdown_start_time
                seconds_left = 3 - int(countdown_elapsed)
                if seconds_left <= 0:
                    countdown_completed = True
                    post_countdown_start_time = time.time()  # ✅ 記錄音樂先行播放的時間
                    pygame.mixer.music.play()  # ✅ 倒數完成 + 音樂先行播放設定時間，才進入主邏輯
                    music_start_time = time.time()  # ✅ 音樂播放時間起點
                else:
                    # 繪製倒數數字（白色、120號）於實拍畫面中央
                    pil_img = Image.fromarray(cropped_frame.copy())
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        countdown_font = ImageFont.truetype("NotoSansTC-Black.ttf", 120)
                    except:
                        countdown_font = ImageFont.load_default()
                    text = str(seconds_left)
                    text_size = draw.textbbox((0, 0), text, font=countdown_font)
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]
                    # ✅ 直接用實拍畫面的寬高來置中
                    frame_w, frame_h = cropped_frame.shape[1], cropped_frame.shape[0]
                    x = (frame_w - text_width) // 2
                    y = (frame_h - text_height) // 3
                    draw.text((x, y), text, fill=(255, 255, 255), font=countdown_font)
                    cropped_frame = np.array(pil_img)
            
            # 🔁 額外獨立檢查 full_body_ready（用於提示語顯示，不受 countdown 影響）
            if result.pose_landmarks:
                landmarks = result.pose_landmarks
                full_body_ready_now = is_full_body_visible(landmarks) and are_legs_together(landmarks)
            else:
                full_body_ready_now = False

            # ✅ 倒數完成後才執行動作辨識
             # ✅ 倒數完成 + 音樂播放滿 1.5 秒後，才開始動作辨識。 結束彈窗出現後停止主迴圈
            if countdown_completed and time.time() - post_countdown_start_time >= 1.5 and not show_finish_popup:
                detected_actions = detect_pose_action(landmarks)
                open_detected |= detected_actions["open"]
                left_detected  |= detected_actions["left"]
                right_detected |= detected_actions["right"]
                head_detected  |= detected_actions["head"]
            else:
                full_body_ready = False

        # 獲取當前時間
        current_time = time.time()

        # 音樂播放完畢 → 顯示結束彈窗（只觸發一次）
        if countdown_completed and not finish_triggered and not pygame.mixer.music.get_busy():
            finish_triggered = True
            show_finish_popup = True
            # print("✅ 音樂播放結束，觸發 finish.png")

        # ✅ 主流程邏輯僅在倒數結束後執行
         # ✅ 倒數完成 + 音樂播放滿 1.5 秒後，才開始主迴圈。 結束彈窗出現後停止主迴圈
        if countdown_completed and time.time() - post_countdown_start_time >= 1.5 and not show_finish_popup:

            # 每 highlight_interval 秒切換一次 highlight_index（循環切換四個位置）
            if current_time - last_switch_time > highlight_interval:
                highlight_index = next_highlight_index  # 更新当前选择框位置
                next_highlight_index = (highlight_index + 1) % 4  # 计算下一个选择框位置
                last_switch_time = current_time  # 更新时间
                sound_triggered_in_current_window = False  # 重置声音触发标记

            # 计算当前时间在选择框切换周期中的位置
            time_in_cycle = current_time - last_switch_time
            is_in_previous_window = time_in_cycle < action_window  # 是否在前一个选择框的判定窗口
            is_in_next_window = time_in_cycle > (highlight_interval - action_window)  # 是否在下一个选择框的判定窗口
            
            # 检查是否可以播放新的声音（与上一次声音间隔足够）
            can_play_sound = current_time - last_sound_time > sound_interval

            # === 疊加顯示當前動作組合中的 GIF 動作圖 ===
            current_combination = motion_combinations[current_motion_index]
            
            # 根據關卡決定疊圖方式
            if level_name == "level_3":
                for motion in current_combination:
                    res = action_resources.get(motion)
                    if not res:
                        continue
                    img_pos = get_image_position(motion)
                    if res["image"] is not None and img_pos:
                        canvas = overlay_image(canvas, res["image"], img_pos[0], img_pos[1], img_pos[2], img_pos[3])
                    word_pos = get_word_position(motion)
                    if res["word_image"] is not None and word_pos:
                        canvas = overlay_image(canvas, res["word_image"], word_pos[0], word_pos[1], word_pos[2], word_pos[3])
            else:
                # 遍歷並疊圖（原本 level_1、level_2 的邏輯）
                for idx, motion in enumerate(current_combination):
                    if motion == "right":
                        canvas = overlay_image(canvas, img_right, img_right_x, img_right_y, img_right_w, img_right_h)
                        canvas = overlay_image(canvas, word_right, img_word2_x, img_word2_y, img_word2_w, img_word2_h)
                    elif motion == "left":
                        canvas = overlay_image(canvas, img_left, img_left_x, img_left_y, img_left_w, img_left_h)
                        canvas = overlay_image(canvas, word_left, img_word3_x, img_word3_y, img_word3_w, img_word3_h)
                    elif motion == "head":
                        if idx == 1:  # 第二個位置
                            canvas = overlay_image(canvas, img_head, img_head_x, img_head_y, img_head_w, img_head_h)
                            canvas = overlay_image(canvas, word_head, img_word4_x, img_word4_y, img_word4_w, img_word4_h)
                        elif idx == 3:  # 第四個位置
                            canvas = overlay_image(canvas, img_head, img_head2_x, img_head2_y, img_head2_w, img_head2_h)
                            canvas = overlay_image(canvas, word_head, img_word4_2_x, img_word4_2_y, img_word4_2_w, img_word4_2_h)
                    elif motion == "open":
                        if idx == 0:  # 第一個位置
                            canvas = overlay_image(canvas, img_open, img_open_x, img_open_y, img_open_w, img_open_h)
                            canvas = overlay_image(canvas, word_open, img_word5_x, img_word5_y, img_word5_w, img_word5_h)
                        elif idx == 2:  # 第三個位置
                            canvas = overlay_image(canvas, img_open, img_open2_x, img_open2_y, img_open2_w, img_open2_h)
                            canvas = overlay_image(canvas, word_open, img_word5_2_x, img_word5_2_y, img_word5_2_w, img_word5_2_h)

            # 检查是否可以播放新的声音
            # if not sound_triggered_in_current_window:  # 添加条件：当前窗口未触发过声音
                current_action = current_combination[highlight_index]
                # if current_action in action_resources:
                resource = action_resources[current_action]
                    
                    # 检测动作并播放对应音效
                action_detected = False
                if current_action == "open" and open_detected and not prev_open:
                        resource["sound"].play()
                        action_detected = True
                        prev_open = True
                        print("level Work")
                if (current_action == "head" or current_action.startswith("head")) and head_detected and not prev_head:
                        resource["sound"].play()
                        action_detected = True
                        prev_head = True
                        print("level Work")
                if current_action == "left" and left_detected and not prev_left:
                        resource["sound"].play()
                        action_detected = True
                        prev_left = True
                        print("level Work")
                if current_action == "right" and right_detected and not prev_right:
                        resource["sound"].play()
                        action_detected = True
                        prev_right = True
                        print("level Work")
                    
                    # 如果检测到动作且在正确的时间窗口内
                if action_detected:
                        # if (highlight_index in resource["positions"] or
                        #     (is_in_next_window and next_highlight_index in resource["positions"]) or
                        #     (is_in_previous_window and ((highlight_index - 1) % 4) in resource["positions"])):
                            if resource["sound"]:
                                # resource["sound"].play()
                                last_sound_time = current_time
                                score_total += 10
                                combo_count += 1
                                sound_triggered_in_current_window = True  # 标记当前窗口已触发声音

                # 重置未检测到的动作状态
                if not open_detected:
                    prev_open = False
                if not head_detected:
                    prev_head = False
                if not left_detected:
                    prev_left = False
                if not right_detected:
                    prev_right = False

            # 🔁 燈光預設關閉
            lamp1_active = lamp2_active = lamp3_active = lamp4_active = False

            # 根据选择框位置和动作检测结果来决定指示灯的透明度
            # 重置所有指示灯为半透明
            lamp1_alpha = lamp2_alpha = lamp3_alpha = lamp4_alpha = 0.1

            # 获取当前组合中的动作
            current_motion = current_combination[highlight_index]
            
            # 依關卡使用不同的指示燈判斷邏輯
            if level_name == "level_3":
                if current_motion == "open" and open_detected:
                    lamp1_alpha = 1.0
                elif current_motion in ("head", "head_1") and head_detected:
                    lamp2_alpha = 1.0
                elif current_motion == "head_2" and head_detected:
                    lamp3_alpha = 1.0
                elif current_motion == "left" and left_detected:
                    lamp4_alpha = 1.0
                elif current_motion == "right" and right_detected:
                    lamp4_alpha = 1.0
            else:
                if highlight_index == 0 and open_detected:
                    lamp1_alpha = 1.0
                elif highlight_index == 1 and head_detected:
                    lamp2_alpha = 1.0
                elif highlight_index == 2:
                    if (current_motion == "right" and right_detected) or \
                       (current_motion == "open" and open_detected):
                        lamp3_alpha = 1.0
                elif highlight_index == 3:
                    if (current_motion == "left" and left_detected) or \
                       (current_motion == "head" and head_detected):
                        lamp4_alpha = 1.0

            # 使用新的叠加函数和透明度
            canvas = overlay_image_with_alpha(canvas, btn1_img, btn1_x, btn1_y, btn1_w, btn1_h, lamp1_alpha)
            canvas = overlay_image_with_alpha(canvas, btn2_img, btn2_x, btn2_y, btn2_w, btn2_h, lamp2_alpha)
            canvas = overlay_image_with_alpha(canvas, btn3_img, btn3_x, btn3_y, btn3_w, btn3_h, lamp3_alpha)
            canvas = overlay_image_with_alpha(canvas, btn4_img, btn4_x, btn4_y, btn4_w, btn4_h, lamp4_alpha)

            # 每 6 秒切換一次當前動作組合
            if current_time - last_motion_switch_time > motion_switch_interval:
                current_motion_index = (current_motion_index + 1) % len(motion_combinations)
                last_motion_switch_time = current_time

        # 將 canvas 轉成 PIL 圖片以便繪字
        canvas_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas_pil)

        # ✅ 裁切：確保攝影畫面大小為 cam_w x cam_h
        cropped_frame = cv2.resize(cropped_frame, (cam_w, cam_h))

        # 建立圓角遮罩
        mask = create_rounded_mask(cam_w, cam_h, radius=20)  # 半徑可自行調整

        # 將 BGR 攝影畫面與單通道 mask 結合（加一個 alpha）
        cropped_bgra = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2BGRA)
        cropped_bgra[:, :, 3] = mask  # 把 mask 加到 alpha 通道
       
        # 选择框叠加
        choose_x, choose_y = choose_positions[highlight_index]
        canvas = overlay_image(canvas, choose_bg, choose_x, choose_y, choose_width, choose_height)

        # 疊加到動作圖 canvas
        # === 疊加顯示當前動作組合中的 GIF 動作圖 ===
        current_combination = motion_combinations[current_motion_index]
        
        if level_name == "level_3":
            # 依 config 給定的位置疊圖，支援 head_1 / head_2 / empty 等動作
            for motion in current_combination:
                res = action_resources.get(motion)
                if not res:
                    continue
                img_pos = get_image_position(motion)
                if res["image"] is not None and img_pos:
                    canvas = overlay_image(canvas, res["image"], img_pos[0], img_pos[1], img_pos[2], img_pos[3])
                word_pos = get_word_position(motion)
                if res["word_image"] is not None and word_pos:
                    canvas = overlay_image(canvas, res["word_image"], word_pos[0], word_pos[1], word_pos[2], word_pos[3])
        else:
            # 原本 level_1、level_2 的疊圖邏輯
            for idx, motion in enumerate(current_combination):
                if motion == "right":
                    canvas = overlay_image(canvas, img_right, img_right_x, img_right_y, img_right_w, img_right_h)
                    canvas = overlay_image(canvas, word_right, img_word2_x, img_word2_y, img_word2_w, img_word2_h)
                elif motion == "left":
                    canvas = overlay_image(canvas, img_left, img_left_x, img_left_y, img_left_w, img_left_h)
                    canvas = overlay_image(canvas, word_left, img_word3_x, img_word3_y, img_word3_w, img_word3_h)
                elif motion == "head":
                    if idx == 1:
                        canvas = overlay_image(canvas, img_head, img_head_x, img_head_y, img_head_w, img_head_h)
                        canvas = overlay_image(canvas, word_head, img_word4_x, img_word4_y, img_word4_w, img_word4_h)
                    elif idx == 3:
                        canvas = overlay_image(canvas, img_head, img_head2_x, img_head2_y, img_head2_w, img_head2_h)
                        canvas = overlay_image(canvas, word_head, img_word4_2_x, img_word4_2_y, img_word4_2_w, img_word4_2_h)
                elif motion == "open":
                    if idx == 0:
                        canvas = overlay_image(canvas, img_open, img_open_x, img_open_y, img_open_w, img_open_h)
                        canvas = overlay_image(canvas, word_open, img_word5_x, img_word5_y, img_word5_w, img_word5_h)
                    elif idx == 2:
                        canvas = overlay_image(canvas, img_open, img_open2_x, img_open2_y, img_open2_w, img_open2_h)
                        canvas = overlay_image(canvas, word_open, img_word5_2_x, img_word5_2_y, img_word5_2_w, img_word5_2_h)

        # 根據狀態選擇對應圖片（紅色或綠色）
        # btn1 = btn1_green if lamp1_active else btn1_red
        # btn2 = btn2_green if lamp2_active else btn2_red
        # btn3 = btn3_green if lamp3_active else btn3_red
        # btn4 = btn4_green if lamp4_active else btn4_red

        # 按钮叠加
        # canvas = overlay_image(canvas, btn1, btn1_x, btn1_y, btn1_w, btn1_h)
        # canvas = overlay_image(canvas, btn2, btn2_x, btn2_y, btn2_w, btn2_h)
        # canvas = overlay_image(canvas, btn3, btn3_x, btn3_y, btn3_w, btn3_h)
        # canvas = overlay_image(canvas, btn4, btn4_x, btn4_y, btn4_w, btn4_h)

        # 使用新的叠加函数和透明度
        canvas = overlay_image_with_alpha(canvas, btn1_img, btn1_x, btn1_y, btn1_w, btn1_h, lamp1_alpha)
        canvas = overlay_image_with_alpha(canvas, btn2_img, btn2_x, btn2_y, btn2_w, btn2_h, lamp2_alpha)
        canvas = overlay_image_with_alpha(canvas, btn3_img, btn3_x, btn3_y, btn3_w, btn3_h, lamp3_alpha)
        canvas = overlay_image_with_alpha(canvas, btn4_img, btn4_x, btn4_y, btn4_w, btn4_h, lamp4_alpha)

        # 根據當前關卡決定按鈕狀態
        prev_button_img = end_btn_disabled if current_level_index <= 0 else end_btn
        next_button_img = next_btn_disabled if current_level_index < 0 or current_level_index >= len(LEVEL_ORDER) - 1 else next_btn

        canvas = overlay_image(canvas, prev_button_img, end_btn_x, end_btn_y, end_btn_w, end_btn_h)
        canvas = overlay_image(canvas, next_button_img, next_btn_x, next_btn_y, next_btn_w, next_btn_h)

        # 疊圖到背景（將攝影畫面貼到 canvas 上指定區域）
        roi = canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
        alpha_s = cropped_bgra[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            roi[:, :, c] = (alpha_s * cropped_bgra[:, :, c] +
                            alpha_l * roi[:, :, c])
        canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = roi

        # 建立 PIL 畫布，準備畫提示文字
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)

        # 載入字型
        try:
            font = ImageFont.truetype("NotoSansTC-Black.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        # 提示文字設定
        lines = ["請保持全身在拍攝畫面以內", "並將雙腿盡量併攏"]
        line_spacing = 10
        line_heights = []
        line_widths = []

        # 計算文字大小
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_widths.append(w)
            line_heights.append(h)

        total_text_height = sum(line_heights) + line_spacing

        # ✅ 中央顯示提示（僅當未準備好時）
        if not full_body_ready_now:
            start_y = cam_y + (cam_h - total_text_height) // 2
            for i, line in enumerate(lines):
                w = line_widths[i]
                x = cam_x + (cam_w // 2) - w // 2
                y = start_y + i * (line_heights[i] + line_spacing)
                draw.text((x, y), line, font=font, fill=(0, 120, 255))

        # === 顯示總分 ===
        score_font = ImageFont.truetype("NotoSansTC-Black.ttf", 48)
        score_text = f"{score_total:03}"   # 格式成三位數
        draw.text((790, 210), score_text, font=score_font, fill=(1, 125, 244))

        # === 顯示連擊數 ===
        combo_font = ImageFont.truetype("NotoSansTC-Black.ttf", 36)
        combo_text = f"{combo_count:02}"   # 格式成兩位數
        draw.text((1332, 218), combo_text, font=combo_font, fill=(1, 125, 244))

        # === 顯示倒數計時（從60到0） ===
        if music_start_time:
            timer_font = ImageFont.truetype("NotoSansTC-Black.ttf", 28)  
            elapsed_time = int(time.time() - music_start_time)
            remaining_time = max(0, countdown_time - elapsed_time)
            time_text = f"00:{remaining_time:02}"
            draw.text((1030, 225), time_text, font=timer_font, fill=(80, 113, 135))  # 棕色

        # 將 PIL 畫布轉回 NumPy 陣列
        canvas = np.array(pil_img)

        # ✅ 疊加 finish 彈窗（如有啟用）
        if show_finish_popup:
            canvas = overlay_image(canvas, finish_popup, 0, 0, 1440, 960)
            # ✅ 疊加 next2 按鈕在彈窗上
            canvas = overlay_image(canvas, next2_btn, next2_btn_x, next2_btn_y, next2_btn_w, next2_btn_h)


        # 顯示畫面（顯示完整大小 1440 x 960）
        # 轉換為RGB格式（Pygame需要）
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # 明確指定圖像尺寸
        height, width, _ = canvas_rgb.shape

        # 創建Pygame表面 - 使用正確的尺寸
        pygame_surface = pygame.image.frombuffer(
            canvas_rgb.tobytes(), 
            (width, height),  # 直接使用圖像的寬高
            "RGB"
        )

        # 顯示畫面（確保縮放到整個視窗）
        scaled_surface = pygame.transform.scale(pygame_surface, (1440, 960))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)
        
        # --- 事件處理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit" # 返回 "quit" 表示要關閉遊戲
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return "quit"
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                if show_finish_popup:
                    if next2_btn_x <= mouse_x <= next2_btn_x + next2_btn_w and \
                       next2_btn_y <= mouse_y <= next2_btn_y + next2_btn_h:
                        if current_level_index < len(LEVEL_ORDER) - 1:
                            return "next_level"
                        else:
                            return "quit" # 最後一關結束後就退出
                else:
                    # 上一關
                    if end_btn_x <= mouse_x <= end_btn_x + end_btn_w and \
                       end_btn_y <= mouse_y <= end_btn_y + end_btn_h and \
                       current_level_index > 0:
                        return "prev_level"
                    
                    # 下一關
                    if next_btn_x <= mouse_x <= next_btn_x + next_btn_w and \
                       next_btn_y <= mouse_y <= next_btn_y + next_btn_h and \
                       current_level_index < len(LEVEL_ORDER) - 1:
                        return "next_level"

def main():
    """程式主進入點，管理 Pygame 視窗和關卡流程"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("姿勢辨識系統")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    # 從命令列參數決定起始關卡，或預設為第一關
    start_level_name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in LEVEL_ORDER else LEVEL_ORDER[0]
    current_level_index = LEVEL_ORDER.index(start_level_name)

    while 0 <= current_level_index < len(LEVEL_ORDER):
        show_loading_screen(screen)
        
        level_name = LEVEL_ORDER[current_level_index]
        action = run_level(screen, cap, level_name)
        
        if action == "quit":
            break
        elif action == "next_level":
            current_level_index += 1
        elif action == "prev_level":
            current_level_index -= 1
            
    print("遊戲結束，感謝遊玩！")
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()