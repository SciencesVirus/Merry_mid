import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont

# --- 導入其他畫面模組 ---
from homepage import run_homepage
from soundselect import run_soundselect
from movements import run_movements

# --- 常數定義 ---
LEVEL_ORDER = ["level_1", "level_2", "level_3"]
SCREEN_WIDTH, SCREEN_HEIGHT = 1440, 960
ASSETS_PATH = "level_Universal"
CORRECTION_ASSETS_PATH = "Correction_2"

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
    config_path = os.path.join(ASSETS_PATH, "config.json")
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
    screen.fill((0, 0, 0))
    try:
        font = ImageFont.truetype(os.path.join(ASSETS_PATH, "NotoSansTC-Black.ttf"), 60)
    except IOError:
        font = ImageFont.load_default()

    pil_img = Image.new('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    
    text = "Loading..."
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (SCREEN_WIDTH - text_width) // 2
    y = (SCREEN_HEIGHT - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    pygame_surface = pygame.image.frombuffer(pil_img.tobytes(), pil_img.size, 'RGB')
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()
    pygame.time.wait(500)

def overlay_image(background, overlay, x, y, width, height):
    """基礎疊圖函式"""
    if overlay is None: return background
    overlay_resized = cv2.resize(overlay, (width, height))
    h, w = overlay_resized.shape[:2]
    bg_h, bg_w = background.shape[:2]
    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0: return background
    if len(overlay_resized.shape) > 2 and overlay_resized.shape[2] == 4:
        alpha_s = overlay_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] +
                                           alpha_l * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay_resized
    return background

# --- 校正畫面邏輯 ---
def run_correction_screen(screen, cap):
    """執行校正畫面主迴圈"""
    pygame.display.set_caption("校正畫面")
    clock = pygame.time.Clock()
    
    # 資源
    ding_sound = pygame.mixer.Sound(os.path.join(CORRECTION_ASSETS_PATH, "ding.wav"))
    bg_image = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "Correction_back.png"))
    yes_img = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "yes.png"), cv2.IMREAD_UNCHANGED)
    no_btn = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "no.png"), cv2.IMREAD_UNCHANGED)
    next_btn = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "next.png"), cv2.IMREAD_UNCHANGED)
    font_path = os.path.join(ASSETS_PATH, "NotoSansTC-Black.ttf")

    # Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # 尺寸
    cam_x, cam_y, cam_w, cam_h = 277, 278, 365, 560
    yes_x, yes_y, yes_w, yes_h = 350, 424, 205, 205
    btn_x, btn_y, btn_w, btn_h = 889, 814, 129, 46

    # 狀態
    success_once = False
    hold_start_time = None

    def is_full_body_visible(landmarks):
        visible_ids = [0, 11, 12, 15, 16, 23, 24, 27, 28]
        if any(landmarks[i].visibility < 0.2 for i in visible_ids): return False
        if landmarks[31].visibility < 0.2 and landmarks[32].visibility < 0.2: return False
        y_coords = [landmarks[i].y for i in visible_ids]
        return all(0.0 <= y <= 1.1 for y in y_coords)

    def are_legs_together(landmarks):
        l_ankle_x, r_ankle_x = landmarks[27].x, landmarks[28].x
        return abs(l_ankle_x - r_ankle_x) < 0.12

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        bg = cv2.resize(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        frame_resized = cv2.resize(frame, (cam_w, cam_h))
        results = pose.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        bg[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_resized[:, :, :3]

        show_text = True
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if is_full_body_visible(landmarks) and are_legs_together(landmarks):
                if hold_start_time is None:
                    hold_start_time = time.time()
                elif time.time() - hold_start_time >= 3.0 and not success_once:
                    success_once = True
                    ding_sound.play()
                show_text = False
            else:
                hold_start_time = None
        
        if show_text:
            pil_img = Image.fromarray(bg)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(font_path, 28)
            except:
                font = ImageFont.load_default()
            lines = ["請保持全身在拍攝畫面以內", "並將雙腿盡量併攏"]
            line_heights = [draw.textbbox((0,0), l, font=font)[3] for l in lines]
            total_text_height = sum(line_heights) + 10 * (len(lines) - 1)
            start_y = cam_y + (cam_h - total_text_height) // 2
            current_y = start_y
            for i, line in enumerate(lines):
                line_width = draw.textbbox((0,0), line, font=font)[2]
                draw.text((cam_x + (cam_w - line_width) // 2, current_y), line, font=font, fill=(0, 100, 255))
                current_y += line_heights[i] + 10
            bg = np.array(pil_img)

        if hold_start_time and not success_once:
            elapsed = time.time() - hold_start_time
            seconds_left = 4 - int(elapsed + 1)
            if 1 <= seconds_left <= 3:
                pil_img = Image.fromarray(bg)
                draw = ImageDraw.Draw(pil_img)
                try:
                    countdown_font = ImageFont.truetype(font_path, 120)
                except:
                    countdown_font = ImageFont.load_default()
                text = str(seconds_left)
                text_bbox = draw.textbbox((0, 0), text, font=countdown_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.text((cam_x + (cam_w - text_width) // 2, cam_y + (cam_h - text_height) // 3), text, fill=(255, 255, 255), font=countdown_font)
                bg = np.array(pil_img)

        if success_once:
            bg = overlay_image(bg, yes_img, yes_x, yes_y, yes_w, yes_h)
            bg = overlay_image(bg, next_btn, btn_x, btn_y, btn_w, btn_h)
        else:
            bg = overlay_image(bg, no_btn, btn_x, btn_y, btn_w, btn_h)

        canvas_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        pygame_surface = pygame.image.frombuffer(canvas_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
        screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if success_once:
                    mx, my = pygame.mouse.get_pos()
                    if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                        return "start_game"

# --- 遊戲關卡邏輯 ---
def run_level(screen, cap, level_name, selected_sound_pack):
    """執行單一關卡的遊戲主迴圈"""
    
    current_level_index = LEVEL_ORDER.index(level_name)
    config = load_config(level_name)
    
    # --- 建立音效對照表 ---
    sound_packs = {
        "percussion": {
            "1_Cajon_Tremolo.wav": "Sound effects/Percussion/1_Cajon_Tremolo.wav",
            "2_HandclapL_1.wav":   "Sound effects/Percussion/2_HandclapL_1.wav",
            "3_HandclapR_1.wav":   "Sound effects/Percussion/3_HandclapR_1.wav",
            "4_Cajon_Side.wav":    "Sound effects/Percussion/4_Cajon_Side.wav",
            "5_Cajon_Hit.wav":     "Sound effects/Percussion/5_Cajon_Hit.wav",
        },
        "daily": {
            "1_Cajon_Tremolo.wav": "Sound effects/Quotidien/1_ScrewsInGlass.wav",
            "2_HandclapL_1.wav":   "Sound effects/Quotidien/2_WaterBottleL.wav",
            "3_HandclapR_1.wav":   "Sound effects/Quotidien/3_WaterBottleR.wav",
            "4_Cajon_Side.wav":    "Sound effects/Quotidien/4_PropaneTank.wav",
            "5_Cajon_Hit.wav":     "Sound effects/Quotidien/5_WaterBasin.mp3",
        },
        "animal": {
            "1_Cajon_Tremolo.wav": "Sound effects/Animal/1_BirdsE.wav",
            "2_HandclapL_1.wav":   "Sound effects/Animal/2_BirdsB_L.wav",
            "3_HandclapR_1.wav":   "Sound effects/Animal/3_BirdsB_R.wav",
            "4_Cajon_Side.wav":    "Sound effects/Animal/4_BirdsD.wav",
            "5_Cajon_Hit.wav":     "Sound effects/Animal/5_BirdsA.wav",
        }
    }
    current_sound_pack_files = sound_packs.get(selected_sound_pack, sound_packs["percussion"])

    bg_path = os.path.join(ASSETS_PATH, config["background"])
    music_path = os.path.join(ASSETS_PATH, config["music"])
    choose_frame_path = os.path.join(ASSETS_PATH, config["choose_frame"])
    font_path = os.path.join(ASSETS_PATH, "NotoSansTC-Black.ttf")

    bg_image = cv2.imread(bg_path)
    if bg_image is None: raise FileNotFoundError(f"❌ 背景圖片讀取失敗: {bg_path}")
    bg_image_resized = cv2.resize(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    cam_x, cam_y, cam_w, cam_h = 66, 220, 405, 610

    pygame.mixer.music.load(music_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    action_resources = {}
    for action_name, action_config in config["actions"].items():
        # 將 config 中的 sound 作為 key
        base_sound_file = action_config.get("sound")
        image_file = action_config.get("image")
        word_image_file = action_config.get("word_image")
        
        final_sound_path = None
        if base_sound_file:
            # 從對照表中查找對應的音效路徑
            final_sound_path = current_sound_pack_files.get(base_sound_file)
            if not final_sound_path or not os.path.exists(final_sound_path):
                print(f"警告: 在音效包 '{selected_sound_pack}' 中找不到 '{base_sound_file}' 的對應音效，路徑 '{final_sound_path}' 不存在。")
                final_sound_path = None

        action_resources[action_name] = {
            "sound": pygame.mixer.Sound(final_sound_path) if final_sound_path else None,
            "image": cv2.imread(os.path.join(ASSETS_PATH, image_file), cv2.IMREAD_UNCHANGED) if image_file else None,
            "word_image": cv2.imread(os.path.join(ASSETS_PATH, word_image_file), cv2.IMREAD_UNCHANGED) if word_image_file else None,
            "positions": action_config.get("positions", [])
        }
        
    img_right = cv2.imread(os.path.join(ASSETS_PATH, "02.png"), cv2.IMREAD_UNCHANGED)
    img_left = cv2.imread(os.path.join(ASSETS_PATH, "03.png"), cv2.IMREAD_UNCHANGED)
    img_head = cv2.imread(os.path.join(ASSETS_PATH, "04.png"), cv2.IMREAD_UNCHANGED)
    img_open = cv2.imread(os.path.join(ASSETS_PATH, "05.png"), cv2.IMREAD_UNCHANGED)
    word_right = cv2.imread(os.path.join(ASSETS_PATH, "word2.png"), cv2.IMREAD_UNCHANGED)
    word_left = cv2.imread(os.path.join(ASSETS_PATH, "word3.png"), cv2.IMREAD_UNCHANGED)
    word_head = cv2.imread(os.path.join(ASSETS_PATH, "word4.png"), cv2.IMREAD_UNCHANGED)
    word_open = cv2.imread(os.path.join(ASSETS_PATH, "word5.png"), cv2.IMREAD_UNCHANGED)
    choose_bg = cv2.imread(choose_frame_path, cv2.IMREAD_UNCHANGED)
    
    finish_popup = cv2.imread(os.path.join(ASSETS_PATH, "finish.png"), cv2.IMREAD_UNCHANGED)
    btn1_img = cv2.imread(os.path.join(ASSETS_PATH, "light1.png"), cv2.IMREAD_UNCHANGED)
    btn2_img = cv2.imread(os.path.join(ASSETS_PATH, "light2.png"), cv2.IMREAD_UNCHANGED)
    btn3_img = cv2.imread(os.path.join(ASSETS_PATH, "light3.png"), cv2.IMREAD_UNCHANGED)
    btn4_img = cv2.imread(os.path.join(ASSETS_PATH, "light4.png"), cv2.IMREAD_UNCHANGED)
    
    end_btn = cv2.imread(os.path.join(ASSETS_PATH, "end.png"), cv2.IMREAD_UNCHANGED)
    next_btn = cv2.imread(os.path.join(ASSETS_PATH, "next.png"), cv2.IMREAD_UNCHANGED)
    next2_btn = cv2.imread(os.path.join(ASSETS_PATH, "next2.png"), cv2.IMREAD_UNCHANGED)
    end_btn_disabled = desaturate_image(end_btn.copy())
    next_btn_disabled = desaturate_image(next_btn.copy())
    
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
    
    btn_positions = config["button_positions"]
    btn1_x, btn1_y, btn1_w, btn1_h = btn_positions["btn1"]["x"], btn_positions["btn1"]["y"], btn_positions["btn1"]["w"], btn_positions["btn1"]["h"]
    btn2_x, btn2_y, btn2_w, btn2_h = btn_positions["btn2"]["x"], btn_positions["btn2"]["y"], btn_positions["btn2"]["w"], btn_positions["btn2"]["h"]
    btn3_x, btn3_y, btn3_w, btn3_h = btn_positions["btn3"]["x"], btn_positions["btn3"]["y"], btn_positions["btn3"]["w"], btn_positions["btn3"]["h"]
    btn4_x, btn4_y, btn4_w, btn4_h = btn_positions["btn4"]["x"], btn_positions["btn4"]["y"], btn_positions["btn4"]["w"], btn_positions["btn4"]["h"]

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
    
    countdown_time = config["countdown_time"]
    motion_switch_interval = config["motion_switch_interval"]
    action_window = config["action_window"]
    highlight_interval = 0.75 if level_name == "level_3" else 1.5
    motion_combinations = config["motion_combinations"]
    current_motion_index = 0

    def overlay_image_with_alpha(background, overlay, x, y, w, h, alpha):
        overlay_resized = cv2.resize(overlay, (w, h))
        if len(overlay_resized.shape) > 2 and overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0 * alpha
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])
        return background
        
    def create_rounded_mask(width, height, radius):
        mask = np.zeros((height, width), dtype=np.uint8)
        submask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(submask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
        return np.array(submask)

    def get_image_position(action_name):
        positions = config.get("image_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None

    def get_word_position(action_name):
        positions = config.get("word_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None

    def is_full_body_visible(landmarks):
        visible_parts = [0, 11, 12, 15, 16, 23, 24, 27, 28]
        return all(landmarks[i].visibility > 0.2 for i in visible_parts) and \
               0.0 <= min(l.y for l in landmarks) and max(l.y for l in landmarks) <= 1.1

    def are_legs_together(landmarks):
        return abs(landmarks[27].x - landmarks[28].x) < 0.3

    def detect_pose_action(landmarks):
        l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        l_wrist = np.array([landmarks[15].x, landmarks[15].y])
        r_wrist = np.array([landmarks[16].x, landmarks[16].y])
        l_ear = np.array([landmarks[7].x, landmarks[7].y])
        r_ear = np.array([landmarks[8].x, landmarks[8].y])
        actions = {"open": False, "head": False, "left": False, "right": False}
        actions["open"] = abs(l_wrist[1] - r_wrist[1]) < 0.05 and abs(l_wrist[1] - l_shoulder[1]) < 0.05 and abs(r_wrist[1] - r_shoulder[1]) < 0.05 and abs(l_wrist[0] - r_wrist[0]) > 0.7 and not (l_wrist[1] < l_ear[1] and r_wrist[1] < r_ear[1])
        actions["head"] = l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1] and abs(l_wrist[0] - l_ear[0]) < 0.15 and abs(r_wrist[0] - r_ear[0]) < 0.15 and abs(l_wrist[1] - r_wrist[1]) < 0.15
        actions["left"] = l_wrist[1] < l_shoulder[1] - 0.1 and r_wrist[1] > r_shoulder[1] + 0.1
        actions["right"] = r_wrist[1] < r_shoulder[1] - 0.1 and l_wrist[1] > l_shoulder[1] + 0.1
        return actions

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break

        canvas = bg_image_resized.copy()
        frame = cv2.flip(frame, 1)

        orig_h, orig_w = frame.shape[:2]
        scale_factor = max(cam_w / orig_w, cam_h / orig_h)
        resized_w = int(orig_w * scale_factor)
        resized_h = int(orig_h * scale_factor)
        resized_frame = cv2.resize(frame, (resized_w, resized_h))
        x_start = (resized_w - cam_w) // 2
        y_start = (resized_h - cam_h) // 2
        cropped_frame = resized_frame[y_start:y_start + cam_h, x_start:x_start + cam_w]
        
        result = pose.process(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        
        # if result.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         cropped_frame, 
        #         # result.pose_landmarks, 
        #         mp_pose.POSE_CONNECTIONS, 
        #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #         )

        right_detected, left_detected, head_detected, open_detected, full_body_ready, full_body_ready_now = False, False, False, False, False, False

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            full_body_ready = is_full_body_visible(landmarks) and are_legs_together(landmarks)
            full_body_ready_now = full_body_ready

            if full_body_ready and not countdown_started and not countdown_completed:
                if hold_start_time is None: hold_start_time = time.time()
                if time.time() - hold_start_time >= 3:
                    countdown_started = True
                    countdown_start_time = time.time()
            else:
                hold_start_time = None

            if countdown_started and not countdown_completed:
                seconds_left = 3 - int(time.time() - countdown_start_time)
                if seconds_left <= 0:
                    countdown_completed = True
                    post_countdown_start_time = time.time()
                    pygame.mixer.music.play()
                    music_start_time = time.time()
                else:
                    pil_img = Image.fromarray(cropped_frame.copy())
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        countdown_font = ImageFont.truetype(font_path, 120)
                    except:
                        countdown_font = ImageFont.load_default()
                    text = str(seconds_left)
                    text_bbox = draw.textbbox((0, 0), text, font=countdown_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.text(((cropped_frame.shape[1] - text_width) // 2, (cropped_frame.shape[0] - text_height) // 3), text, fill=(255, 255, 255), font=countdown_font)
                    cropped_frame = np.array(pil_img)

            if countdown_completed and time.time() - post_countdown_start_time >= 1.5:
                detected_actions = detect_pose_action(landmarks)
                open_detected |= detected_actions["open"]
                left_detected  |= detected_actions["left"]
                right_detected |= detected_actions["right"]
                head_detected  |= detected_actions["head"]

        current_time = time.time()
        if countdown_completed and not finish_triggered and not pygame.mixer.music.get_busy():
            finish_triggered = True
            show_finish_popup = True

        if countdown_completed and time.time() - post_countdown_start_time >= 1.5 and not show_finish_popup:
            if current_time - last_switch_time > highlight_interval:
                highlight_index = next_highlight_index
                next_highlight_index = (highlight_index + 1) % 4
                last_switch_time = current_time
                sound_triggered_in_current_window = False

            time_in_cycle = current_time - last_switch_time
            is_in_previous_window = time_in_cycle < action_window
            is_in_next_window = time_in_cycle > (highlight_interval - action_window)
            
            can_play_sound = current_time - last_sound_time > config["sound_interval"]

            # if can_play_sound and not sound_triggered_in_current_window:
        current_action_name = motion_combinations[current_motion_index][highlight_index]
        resource = action_resources.get(current_action_name)

            # if resource and resource["sound"]:
        action_detected = False
        if current_action_name == "open" and open_detected and not prev_open:
                        resource["sound"].play()
                        action_detected = True
                        print("main Work")
        if current_action_name.startswith("head") and head_detected and not prev_head:
                        resource["sound"].play()
                        action_detected = True
                        print("main Work")
        if current_action_name == "left" and left_detected and not prev_left:
                        resource["sound"].play()
                        action_detected = True
                        print("main Work")
        if current_action_name == "right" and right_detected and not prev_right:
                        resource["sound"].play()
                        action_detected = True
                        print("main Work")
                    
        if action_detected:
                        # resource["sound"].play()
                        last_sound_time = current_time
                        score_total += 10
                        combo_count += 1
                        sound_triggered_in_current_window = True
                        
        if current_action_name == "open": prev_open = True
        if current_action_name.startswith("head"): prev_head = True
        if current_action_name == "left": prev_left = True
        if current_action_name == "right": prev_right = True
            
            # --- 重置未偵測到的動作狀態 ---
        if not open_detected: prev_open = False
        if not head_detected: prev_head = False
        if not left_detected: prev_left = False
        if not right_detected: prev_right = False
            
        lamp1_alpha, lamp2_alpha, lamp3_alpha, lamp4_alpha = 0.1, 0.1, 0.1, 0.1
        current_motion = motion_combinations[current_motion_index][highlight_index]
        if level_name == "level_3":
                if current_motion == "open" and open_detected: lamp1_alpha = 1.0
                elif current_motion in ("head", "head_1") and head_detected: lamp2_alpha = 1.0
                elif current_motion == "head_2" and head_detected: lamp3_alpha = 1.0
                elif (current_motion == "left" and left_detected) or (current_motion == "right" and right_detected): lamp4_alpha = 1.0
        else:
                if highlight_index == 0 and open_detected: lamp1_alpha = 1.0
                elif highlight_index == 1 and head_detected: lamp2_alpha = 1.0
                elif highlight_index == 2 and ((current_motion == "right" and right_detected) or (current_motion == "open" and open_detected)): lamp3_alpha = 1.0
                elif highlight_index == 3 and ((current_motion == "left" and left_detected) or (current_motion == "head" and head_detected)): lamp4_alpha = 1.0
            
        if current_time - last_motion_switch_time > motion_switch_interval:
                current_motion_index = (current_motion_index + 1) % len(motion_combinations)
                last_motion_switch_time = current_time

        # --- 繪圖 ---
        mask = create_rounded_mask(cam_w, cam_h, radius=20)
        cropped_bgra = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2BGRA)
        cropped_bgra[:, :, 3] = mask
        roi = canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
        alpha_s = cropped_bgra[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            roi[:, :, c] = (alpha_s * cropped_bgra[:, :, c] + alpha_l * roi[:, :, c])
        
        choose_x, choose_y = choose_positions[highlight_index]
        canvas = overlay_image(canvas, choose_bg, choose_x, choose_y, choose_width, choose_height)
        
        current_combination = motion_combinations[current_motion_index]
        if level_name == "level_3":
            for motion in current_combination:
                res = action_resources.get(motion)
                if res:
                    img_pos, word_pos = get_image_position(motion), get_word_position(motion)
                    if res["image"] is not None and img_pos: canvas = overlay_image(canvas, res["image"], *img_pos)
                    if res["word_image"] is not None and word_pos: canvas = overlay_image(canvas, res["word_image"], *word_pos)
        else:
            for idx, motion in enumerate(current_combination):
                if motion == "right": canvas = overlay_image(canvas, img_right, img_right_x, img_right_y, img_right_w, img_right_h); canvas = overlay_image(canvas, word_right, img_word2_x, img_word2_y, img_word2_w, img_word2_h)
                elif motion == "left": canvas = overlay_image(canvas, img_left, img_left_x, img_left_y, img_left_w, img_left_h); canvas = overlay_image(canvas, word_left, img_word3_x, img_word3_y, img_word3_w, img_word3_h)
                elif motion == "head":
                    if idx == 1: canvas = overlay_image(canvas, img_head, img_head_x, img_head_y, img_head_w, img_head_h); canvas = overlay_image(canvas, word_head, img_word4_x, img_word4_y, img_word4_w, img_word4_h)
                    elif idx == 3: canvas = overlay_image(canvas, img_head, img_head2_x, img_head2_y, img_head2_w, img_head2_h); canvas = overlay_image(canvas, word_head, img_word4_2_x, img_word4_2_y, img_word4_2_w, img_word4_2_h)
                elif motion == "open":
                    if idx == 0: canvas = overlay_image(canvas, img_open, img_open_x, img_open_y, img_open_w, img_open_h); canvas = overlay_image(canvas, word_open, img_word5_x, img_word5_y, img_word5_w, img_word5_h)
                    elif idx == 2: canvas = overlay_image(canvas, img_open, img_open2_x, img_open2_y, img_open2_w, img_open2_h); canvas = overlay_image(canvas, word_open, img_word5_2_x, img_word5_2_y, img_word5_2_w, img_word5_2_h)
        
        canvas = overlay_image_with_alpha(canvas, btn1_img, btn1_x, btn1_y, btn1_w, btn1_h, lamp1_alpha)
        canvas = overlay_image_with_alpha(canvas, btn2_img, btn2_x, btn2_y, btn2_w, btn2_h, lamp2_alpha)
        canvas = overlay_image_with_alpha(canvas, btn3_img, btn3_x, btn3_y, btn3_w, btn3_h, lamp3_alpha)
        canvas = overlay_image_with_alpha(canvas, btn4_img, btn4_x, btn4_y, btn4_w, btn4_h, lamp4_alpha)
        
        prev_button_img = end_btn_disabled if current_level_index <= 0 else end_btn
        next_button_img = next_btn_disabled if current_level_index >= len(LEVEL_ORDER) - 1 else next_btn
        canvas = overlay_image(canvas, prev_button_img, end_btn_x, end_btn_y, end_btn_w, end_btn_h)
        canvas = overlay_image(canvas, next_button_img, next_btn_x, next_btn_y, next_btn_w, next_btn_h)
        
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(font_path, 24)
            score_font = ImageFont.truetype(font_path, 48)
            combo_font = ImageFont.truetype(font_path, 36)
            timer_font = ImageFont.truetype(font_path, 28)
        except IOError:
            font = score_font = combo_font = timer_font = ImageFont.load_default()
        
        if not full_body_ready_now and not countdown_completed:
            lines = ["請保持全身在拍攝畫面以內", "並將雙腿盡量併攏"]
            line_heights = [draw.textbbox((0,0), l, font=font)[3] for l in lines]
            total_text_height = sum(line_heights) + 10 * (len(lines) - 1)
            start_y = cam_y + (cam_h - total_text_height) // 2
            current_y = start_y
            for i, line in enumerate(lines):
                line_width = draw.textbbox((0,0), line, font=font)[2]
                draw.text((cam_x + (cam_w - line_width) // 2, current_y), line, font=font, fill=(0, 120, 255))
                current_y += line_heights[i] + 10
        
        draw.text((790, 210), f"{score_total:03}", font=score_font, fill=(1, 125, 244))
        draw.text((1332, 218), f"{combo_count:02}", font=combo_font, fill=(1, 125, 244))
        if music_start_time:
            remaining_time = max(0, countdown_time - int(time.time() - music_start_time))
            draw.text((1030, 225), f"00:{remaining_time:02}", font=timer_font, fill=(80, 113, 135))
        
        canvas = np.array(pil_img)

        if show_finish_popup:
            canvas = overlay_image(canvas, finish_popup, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
            canvas = overlay_image(canvas, next2_btn, next2_btn_x, next2_btn_y, next2_btn_w, next2_btn_h)

        pygame_surface = pygame.image.frombuffer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
        screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.mixer.music.stop()
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if show_finish_popup:
                    if next2_btn_x <= mx <= next2_btn_x + next2_btn_w and next2_btn_y <= my <= next2_btn_y + next2_btn_h:
                        pygame.mixer.music.stop()
                        return "next_level" if current_level_index < len(LEVEL_ORDER) - 1 else "quit"
                else:
                    if end_btn_x <= mx <= end_btn_x + end_btn_w and end_btn_y <= my <= end_btn_y + end_btn_h and current_level_index > 0:
                        pygame.mixer.music.stop()
                        return "prev_level"
                    if next_btn_x <= mx <= next_btn_x + next_btn_w and next_btn_y <= my <= next_btn_y + next_btn_h and current_level_index < len(LEVEL_ORDER) - 1:
                        pygame.mixer.music.stop()
                        return "next_level"

# --- 主程式進入點 ---
def main():
    """程式主進入點，管理 Pygame 視窗和遊戲流程"""
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    # 遊戲狀態機
    game_state = "homepage"
    selected_sound_pack = "percussion" # 預設音效包
    
    while True:
        if game_state == "homepage":
            action = run_homepage(screen)
            if action == "start":
                game_state = "sound_select"
            else: # quit
                break

        elif game_state == "sound_select":
            result = run_soundselect(screen)
            if result and result.get("action") == "confirm":
                selected_sound_pack = result.get("pack", "percussion")
                print(f"音效包選擇: {selected_sound_pack}")
                game_state = "movements"
            elif result == "back":
                 game_state = "homepage"
            else: # quit or back
                break
        
        elif game_state == "movements":
            action = run_movements(screen)
            if action == "finish":
                game_state = "correction"
            elif action == "back":
                game_state = "sound_select"
            else: # quit
                break

        elif game_state == "correction":
            action = run_correction_screen(screen, cap)
            if action == "start_game":
                game_state = "levels"
            else: # quit
                break
        
        elif game_state == "levels":
            current_level_index = 0
            while 0 <= current_level_index < len(LEVEL_ORDER):
                show_loading_screen(screen)
                
                level_name = LEVEL_ORDER[current_level_index]
                # 將選擇的音效包傳入關卡
                level_action = run_level(screen, cap, level_name, selected_sound_pack)
                
                if level_action == "quit":
                    game_state = "homepage" # 結束關卡後回到首頁
                    break 
                elif level_action == "next_level":
                    if current_level_index < len(LEVEL_ORDER) - 1:
                        current_level_index += 1
                    else:
                        game_state = "homepage" # 完成最後一關後回到首頁
                        break
                elif level_action == "prev_level":
                    current_level_index -= 1
            if game_state != "levels": # 如果是跳出迴圈，則繼續外層的 state machine
                continue

    print("遊戲結束，感謝遊玩！")
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    # 確保資源路徑正確
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
