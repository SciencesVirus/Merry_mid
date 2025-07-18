import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import librosa # New import: For audio analysis and beat tracking
import soundfile as sf # New import: Often required by librosa for loading various audio formats

# --- 導入其他畫面模組 ---
# Ensure these Python files (homepage.py, soundselect.py, movements.py, camera_manager.py, fullscreen_manager.py)
# are in the same directory as this main script, or accessible via your Python path.
from homepage import run_homepage
from soundselect import run_soundselect
from movements import run_movements
from camera_manager import CameraManager
from fullscreen_manager import FullscreenManager

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

def create_rounded_mask(width, height, radius):
    """創建圓角遮罩"""
    mask = np.zeros((height, width), dtype=np.uint8)
    submask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(submask)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    return np.array(submask)

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
def run_correction_screen(screen, camera_manager, fullscreen_manager):
    """執行校正畫面主迴圈"""
    pygame.display.set_caption("校正畫面")
    clock = pygame.time.Clock()
    
    # 設置按鈕顏色為 RGB 格式（因為是在 PIL Image 上繪製）
    camera_manager.button_bg_color = (255, 165, 0, 128)  # 淡橘色半透明 (RGB)
    fullscreen_manager.button_bg_color = (255, 165, 0, 128)  # 淡橘色半透明 (RGB)
    
    # 資源
    ding_sound = pygame.mixer.Sound(os.path.join(CORRECTION_ASSETS_PATH, "ding.wav"))
    bg_image = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "Correction_back.png"))
    yes_img = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "yes.png"), cv2.IMREAD_UNCHANGED)
    no_btn = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "no.png"), cv2.IMREAD_UNCHANGED)
    next_btn = cv2.imread(os.path.join(CORRECTION_ASSETS_PATH, "next.png"), cv2.IMREAD_UNCHANGED)
    font_path = os.path.join(ASSETS_PATH, "NotoSansTC-Black.ttf")

    # Mediapipe
    mp_pose = mp.solutions.pose
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose()

    # 尺寸
    cam_x, cam_y, cam_w, cam_h = 277, 278, 365, 560
    yes_x, yes_y, yes_w, yes_h = 350, 424, 205, 205
    btn_x, btn_y, btn_w, btn_h = 889, 814, 129, 46

    # 狀態
    success_once = False
    hold_start_time = None
    camera_switch_button_rect = None
    fullscreen_button_rect = None

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
        ret, frame = camera_manager.get_frame()
        if not ret: break

        frame = cv2.flip(frame, 1)
        bg = cv2.resize(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # 攝影機畫面處理 - 使用與關卡相同的比例處理
        orig_h, orig_w = frame.shape[:2]
        scale_factor = max(cam_w / orig_w, cam_h / orig_h)
        resized_w = int(orig_w * scale_factor)
        resized_h = int(orig_h * scale_factor)
        resized_frame = cv2.resize(frame, (resized_w, resized_h))
        x_start = (resized_w - cam_w) // 2
        y_start = (resized_h - cam_h) // 2
        frame_resized = resized_frame[y_start:y_start + cam_h, x_start:x_start + cam_w]
        
        # 姿勢識別處理
        results = pose.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        # 骨架可視化
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame_resized,
        #         results.pose_landmarks,
        #         mp_pose.POSE_CONNECTIONS,
        #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #     )

        # 圓角遮罩處理
        mask = create_rounded_mask(cam_w, cam_h, radius=20)
        cropped_bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
        cropped_bgra[:, :, 3] = mask
        
        # 疊加到背景
        roi = bg[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
        alpha_s = cropped_bgra[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            roi[:, :, c] = (alpha_s * cropped_bgra[:, :, c] + alpha_l * roi[:, :, c])
        bg[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = roi

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

        # 繪製攝像頭切換按鈕
        bg, camera_switch_button_rect = camera_manager.draw_switch_button(bg, cam_x, cam_y, cam_w, cam_h)
        
        # 繪製全屏按鈕
        bg, fullscreen_button_rect = fullscreen_manager.draw_fullscreen_button(bg)

        canvas_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        game_surface = pygame.image.frombuffer(canvas_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
        
        # 使用全屏管理器渲染畫面
        fullscreen_manager.render_frame(game_surface)
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                # 處理全屏快捷鍵
                new_screen = fullscreen_manager.handle_keydown(event)
                if new_screen:
                    screen = new_screen
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                # 將螢幕座標轉換為遊戲座標
                game_x, game_y = fullscreen_manager.map_click_position((mx, my))
                
                # 檢查全屏按鈕點擊
                if fullscreen_manager.is_button_clicked((game_x, game_y), fullscreen_button_rect):
                    new_screen = fullscreen_manager.toggle_fullscreen()
                    if new_screen:
                        screen = new_screen
                    continue
                
                # 檢查攝像頭切換按鈕點擊
                if camera_manager.is_button_clicked((game_x, game_y), camera_switch_button_rect):
                    camera_manager.switch_camera()
                    continue
                
                if success_once:
                    if btn_x <= game_x <= btn_x + btn_w and btn_y <= game_y <= btn_y + btn_h:
                        return "start_game"

# --- 遊戲關卡邏輯 ---
def run_level(screen, camera_manager, fullscreen_manager, level_name, selected_sound_pack):
    """執行單一關卡的遊戲主迴圈"""
    
    # 宣告使用全域變數 
    current_level_index = LEVEL_ORDER.index(level_name)

    # 載入當前關卡配置
    config = load_config(level_name)
    # level_name = LEVEL_ORDER[current_level_index] # 重新取得 level_name

    # 初始化時間戳記和遊戲狀態變數
    music_start_time = None # 記錄音樂開始播放的系統時間，用於 UI 倒數計時顯示
    last_sound_time = time.time() # 控制所有動作音效的冷卻時間
    
    last_highlight_switch_music_ms = 0 # 上一次動作提示切換時，音樂的毫秒進度
    last_motion_switch_music_ms = 0   # 上一次動作組合切換時，音樂的毫秒進度

    # 初始化高亮索引
    highlight_index = 0  # 初始設定為第一個高亮位置
    next_highlight_index = 1 # 初始設定為下一個高亮位置 
    
    # 載入預先處理好的節拍時間 (秒)
    full_beat_times_s = config.get("beat_times", [])
    beat_times_s = [time_val for i, time_val in enumerate(full_beat_times_s) if i % 2 != 0]
 
    beat_index = 0 # 當前目標節拍的索引

    # 初始化下一個高亮音樂時間點
    # 如果有節拍數據，則將第一個節拍作為初始高亮時間點
    # 否則，使用一個預設值，例如 0
    next_highlight_music_ms = 0
    if beat_times_s:
        next_highlight_music_ms = beat_times_s[beat_index] * 1000

    # --- 建立音效對照表 ---
    sound_packs = {
        "percussion": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Percussion/1_Cajon_Tremolo.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Percussion/2_HandclapL_1.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Percussion/3_HandclapR_1.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Percussion/4_Cajon_Side.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Percussion/5_Cajon_Hit.wav",
        },
        "daily": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Quotidien/1_ScrewsInGlass.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Quotidien/2_WaterBottleL.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Quotidien/3_WaterBottleR.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Quotidien/4_PropaneTank.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Quotidien/5_WaterBasin.mp3",
        },
        "animal": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Animal/1_BirdsE.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Animal/2_BirdsB_L.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Animal/3_BirdsB_R.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Animal/4_BirdsD.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Animal/5_BirdsA.wav",
        }
    }
    current_sound_pack_files = sound_packs.get(selected_sound_pack, sound_packs["percussion"])

    # 載入資源
    bg_path = os.path.join(ASSETS_PATH, config["background"])
    music_path = os.path.join(ASSETS_PATH, config["music"])
    choose_frame_path = os.path.join(ASSETS_PATH, config["choose_frame"])
    font_path = os.path.join(ASSETS_PATH, "NotoSansTC-Black.ttf")

    bg_image = cv2.imread(bg_path)
    if bg_image is None: raise FileNotFoundError(f"❌ 背景圖片讀取失敗: {bg_path}")
    bg_image_resized = cv2.resize(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    cam_x, cam_y, cam_w, cam_h = 66, 220, 405, 610

    # 載入背景音樂
    pygame.mixer.music.load(music_path)

    # 初始化 Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # mp_drawing = mp.solutions.drawing_utils # 如果不需要顯示骨架可以註釋掉
    # mp_drawing_styles = mp.solutions.drawing_styles # 如果不需要顯示骨架可以註釋掉

    # 載入所有動作所需的音效和圖片
    action_resources = {}
    for action_name, action_config in config["actions"].items():
        print('action_config', action_config)
        base_sound_file = action_config.get("sound")
        image_file = action_config.get("image")
        word_image_file = action_config.get("word_image")
        print('base_sound_file', base_sound_file,)
        final_sound_path = None
        if base_sound_file:
            # 從對照表中查找對應的音效路徑
            final_sound_path = current_sound_pack_files.get(base_sound_file)
            print('final_sound_path', final_sound_path)
            if final_sound_path: # 確保有找到對應的檔案名
                # full_path = os.path.join(ASSETS_PATH, final_sound_path) # 組合完整路徑
                full_path = final_sound_path
                if not os.path.exists(full_path):
                    print(f"警告: 音效檔案不存在: {full_path}")
                    final_sound_path = None # 如果檔案不存在，則設為 None
                else:
                    final_sound_path = full_path # 使用完整路徑
            else:
                print(f"警告: 在音效包 '{selected_sound_pack}' 中找不到 '{base_sound_file}' 的對應音效。")


        action_resources[action_name] = {
            "sound": pygame.mixer.Sound(final_sound_path) if final_sound_path else None,
            "image": cv2.imread(os.path.join(ASSETS_PATH, image_file), cv2.IMREAD_UNCHANGED) if image_file else None,
            "word_image": cv2.imread(os.path.join(ASSETS_PATH, word_image_file), cv2.IMREAD_UNCHANGED) if word_image_file else None,
            "positions": action_config.get("positions", []) # 雖然這裡沒用到，但保留以防萬一
        }
    
    # 載入通用圖片和按鈕
    img_right = cv2.imread(os.path.join(ASSETS_PATH, "02.png"), cv2.IMREAD_UNCHANGED)
    img_left = cv2.imread(os.path.join(ASSETS_PATH, "03.png"), cv2.IMREAD_UNCHANGED)
    img_head = cv2.imread(os.path.join(ASSETS_PATH, "04.png"), cv2.IMREAD_UNCHANGED)
    img_open = cv2.imread(os.path.join(ASSETS_PATH, "05.png"), cv2.IMREAD_UNCHANGED)
    word_right = cv2.imread(os.path.join(ASSETS_PATH, "word2.png"), cv2.IMREAD_UNCHANGED)
    word_left = cv2.imread(os.path.join(ASSETS_PATH, "word3.png"), cv2.IMREAD_UNCHANGED)
    word_head = cv2.imread(os.path.join(ASSETS_PATH, "word4.png"), cv2.IMREAD_UNCHANGED)
    word_open = cv2.imread(os.path.join(ASSETS_PATH, "word5.png"), cv2.IMREAD_UNCHANGED)
    choose_bg = cv2.imread(choose_frame_path, cv2.IMREAD_UNCHANGED) # 高亮框
    
    finish_popup = cv2.imread(os.path.join(ASSETS_PATH, "finish.png"), cv2.IMREAD_UNCHANGED)
    btn1_img = cv2.imread(os.path.join(ASSETS_PATH, "light1.png"), cv2.IMREAD_UNCHANGED)
    btn2_img = cv2.imread(os.path.join(ASSETS_PATH, "light2.png"), cv2.IMREAD_UNCHANGED)
    btn3_img = cv2.imread(os.path.join(ASSETS_PATH, "light3.png"), cv2.IMREAD_UNCHANGED)
    btn4_img = cv2.imread(os.path.join(ASSETS_PATH, "light4.png"), cv2.IMREAD_UNCHANGED)
    
    # 優化：只載入一次並處理去飽和度
    end_btn_original = cv2.imread(os.path.join(ASSETS_PATH, "end.png"), cv2.IMREAD_UNCHANGED)
    next_btn_original = cv2.imread(os.path.join(ASSETS_PATH, "next.png"), cv2.IMREAD_UNCHANGED)
    next2_btn = cv2.imread(os.path.join(ASSETS_PATH, "next2.png"), cv2.IMREAD_UNCHANGED) # 結束彈窗上的「下一關」
    end_btn_disabled = desaturate_image(end_btn_original.copy())
    next_btn_disabled = desaturate_image(next_btn_original.copy())
    
    # 各類圖片和按鈕的座標及尺寸 (部分用於 level_1, level_2 的通用動作圖片)
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
    next2_btn_x, next2_btn_y, next2_btn_w, next2_btn_h = 610, 565, 210, 50 # 結束彈窗的按鈕位置
    
    # 來自 config.json 的按鈕位置
    btn_positions = config["button_positions"]
    btn1_x, btn1_y, btn1_w, btn1_h = btn_positions["btn1"]["x"], btn_positions["btn1"]["y"], btn_positions["btn1"]["w"], btn_positions["btn1"]["h"]
    btn2_x, btn2_y, btn2_w, btn2_h = btn_positions["btn2"]["x"], btn_positions["btn2"]["y"], btn_positions["btn2"]["w"], btn_positions["btn2"]["h"]
    btn3_x, btn3_y, btn3_w, btn3_h = btn_positions["btn3"]["x"], btn_positions["btn3"]["y"], btn_positions["btn3"]["w"], btn_positions["btn3"]["h"]
    btn4_x, btn4_y, btn4_w, btn4_h = btn_positions["btn4"]["x"], btn_positions["btn4"]["y"], btn_positions["btn4"]["w"], btn_positions["btn4"]["h"]

    clock = pygame.time.Clock()

    # 動作狀態追蹤
    prev_right, prev_left, prev_head, prev_open = False, False, False, False

    # 燈光透明度 (預設較暗)
    lamp1_alpha, lamp2_alpha, lamp3_alpha, lamp4_alpha = 0.1, 0.1, 0.1, 0.1
    
    # 遊戲開始前倒數計時狀態
    countdown_started, countdown_completed = False, False
    hold_start_time, post_countdown_start_time = None, None

    # 遊戲結束狀態
    show_finish_popup, finish_triggered = False, False

    # 遊戲分數和連擊
    score_total, combo_count = 0, 0
    
    # 關卡配置的時間參數
    countdown_time = config["countdown_time"] # 遊戲總時長
    motion_switch_interval = config["motion_switch_interval"] # 動作組合切換間隔 (秒)
    action_window = config["action_window"] # 動作判定寬容時間 (秒)
    # highlight_interval = 0.75 if level_name == "level_3" else 1.5 # 動作提示高亮間隔 (秒) - 現在由節拍控制
    motion_combinations = config["motion_combinations"] # 動作組合列表
    current_motion_index = 0 # 當前動作組合在列表中的索引

    # 新增一個音效冷卻時間，避免連續播放過快 (秒)
    sound_cooldown = 0.2 

    # 疊圖函式，帶有透明度
    def overlay_image_with_alpha(background, overlay, x, y, w, h, alpha):
        if overlay is None: return background
        overlay_resized = cv2.resize(overlay, (w, h))
        if len(overlay_resized.shape) > 2 and overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0 * alpha # 將整體透明度也考慮進去
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])
        return background
        
    # 創建圓角遮罩
    def create_rounded_mask(width, height, radius):
        mask = np.zeros((height, width), dtype=np.uint8)
        submask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(submask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
        return np.array(submask)

    # 獲取動作圖片位置
    def get_image_position(action_name):
        positions = config.get("image_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None

    # 獲取動作文字圖片位置
    def get_word_position(action_name):
        positions = config.get("word_positions", {})
        if action_name in positions:
            pos = positions[action_name]
            return pos["x"], pos["y"], pos["w"], pos["h"]
        return None

    # 判斷全身是否可見 (同校正畫面邏輯)
    def is_full_body_visible(landmarks):
        visible_parts = [0, 11, 12, 15, 16, 23, 24, 27, 28] # 關鍵點索引
        return all(landmarks[i].visibility > 0.2 for i in visible_parts) and \
               0.0 <= min(l.y for l in landmarks) and max(l.y for l in landmarks) <= 1.1

    # 判斷雙腿是否併攏 (同校正畫面邏輯)
    def are_legs_together(landmarks):
        return abs(landmarks[27].x - landmarks[28].x) < 0.3 # 稍微放寬了標準

    # 動作偵測邏輯
    def detect_pose_action(landmarks):
        l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        l_wrist = np.array([landmarks[15].x, landmarks[15].y])
        r_wrist = np.array([landmarks[16].x, landmarks[16].y])
        l_ear = np.array([landmarks[7].x, landmarks[7].y])
        r_ear = np.array([landmarks[8].x, landmarks[8].y])
        
        actions = {"open": False, "head": False, "left": False, "right": False}
        
        # 展開雙手 (open)
        # 腕部Y座標與肩部Y座標接近，左右腕部Y座標接近，左右腕部X座標距離大，且手腕不在耳朵上方
        actions["open"] = (abs(l_wrist[1] - r_wrist[1]) < 0.1 and 
                           abs(l_wrist[1] - l_shoulder[1]) < 0.1 and 
                           abs(r_wrist[1] - r_shoulder[1]) < 0.1 and 
                           abs(l_wrist[0] - r_wrist[0]) > 0.7 and 
                           not (l_wrist[1] < l_ear[1] and r_wrist[1] < r_ear[1]))
        
        # 舉高雙手 (head)
        # 腕部Y座標高於肩部，且X座標與耳朵接近，且左右腕部Y座標接近
        actions["head"] = (l_wrist[1] < l_shoulder[1] and 
                           r_wrist[1] < r_shoulder[1] and 
                           abs(l_wrist[0] - l_ear[0]) < 0.3 and 
                           abs(r_wrist[0] - r_ear[0]) < 0.3 and 
                           abs(l_wrist[1] - r_wrist[1]) < 0.3)
        
        # 左手舉高右手放下 (left)
        actions["left"] = l_wrist[1] < l_shoulder[1] - 0.1 and r_wrist[1] > r_shoulder[1] + 0.1
        
        # 右手舉高左手放下 (right)
        actions["right"] = r_wrist[1] < r_shoulder[1] - 0.1 and l_wrist[1] > l_shoulder[1] + 0.1
        
        return actions

    # 初始化下一個高亮音樂時間點 (毫秒)
    # 這是基於 Librosa 節拍的時間點，而不是固定的 interval
    next_highlight_music_ms = 0
    if beat_times_s:
        next_highlight_music_ms = beat_times_s[beat_index] * 1000
    
    # 節拍偵測的容忍度 (毫秒)，允許在節拍點前後一點點時間觸發
    beat_tolerance_ms = 400 

    running = True
    try: # 添加 try-finally 確保資源釋放
        while running:
            ret, frame = camera_manager.get_frame()
            if not ret: break

            canvas = bg_image_resized.copy() # 每個循環都重新複製背景圖
            frame = cv2.flip(frame, 1) # 鏡像翻轉

            # 裁剪並調整攝影機畫面以適應顯示區域
            orig_h, orig_w = frame.shape[:2]
            scale_factor = max(cam_w / orig_w, cam_h / orig_h)
            resized_w = int(orig_w * scale_factor)
            resized_h = int(orig_h * scale_factor)
            resized_frame = cv2.resize(frame, (resized_w, resized_h))
            x_start = (resized_w - cam_w) // 2
            y_start = (resized_h - cam_h) // 2
            cropped_frame = resized_frame[y_start:y_start + cam_h, x_start:x_start + cam_w]
            
            # MediaPipe 處理
            result = pose.process(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            
            # 如果需要顯示骨架，取消註釋以下代碼
            # if result.pose_landmarks:
            #     mp_drawing.draw_landmarks(
            #         cropped_frame, 
            #         result.pose_landmarks, 
            #         mp_pose.POSE_CONNECTIONS, 
            #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            #     )

            # 動作偵測狀態
            right_detected, left_detected, head_detected, open_detected = False, False, False, False
            full_body_ready_now = False # full_body_ready_now 用於當前幀狀態

            # 判斷全身是否準備好 (用於遊戲開始前的倒數和提示)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                full_body_ready_now = is_full_body_visible(landmarks) and are_legs_together(landmarks)

                # 遊戲開始前倒數邏輯 (與校正畫面類似，但這裡針對遊戲開始)
                if full_body_ready_now and not countdown_started and not countdown_completed:
                    if hold_start_time is None: hold_start_time = time.time()
                    if time.time() - hold_start_time >= 3: # 持續 3 秒後開始遊戲倒數
                        countdown_started = True
                        countdown_start_time = time.time()
                else:
                    hold_start_time = None # 如果不滿足全身可見並雙腿併攏，則重置計時

                if countdown_started and not countdown_completed:
                    seconds_left = 3 - int(time.time() - countdown_start_time) # 遊戲開始前 3, 2, 1 倒數
                    if seconds_left <= 0:
                        countdown_completed = True
                        post_countdown_start_time = time.time() # 記錄倒數結束的系統時間
                        pygame.mixer.music.play() # **開始播放背景音樂**
                        music_start_time = time.time() # 記錄音樂開始的系統時間 (用於UI顯示)
                        # Librosa 相關的時間基準在音樂播放後才開始更新
                        # 這裡不需要重置 last_highlight_switch_music_ms 和 last_motion_switch_music_ms
                        # 因為它們會在後面的邏輯中根據節拍數據自動更新
                    else:
                        # 在攝影機畫面中顯示倒數數字
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
                
                # 如果倒數已完成且遊戲正式開始，才偵測動作
                if countdown_completed:
                    # 延遲 1.5 秒後才開始偵測動作，給予音樂和畫面一些同步時間
                    if time.time() - post_countdown_start_time >= 1.5:
                        detected_actions = detect_pose_action(landmarks)
                        open_detected = detected_actions["open"] # 直接賦值
                        left_detected = detected_actions["left"]
                        right_detected = detected_actions["right"]
                        head_detected = detected_actions["head"]
            else: # 如果沒有偵測到任何骨架，則身體不準備就緒
                full_body_ready_now = False

            # 遊戲計時結束判斷
            current_time = time.time() # 系統時間
            if countdown_completed and not finish_triggered and not pygame.mixer.music.get_busy():
                # 音樂停止播放且未觸發結束彈窗，則觸發遊戲結束
                finish_triggered = True
                show_finish_popup = True

            # --- 核心：基於音樂播放進度來切換動作提示 ---
            if countdown_completed and not show_finish_popup:
                # 取得當前音樂播放的毫秒數，這是所有時間同步的基準！
                current_music_ms = pygame.mixer.music.get_pos()
                # 處理 get_pos() 可能回傳負值的情況 (例如音樂剛停止)
                if current_music_ms < 0:
                    current_music_ms = 0

                # 判斷是否需要切換動作提示高亮 (基於 Librosa 節拍)
                if beat_index < len(beat_times_s) and \
                   current_music_ms >= beat_times_s[beat_index] * 1000 - beat_tolerance_ms: # 允許稍微提前觸發
                    
                    highlight_index = next_highlight_index
                    next_highlight_index = (highlight_index + 1) % 4 # 還是按順序切換高亮位置

                    # 更新到下一個節拍時間
                    beat_index += 1
                    if beat_index < len(beat_times_s):
                        next_highlight_music_ms = beat_times_s[beat_index] * 1000
                    else:
                        # 所有節拍已播放完畢，可以選擇停止遊戲或循環
                        # 如果音樂結束，這裡可以觸發 finish_triggered = True
                        pass 

                # 判斷是否需要切換動作組合 (如果 config 中有設定 motion_switch_interval)
                # 這裡仍然使用固定間隔，因為 motion_combinations 通常是固定的模式
                if current_music_ms - last_motion_switch_music_ms >= motion_switch_interval * 1000:
                    current_motion_index = (current_motion_index + 1) % len(motion_combinations)
                    last_motion_switch_music_ms = current_music_ms # **更新為當前音樂時間**
                    # 可以選擇在這裡重置 highlight_index 和相關時間戳，讓新組合從第一個動作開始
                    # highlight_index = 0
                    # next_highlight_index = 1
                    # last_highlight_switch_music_ms = current_music_ms


                # --- 動作判定與音效播放邏輯 (重要修改區塊) ---
                # 只有在全身準備好並且音效冷卻時間已過才允許播放音效
                if full_body_ready_now and (time.time() - last_sound_time > sound_cooldown):
                    
                    # 取得當前動作組合中所有可能的動作名稱
                    current_possible_actions = motion_combinations[current_motion_index]

                    # 檢查每個偵測到的動作是否是當前組合中的一個，並且是新偵測到的
                    played_sound = False
                    # 檢查 "open" 動作
                    if "open" in current_possible_actions and open_detected and not prev_open:
                        resource = action_resources.get("open")
                        # print('resource===', resource)
                        if resource and resource["sound"]: 
                            resource["sound"].play()
                            score_total += 10
                            combo_count += 1
                            played_sound = True
                    
                    # 如果已經播放了音效，則不再檢查其他動作，並更新冷卻時間
                    if played_sound:
                        last_sound_time = time.time()
                    else: # 只有在沒有播放音效的情況下才檢查下一個動作
                        # 檢查 "head" 動作 (處理 config 中可能的 head, head_1, head_2)
                        head_action_names = ["head", "head_1", "head_2"]
                        detected_head_action = None
                        for action_name in head_action_names:
                            if action_name in current_possible_actions:
                                detected_head_action = action_name
                                break
                        
                        if detected_head_action and head_detected and not prev_head:
                            resource = action_resources.get(detected_head_action) # 使用實際配置的 head 資源
                            if resource and resource["sound"]: 
                                resource["sound"].play()
                                score_total += 10
                                combo_count += 1
                                played_sound = True
                        
                        if played_sound:
                            last_sound_time = time.time()
                        else:
                            # 檢查 "left" 動作
                            if "left" in current_possible_actions and left_detected and not prev_left:
                                resource = action_resources.get("left")
                                if resource and resource["sound"]: 
                                    resource["sound"].play()
                                    score_total += 10
                                    combo_count += 1
                                    played_sound = True
                            
                            if played_sound:
                                last_sound_time = time.time()
                            else:
                                # 檢查 "right" 動作
                                if "right" in current_possible_actions and right_detected and not prev_right:
                                    resource = action_resources.get("right")
                                    if resource and resource["sound"]: 
                                        resource["sound"].play()
                                        score_total += 10
                                        combo_count += 1
                                        played_sound = True
                                        
                                # 如果播放了音效，更新冷卻時間
                                if played_sound:
                                    last_sound_time = time.time()


                # 更新 prev 狀態，用於下一個循環的「動作開始瞬間」判斷
                prev_open = open_detected
                prev_head = head_detected
                prev_left = left_detected
                prev_right = right_detected
                
            # --- 燈光效果控制 ---
            # 燈光效果仍然與高亮索引綁定，因為這是視覺提示
            lamp1_alpha, lamp2_alpha, lamp3_alpha, lamp4_alpha = 0.1, 0.1, 0.1, 0.1
            current_motion_to_highlight = motion_combinations[current_motion_index][highlight_index]

            # 這裡的邏輯需要根據你的 level_3 配置來調整
            if level_name == "level_3":
                # Level 3 使用 action_resources 中的特定圖片和文字
                # 並且可能每個位置對應的動作是固定的，或者直接從 config 中定義
                if current_motion_to_highlight == "open": lamp1_alpha = 1.0 # 只要是高亮，就亮燈
                elif current_motion_to_highlight in ("head", "head_1"): lamp2_alpha = 1.0
                elif current_motion_to_highlight == "head_2": lamp3_alpha = 1.0 # 假設 head_2 在第三個位置
                elif (current_motion_to_highlight == "left" or current_motion_to_highlight == "right"): lamp4_alpha = 1.0 # 假設左右動作在第四個位置
            else: # level_1, level_2 (使用通用圖片)
                # 根據高亮索引來判斷
                if highlight_index == 0: lamp1_alpha = 1.0
                elif highlight_index == 1: lamp2_alpha = 1.0
                elif highlight_index == 2: lamp3_alpha = 1.0
                elif highlight_index == 3: lamp4_alpha = 1.0

            # --- 繪圖邏輯 ---
            # 繪製攝影機畫面（帶圓角）
            mask = create_rounded_mask(cam_w, cam_h, radius=20)
            cropped_bgra = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2BGRA)
            cropped_bgra[:, :, 3] = mask # 將圓角遮罩應用於 alpha 通道
            roi = canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
            alpha_s = cropped_bgra[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                roi[:, :, c] = (alpha_s * cropped_bgra[:, :, c] + alpha_l * roi[:, :, c])
            
            # 繪製高亮框
            choose_x, choose_y = choose_positions[highlight_index]
            canvas = overlay_image(canvas, choose_bg, choose_x, choose_y, choose_width, choose_height)
            
            # 繪製當前動作組合的圖片和文字
            current_combination = motion_combinations[current_motion_index]
            if level_name == "level_3":
                # Level 3 根據配置中的圖片和文字位置繪製
                for motion in current_combination:
                    res = action_resources.get(motion)
                    if res:
                        img_pos = get_image_position(motion)
                        word_pos = get_word_position(motion)
                        if res["image"] is not None and img_pos: canvas = overlay_image(canvas, res["image"], *img_pos)
                        if res["word_image"] is not None and word_pos: canvas = overlay_image(canvas, res["word_image"], *word_pos)
            else:
                # level_1, level_2 使用通用的圖片和文字，根據索引和預設位置繪製
                for idx, motion in enumerate(current_combination):
                    if motion == "right": 
                        if idx == 2: # 假設在索引 2 的位置是 right
                            canvas = overlay_image(canvas, img_right, img_right_x, img_right_y, img_right_w, img_right_h)
                            canvas = overlay_image(canvas, word_right, img_word2_x, img_word2_y, img_word2_w, img_word2_h)
                    elif motion == "left": 
                        if idx == 3: # 假設在索引 3 的位置是 left
                            canvas = overlay_image(canvas, img_left, img_left_x, img_left_y, img_left_w, img_left_h)
                            canvas = overlay_image(canvas, word_left, img_word3_x, img_word3_y, img_word3_w, img_word3_h)
                    elif motion == "head":
                        if idx == 1: # 假設在索引 1 的位置是 head
                            canvas = overlay_image(canvas, img_head, img_head_x, img_head_y, img_head_w, img_head_h)
                            canvas = overlay_image(canvas, word_head, img_word4_x, img_word4_y, img_word4_w, img_word4_h)
                        elif idx == 3: # 如果有第二個 head (level_2), 假設在索引 3
                            canvas = overlay_image(canvas, img_head, img_head2_x, img_head2_y, img_head2_w, img_head2_h)
                            canvas = overlay_image(canvas, word_head, img_word4_2_x, img_word4_2_y, img_word4_2_w, img_word4_2_h)
                    elif motion == "open":
                        if idx == 0: # 假設在索引 0 的位置是 open
                            canvas = overlay_image(canvas, img_open, img_open_x, img_open_y, img_open_w, img_open_h)
                            canvas = overlay_image(canvas, word_open, img_word5_x, img_word5_y, img_word5_w, img_word5_h)
                        elif idx == 2: # 如果有第二個 open (level_2), 假設在索引 2
                            canvas = overlay_image(canvas, img_open, img_open2_x, img_open2_y, img_open2_w, img_open2_h)
                            canvas = overlay_image(canvas, word_open, img_word5_2_x, img_word5_2_y, img_word5_2_w, img_word5_2_h)
            
            # 繪製燈光效果 (根據 lamp_alpha)
            canvas = overlay_image_with_alpha(canvas, btn1_img, btn1_x, btn1_y, btn1_w, btn1_h, lamp1_alpha)
            canvas = overlay_image_with_alpha(canvas, btn2_img, btn2_x, btn2_y, btn2_w, btn2_h, lamp2_alpha)
            canvas = overlay_image_with_alpha(canvas, btn3_img, btn3_x, btn3_y, btn3_w, btn3_h, lamp3_alpha)
            canvas = overlay_image_with_alpha(canvas, btn4_img, btn4_x, btn4_y, btn4_w, btn4_h, lamp4_alpha)
            
            # 繪製關卡切換按鈕 (上/下關)
            prev_button_img = end_btn_disabled if current_level_index <= 0 else end_btn_original
            next_button_img = next_btn_disabled if current_level_index >= len(LEVEL_ORDER) - 1 else next_btn_original
            canvas = overlay_image(canvas, prev_button_img, end_btn_x, end_btn_y, end_btn_w, end_btn_h)
            canvas = overlay_image(canvas, next_button_img, next_btn_x, next_btn_y, next_btn_w, next_btn_h)
            
            # 將 OpenCV 圖片轉換為 PIL 圖片以便繪製文字
            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(font_path, 24)
                score_font = ImageFont.truetype(font_path, 48)
                combo_font = ImageFont.truetype(font_path, 36)
                timer_font = ImageFont.truetype(font_path, 28)
            except IOError: # 如果字體文件不存在，則載入預設字體
                font = score_font = combo_font = timer_font = ImageFont.load_default()
            
            # 如果全身未準備好，顯示提示文字
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
            
            # 繪製分數、連擊和剩餘時間
            draw.text((790, 210), f"{score_total:03}", font=score_font, fill=(1, 125, 244))
            draw.text((1332, 218), f"{combo_count:02}", font=combo_font, fill=(1, 125, 244))
            
            # 顯示剩餘時間 (基於音樂開始的系統時間)
            if music_start_time:
                remaining_time = max(0, countdown_time - int(time.time() - music_start_time))
                draw.text((1030, 225), f"00:{remaining_time:02}", font=timer_font, fill=(80, 113, 135))
            
            canvas = np.array(pil_img)

            # 顯示遊戲結束彈窗
            if show_finish_popup:
                canvas = overlay_image(canvas, finish_popup, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
                canvas = overlay_image(canvas, next2_btn, next2_btn_x, next2_btn_y, next2_btn_w, next2_btn_h)

            # 將最終畫面轉換為 Pygame Surface 並顯示
            pygame_surface = pygame.image.frombuffer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
            screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()

            clock.tick(30) # 控制幀率

            # --- 事件處理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    return "quit"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if show_finish_popup: # 處理結束彈窗的按鈕點擊
                        if next2_btn_x <= mx <= next2_btn_x + next2_btn_w and next2_btn_y <= my <= next2_btn_y + next2_btn_h:
                            pygame.mixer.music.stop()
                            running = False
                            return "next_level" if current_level_index < len(LEVEL_ORDER) - 1 else "quit" # 完成最後一關則回到首頁
                    else: # 處理遊戲中的上/下關按鈕
                        # 使用原始的按鈕圖片來判斷點擊區域，而不是去飽和的圖片
                        if end_btn_x <= mx <= end_btn_x + end_btn_w and end_btn_y <= my <= end_btn_y + end_btn_h and current_level_index > 0:
                            pygame.mixer.music.stop()
                            running = False
                            return "prev_level"
                        if next_btn_x <= mx <= next_btn_x + next_btn_w and next_btn_y <= my <= next_btn_y + next_btn_h and current_level_index < len(LEVEL_ORDER) - 1:
                            pygame.mixer.music.stop()
                            running = False
                            return "next_level"
    finally:
        if 'pose' in locals() and pose: # 確保 pose 物件存在才呼叫 close
            pose.close() # 確保釋放 MediaPipe 資源
        pygame.mixer.music.stop() # 確保離開時停止音樂

# --- 主程式進入點 ---
def main():
    """程式主進入點，管理 Pygame 視窗和遊戲流程"""
    pygame.init()
    pygame.mixer.init()
    
    # 初始化全屏管理器
    fullscreen_manager = FullscreenManager()
    screen = fullscreen_manager.initialize_screen()
    
    # 初始化攝像頭管理器
    camera_manager = CameraManager()
    if not camera_manager.initialize_camera():
        print("❌ 無法開啟攝影機")
        return

    # 遊戲狀態機
    game_state = "homepage"
    selected_sound_pack = "percussion" # 預設音效包
    
    while True:
        if game_state == "homepage":
            action = run_homepage(screen, fullscreen_manager)
            if action == "start":
                game_state = "sound_select"
            else: # quit
                break

        elif game_state == "sound_select":
            result = run_soundselect(screen, fullscreen_manager)
            if result and result.get("action") == "confirm":
                selected_sound_pack = result.get("pack", "percussion")
                print(f"音效包選擇: {selected_sound_pack}")
                game_state = "movements"
            elif result == "back":
                game_state = "homepage"
            else: # quit or back
                break
        
        elif game_state == "movements":
            action = run_movements(screen, fullscreen_manager)
            if action == "finish":
                game_state = "correction"
            elif action == "back":
                game_state = "sound_select"
            else: # quit
                break

        elif game_state == "correction":
            action = run_correction_screen(screen, camera_manager, fullscreen_manager)
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
                level_action = run_level(screen, camera_manager, fullscreen_manager, level_name, selected_sound_pack)
                
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
    camera_manager.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    # 確保資源路徑正確
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()