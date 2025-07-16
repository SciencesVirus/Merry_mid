import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('..')
from camera_manager import CameraManager
from fullscreen_manager import FullscreenManager

# 初始化
pygame.init()
screen = pygame.display.set_mode((1440, 960))
pygame.display.set_caption("校正畫面")
clock = pygame.time.Clock()

# 初始化相機和全螢幕管理器
camera_manager = CameraManager(screen)
fullscreen_manager = FullscreenManager(screen)

# 設置按鈕顏色為 RGB 格式（因為是在 PIL Image 上繪製）
camera_manager.button_bg_color = (255, 165, 0, 128)  # 淡橘色半透明 (RGB)
fullscreen_manager.button_bg_color = (255, 165, 0, 128)  # 淡橘色半透明 (RGB)

# 音效
ding_sound = pygame.mixer.Sound("Correction_2/ding.wav")
ding_played = False

# 載入圖像
bg_image = cv2.imread("Correction_2/Correction_back.png")
yes_img = cv2.imread("Correction_2/yes.png", cv2.IMREAD_UNCHANGED)
no_btn = cv2.imread("Correction_2/no.png", cv2.IMREAD_UNCHANGED)
next_btn = cv2.imread("Correction_2/next.png", cv2.IMREAD_UNCHANGED)

if bg_image is None:
    raise FileNotFoundError("❌ 背景 Correction back.png 未讀取")
if yes_img is None:
    raise FileNotFoundError("❌ 圖片 yes.png 未讀取")
if no_btn is None or next_btn is None:
    raise FileNotFoundError("❌ no.png 或 next.png 未讀取")

# 尺寸設定
cam_x, cam_y, cam_w, cam_h = 277, 278, 365, 560
yes_x, yes_y, yes_w, yes_h = 350, 424, 205, 205
btn_x, btn_y, btn_w, btn_h = 889, 814, 129, 46

# Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# 疊圖函式
def overlay_image(background, overlay, x, y, width, height):
    overlay_resized = cv2.resize(overlay, (width, height))
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+height, x:x+width, c] = \
                alpha * overlay_resized[:, :, c] + \
                (1 - alpha) * background[y:y+height, x:x+width, c]
    else:
        background[y:y+height, x:x+width] = overlay_resized
    return background

# 判斷是否全身入鏡
def is_full_body_visible(landmarks):
    # 基本部位：頭、肩、手腕、髖、腳踝
    visible_ids = [0, 11, 12, 15, 16, 23, 24, 27, 28]
    for i in visible_ids:
        if landmarks[i].visibility < 0.2:
            return False

    # 額外要求：至少一隻腳的腳尖可見（左腳31 或 右腳32）
    if landmarks[31].visibility < 0.2 and landmarks[32].visibility < 0.2:
        return False

    # 所有點的 Y 座標需在畫面範圍內
    y_coords = [landmarks[i].y for i in visible_ids]
    return all(0.0 <= y <= 1.1 for y in y_coords)

# 判斷雙腳是否併攏
def are_legs_together(landmarks):
    l_ankle_x = landmarks[27].x
    r_ankle_x = landmarks[28].x

    l_foot_visible = landmarks[31].visibility > 0.2
    r_foot_visible = landmarks[32].visibility > 0.2

    # 若腳尖可見，額外納入距離判斷
    if l_foot_visible and r_foot_visible:
        l_foot_x = landmarks[31].x
        r_foot_x = landmarks[32].x
        return abs(l_ankle_x - r_ankle_x) < 0.12 and abs(l_foot_x - r_foot_x) < 0.15
    else:
        return abs(l_ankle_x - r_ankle_x) < 0.12

# 狀態
success_once = False
running = True

hold_start_time = None  # 用來記錄開始符合條件的時間

while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    bg = cv2.resize(bg_image, (1440, 960))
    frame_resized = cv2.resize(frame, (cam_w, cam_h))
    results = pose.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

    # 骨架可視化
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_resized,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    # 疊上實拍畫面
    cropped_bgra = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
    bg[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cropped_bgra[:, :, :3]

    # 更新相機和全螢幕按鈕
    camera_manager.update()
    fullscreen_manager.update()

    # 判斷狀態
    show_text = True
    if results.pose_landmarks:
        full = is_full_body_visible(results.pose_landmarks.landmark)
        legs = are_legs_together(results.pose_landmarks.landmark)

        if full and legs:
            if hold_start_time is None:
                hold_start_time = time.time()  # 第一次達成，開始計時
            elif time.time() - hold_start_time >= 3.0 and not success_once:
                success_once = True
                ding_sound.play()
            show_text = False  # 隱藏提示語（不論有無達成3秒）
        else:
            hold_start_time = None  # 條件中斷就重新開始計時


    # 顯示提示語
    if show_text:
        pil_img = Image.fromarray(bg)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("NotoSansTC-Black.ttf", 28)
            countdown_font = ImageFont.truetype("NotoSansTC-Black.ttf", 120)
        except:
            font = ImageFont.load_default()
            countdown_font = ImageFont.load_default()

        lines = ["請保持全身在拍攝畫面以內", "並將雙腿盡量併攏"]
        spacing = 10
        total_height = 0
        widths = []
        heights = []

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            widths.append(w)
            heights.append(h)
            total_height += h + spacing
        total_height -= spacing  # 去掉最後一個間距

        start_y = cam_y + (cam_h - total_height) // 2

        for i, line in enumerate(lines):
            x = cam_x + (cam_w - widths[i]) // 2
            y = start_y + sum(heights[:i]) + spacing * i
            draw.text((x, y), line, fill=(0, 100, 255), font=font)

        bg = np.array(pil_img)

    # 顯示倒數計時（獨立處理）
    if hold_start_time and not success_once:
        elapsed = time.time() - hold_start_time
        seconds_left = 4 - int(elapsed + 1)  # 核心改這一行

        if 1 <= seconds_left <= 3:
            pil_img = Image.fromarray(bg)
            draw = ImageDraw.Draw(pil_img)

            text = str(seconds_left)
            text_size = draw.textbbox((0, 0), text, font=countdown_font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]

            x = cam_x + (cam_w - text_width) // 2
            y = cam_y + (cam_h - text_height) // 3

            draw.text((x, y), text, fill=(255, 255, 255), font=countdown_font)
            bg = np.array(pil_img)

    # 成功後顯示 yes.png
    if success_once:
        bg = overlay_image(bg, yes_img, yes_x, yes_y, yes_w, yes_h)
        bg = overlay_image(bg, next_btn, btn_x, btn_y, btn_w, btn_h)
    else:
        bg = overlay_image(bg, no_btn, btn_x, btn_y, btn_w, btn_h)

    # 顯示畫面
    canvas_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pygame_surface = pygame.image.frombuffer(canvas_rgb.tobytes(), (1440, 960), "RGB")
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)

    # 事件處理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 處理相機和全螢幕按鈕點擊
            camera_manager.handle_click(event)
            fullscreen_manager.handle_click(event)
            
            if success_once:
                mx, my = pygame.mouse.get_pos()
                if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                    print("✅ 點擊 next，關閉畫面")
                    running = False

# 結尾
cap.release()
pygame.quit()
cv2.destroyAllWindows()