import os
import sys
import pygame
from PIL import Image

def load_gif_frames(path):
    """把 GIF 拆幀並轉換為 Pygame Surface"""
    try:
        img = Image.open(path)
    except FileNotFoundError:
        print(f"錯誤: 找不到 GIF 檔案 {path}")
        return []
        
    frames = []
    try:
        while True:
            frame = img.convert('RGBA')
            py_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(py_image)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames

def run_movements(screen):
    pygame.display.set_caption("動作介紹")
    GAME_W, GAME_H = 1440, 960

    # --- 資源載入 ---
    try:
        bg = pygame.image.load("assets/movements/movements-bg.png").convert()
        back_btn = pygame.image.load("assets/movements/back-b.png").convert_alpha()
        finish_btn = pygame.image.load("assets/movements/finish-b.png").convert_alpha()
        left_btn = pygame.image.load("assets/movements/n-l-b.png").convert_alpha()
        right_btn = pygame.image.load("assets/movements/n-r-b.png").convert_alpha()
        card_texts = [pygame.image.load(f"assets/movements/t{i}.png").convert_alpha() for i in range(1, 7)]
        dot_on = pygame.image.load("assets/movements/dot-o.png").convert_alpha()
        dot_off = pygame.image.load("assets/movements/dot-g.png").convert_alpha()
        cards = [load_gif_frames(f"assets/gif/0{i}.GIF") for i in range(1, 7)]
        # 修正第五個 gif 的檔名
        cards[4] = load_gif_frames("assets/gif/05.gif")
    except pygame.error as e:
        print(f"無法載入資源: {e}")
        return "quit"

    # --- 座標與 Rect ---
    POS_BACK, POS_FINISH = (106, 90), (613, 796)
    POS_LEFT, POS_RIGHT = (290, 440), (1125, 440)
    POS_TEXT, POS_GIF = (420, 335), (715, 249)
    DOT_START_X, DOT_Y, DOT_SPACING = 624, 698, 34
    TARGET_W, TARGET_H = 319, 451

    rect_back = back_btn.get_rect(topleft=POS_BACK)
    rect_finish = finish_btn.get_rect(topleft=POS_FINISH)
    rect_left = left_btn.get_rect(topleft=POS_LEFT)
    rect_right = right_btn.get_rect(topleft=POS_RIGHT)

    # --- 狀態變數 ---
    current_idx = 0
    current_frame_idx = 0
    last_update = pygame.time.get_ticks()
    frame_duration = 100  # ms
    
    game_surface = pygame.Surface((GAME_W, GAME_H))

    def map_click(pos):
        w, h = screen.get_size()
        scale = min(w / GAME_W, h / GAME_H)
        offset_x = (w - GAME_W * scale) / 2
        offset_y = (h - GAME_H * scale) / 2
        return (pos[0] - offset_x) / scale, (pos[1] - offset_y) / scale

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = map_click(event.pos)
                if rect_left.collidepoint(click_pos):
                    current_idx = (current_idx - 1) % len(cards)
                    current_frame_idx = 0
                elif rect_right.collidepoint(click_pos):
                    current_idx = (current_idx + 1) % len(cards)
                    current_frame_idx = 0
                elif rect_back.collidepoint(click_pos):
                    return "back"
                elif rect_finish.collidepoint(click_pos):
                    return "finish"

        # --- 動畫幀更新 ---
        if pygame.time.get_ticks() - last_update > frame_duration:
            last_update = pygame.time.get_ticks()
            if cards[current_idx]:
                 current_frame_idx = (current_frame_idx + 1) % len(cards[current_idx])

        # --- 繪圖 ---
        game_surface.blit(bg, (0, 0))
        game_surface.blit(back_btn, POS_BACK)
        game_surface.blit(finish_btn, POS_FINISH)
        game_surface.blit(left_btn, POS_LEFT)
        game_surface.blit(right_btn, POS_RIGHT)
        game_surface.blit(card_texts[current_idx], POS_TEXT)

        # 繪製 GIF
        if cards[current_idx]:
            frame = cards[current_idx][current_frame_idx]
            original_w, original_h = frame.get_size()
            scale_ratio = min(TARGET_W / original_w, TARGET_H / original_h)
            new_w, new_h = int(original_w * scale_ratio), int(original_h * scale_ratio)
            frame_scaled = pygame.transform.smoothscale(frame, (new_w, new_h))
            game_surface.blit(frame_scaled, POS_GIF)

        # 繪製指示點
        for i in range(len(cards)):
            dot_x = DOT_START_X + i * DOT_SPACING
            dot_image = dot_on if i == current_idx else dot_off
            game_surface.blit(dot_image, (dot_x, DOT_Y))

        # --- 縮放與顯示 ---
        w, h = screen.get_size()
        scale = min(w / GAME_W, h / GAME_H)
        scaled_surface = pygame.transform.smoothscale(game_surface, (int(GAME_W * scale), int(GAME_H * scale)))
        offset_x, offset_y = (w - scaled_surface.get_width()) // 2, (h - scaled_surface.get_height()) // 2
        screen.fill((0, 0, 0))
        screen.blit(scaled_surface, (offset_x, offset_y))
        pygame.display.flip()
