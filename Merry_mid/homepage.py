import os
import sys
import pygame

def run_homepage(screen):
    pygame.display.set_caption("首頁")

    GAME_W, GAME_H = 1440, 960
    
    # --- 資源載入 ---
    try:
        bg = pygame.image.load("assets/homepage/homepage-bg.png").convert()
        start_btn = pygame.image.load("assets/homepage/start-b.png").convert_alpha()
        ad_img = pygame.image.load("assets/homepage/ad.png").convert_alpha()
        diary_btn = pygame.image.load("assets/homepage/diary.png").convert_alpha()
        setting_btn = pygame.image.load("assets/homepage/setting-b.png").convert_alpha()
        sensor_btn = pygame.image.load("assets/homepage/sensor-b.png").convert_alpha()
        replace_btn = pygame.image.load("assets/homepage/replace-b.png").convert_alpha()
    except pygame.error as e:
        print(f"無法載入圖片資源: {e}")
        return "quit"

    # --- 座標與 Rect ---
    START_POS = (275, 230)
    AD_POS = (637, 95)
    DIARY_POS = (637, 426)
    SETTING_POS = (637, 605)
    SENSOR_POS = (913, 605)
    REPLACE_POS = (1203, 853)

    start_rect = start_btn.get_rect(topleft=START_POS)
    
    # 建立一個固定的 Surface 進行繪圖，再縮放到主螢幕上
    game_surface = pygame.Surface((GAME_W, GAME_H))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.MOUSEBUTTONDOWN:
                # 將點擊座標從螢幕座標轉換為遊戲內座標
                w, h = screen.get_size()
                scale = min(w / GAME_W, h / GAME_H)
                offset_x = (w - GAME_W * scale) / 2
                offset_y = (h - GAME_H * scale) / 2
                game_x = (event.pos[0] - offset_x) / scale
                game_y = (event.pos[1] - offset_y) / scale
                check_pos = (game_x, game_y)

                if start_rect.collidepoint(check_pos):
                    print("點擊『開始訓練』")
                    return "start"

        # --- 繪圖 ---
        game_surface.blit(bg, (0, 0))
        game_surface.blit(start_btn, START_POS)
        game_surface.blit(ad_img, AD_POS)
        game_surface.blit(diary_btn, DIARY_POS)
        game_surface.blit(setting_btn, SETTING_POS)
        game_surface.blit(sensor_btn, SENSOR_POS)
        game_surface.blit(replace_btn, REPLACE_POS)

        # --- 縮放與顯示 ---
        w, h = screen.get_size()
        scale = min(w / GAME_W, h / GAME_H)
        scaled_surface = pygame.transform.smoothscale(game_surface, (int(GAME_W * scale), int(GAME_H * scale)))
        offset_x = (w - scaled_surface.get_width()) // 2
        offset_y = (h - scaled_surface.get_height()) // 2

        screen.fill((0, 0, 0))
        screen.blit(scaled_surface, (offset_x, offset_y))
        pygame.display.flip()
