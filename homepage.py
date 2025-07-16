import os
import sys
import pygame

def run_homepage(screen, fullscreen_manager):
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
    fullscreen_button_rect = None
    
    # 建立一個固定的 Surface 進行繪圖，再縮放到主螢幕上
    game_surface = pygame.Surface((GAME_W, GAME_H))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                # 處理全屏快捷鍵
                new_screen = fullscreen_manager.handle_keydown(event)
                if new_screen:
                    screen = new_screen

            if event.type == pygame.MOUSEBUTTONDOWN:
                # 使用全屏管理器轉換座標
                game_x, game_y = fullscreen_manager.map_click_position(event.pos)
                
                # 檢查全屏按鈕點擊
                if fullscreen_manager.is_button_clicked((game_x, game_y), fullscreen_button_rect):
                    new_screen = fullscreen_manager.toggle_fullscreen()
                    if new_screen:
                        screen = new_screen
                    continue

                if start_rect.collidepoint((game_x, game_y)):
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

        # 轉換為numpy數組以便繪製全屏按鈕
        import numpy as np
        canvas = pygame.surfarray.array3d(game_surface)
        canvas = np.transpose(canvas, (1, 0, 2))  # 調整維度順序
        
        # 繪製全屏按鈕
        canvas, fullscreen_button_rect = fullscreen_manager.draw_fullscreen_button(canvas)
        
        # 轉換回pygame surface
        canvas = np.transpose(canvas, (1, 0, 2))
        game_surface = pygame.surfarray.make_surface(canvas)

        # --- 使用全屏管理器渲染畫面 ---
        fullscreen_manager.render_frame(game_surface)
