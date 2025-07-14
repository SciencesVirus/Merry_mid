import os
import pygame
import sys

def run_soundselect(screen):
    pygame.display.set_caption("音效選擇")
    GAME_W, GAME_H = 1440, 960

    # --- 資源載入 ---
    try:
    # 載入圖片
        bg = pygame.image.load("assets/sound select/soundselect-bg.png").convert()
        btn_i_dark = pygame.image.load("assets/sound select/i-b.png").convert_alpha()
        btn_i_light = pygame.image.load("assets/sound select/i2-b.png").convert_alpha()
        btn_d_dark = pygame.image.load("assets/sound select/d1-b.png").convert_alpha()
        btn_d_light = pygame.image.load("assets/sound select/d2-b.png").convert_alpha()
        btn_a_dark = pygame.image.load("assets/sound select/a1-b.png").convert_alpha()
        btn_a_light = pygame.image.load("assets/sound select/a2-b.png").convert_alpha()
        # btn_r1 = pygame.image.load("assets/sound select/p1-b.png").convert_alpha()
        # btn_r2 = pygame.image.load("assets/sound select/p2-b.png").convert_alpha()
        # btn_r3 = pygame.image.load("assets/sound select/p3-b.png").convert_alpha()
        back_btn = pygame.image.load("assets/sound select/back-b.png").convert_alpha()
        check_btn = pygame.image.load("assets/sound select/check-b.png").convert_alpha()

        # 載入音效
        percussion_sounds = [pygame.mixer.Sound(f"Sound effects/Percussion/{f}") for f in ["5_Cajon_Hit.wav", "4_Cajon_Side.wav", "3_HandclapR_1.wav", "2_HandclapL_1.wav", "1_Cajon_Tremolo.wav"]]
        daily_sounds = [pygame.mixer.Sound(f"Sound effects/Quotidien/{f}") for f in ["5_WaterBasin.mp3", "4_PropaneTank.wav", "3_WaterBottleR.wav", "2_WaterBottleL.wav", "1_ScrewsInGlass.wav"]]
        animal_sounds = [pygame.mixer.Sound(f"Sound effects/Animal/{f}") for f in ["5_BirdsA.wav", "4_BirdsD.wav", "3_BirdsB_R.wav", "2_BirdsB_L.wav", "1_BirdsE.wav"]]
    except pygame.error as e:
        print(f"無法載入資源: {e}")
        return "quit"

    # --- 座標與 Rect ---
    POS_I, POS_D, POS_A = (345, 375), (620, 375), (895, 375)
    R_I, R_D, R_A = (393, 598), (668, 598), (945, 598)
    POS_BACK, POS_CHECK = (106, 90), (613, 796)
    
    rect_i = btn_i_dark.get_rect(topleft=POS_I)
    rect_d = btn_d_dark.get_rect(topleft=POS_D)
    rect_a = btn_a_dark.get_rect(topleft=POS_A)
    # rect_ri = btn_r1.get_rect(topleft=R_I)
    # rect_rd = btn_r2.get_rect(topleft=R_D)
    # rect_ra = btn_r3.get_rect(topleft=R_A)
    rect_back = back_btn.get_rect(topleft=POS_BACK)
    rect_check = check_btn.get_rect(topleft=POS_CHECK)

    # --- 狀態變數 ---
    selected_pack = "percussion"
    sound_map = {"percussion": percussion_sounds, "daily": daily_sounds, "animal": animal_sounds}
    sound_indices = {"percussion": 0, "daily": 0, "animal": 0}
    
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

                if rect_i.collidepoint(click_pos): 
                    selected_pack = "percussion"
                    pack_to_play = "percussion"
                    sounds = sound_map[pack_to_play]
                    current_idx = sound_indices[pack_to_play]
                    sounds[current_idx].play()
                    sound_indices[pack_to_play] = (current_idx + 1) % len(sounds)
                elif rect_d.collidepoint(click_pos): 
                    selected_pack = "daily"
                    pack_to_play = "daily"
                    sounds = sound_map[pack_to_play]
                    current_idx = sound_indices[pack_to_play]
                    sounds[current_idx].play()
                    sound_indices[pack_to_play] = (current_idx + 1) % len(sounds)
                elif rect_a.collidepoint(click_pos): 
                    selected_pack = "animal"
                    pack_to_play = "animal"
                    sounds = sound_map[pack_to_play]
                    current_idx = sound_indices[pack_to_play]
                    sounds[current_idx].play()
                    sound_indices[pack_to_play] = (current_idx + 1) % len(sounds)
                # 點選圖示即可播放音效
                # elif rect_i.collidepoint(click_pos) or rect_d.collidepoint(click_pos) or rect_a.collidepoint(click_pos):
                #     pack_to_play = "percussion" if rect_i.collidepoint(click_pos) else \
                #                    "daily" if rect_d.collidepoint(click_pos) else "animal"
                #     sounds = sound_map[pack_to_play]
                #     current_idx = sound_indices[pack_to_play]
                #     sounds[current_idx].play()
                #     sound_indices[pack_to_play] = (current_idx + 1) % len(sounds)

                elif rect_back.collidepoint(click_pos):
                    return "back"
                elif rect_check.collidepoint(click_pos):
                    return {"action": "confirm", "pack": selected_pack}
        
        # --- 繪圖 ---
        game_surface.blit(bg, (0, 0))
        game_surface.blit(btn_i_light if selected_pack == "percussion" else btn_i_dark, POS_I)
        game_surface.blit(btn_d_light if selected_pack == "daily" else btn_d_dark, POS_D)
        game_surface.blit(btn_a_light if selected_pack == "animal" else btn_a_dark, POS_A)
        # game_surface.blit(btn_r1, R_I)
        # game_surface.blit(btn_r2, R_D)
        # game_surface.blit(btn_r3, R_A)
        game_surface.blit(back_btn, POS_BACK)
        game_surface.blit(check_btn, POS_CHECK)

        # --- 縮放與顯示 ---
        w, h = screen.get_size()
        scale = min(w / GAME_W, h / GAME_H)
        scaled_surface = pygame.transform.smoothscale(game_surface, (int(GAME_W * scale), int(GAME_H * scale)))
        offset_x, offset_y = (w - scaled_surface.get_width()) // 2, (h - scaled_surface.get_height()) // 2
        screen.fill((0, 0, 0))
        screen.blit(scaled_surface, (offset_x, offset_y))
        pygame.display.flip()
