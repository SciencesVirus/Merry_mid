import pygame
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

class FullscreenManager:
    def __init__(self, config_file="fullscreen_config.json"):
        self.config_file = config_file
        self.is_fullscreen = False
        self.original_size = (1440, 960)  # 原始遊戲大小
        self.current_screen = None
        self.fullscreen_icon = None
        self.button_bg_color = (255, 165, 0, 128)  # 預設淡橘色半透明 (RGB)
        try:
            self.fullscreen_icon = cv2.imread("assets/icon/fullscreen.png", cv2.IMREAD_UNCHANGED)
            if self.fullscreen_icon is None:
                print("❌ 無法載入全螢幕圖示")
        except Exception as e:
            print(f"❌ 載入全螢幕圖示失敗: {e}")
        self.load_config()
    
    def load_config(self):
        """載入全屏配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.is_fullscreen = config.get('is_fullscreen', False)
            else:
                self.is_fullscreen = False
        except Exception as e:
            print(f"載入全屏配置失敗: {e}")
            self.is_fullscreen = False
    
    def save_config(self):
        """保存全屏配置"""
        try:
            config = {
                'is_fullscreen': self.is_fullscreen
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存全屏配置失敗: {e}")
    
    def initialize_screen(self):
        """初始化螢幕，根據配置設置全屏或視窗模式"""
        if self.is_fullscreen:
            self.current_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.current_screen = pygame.display.set_mode(self.original_size)
        return self.current_screen
    
    def toggle_fullscreen(self):
        """切換全屏狀態"""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            self.current_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            print("切換到全屏模式")
        else:
            self.current_screen = pygame.display.set_mode(self.original_size)
            print("切換到視窗模式")
        
        self.save_config()
        return self.current_screen
    
    def get_scaled_surface(self, game_surface):
        """獲取按比例縮放的遊戲畫面"""
        if not self.current_screen:
            return game_surface
        
        screen_w, screen_h = self.current_screen.get_size()
        game_w, game_h = self.original_size
        
        # 計算縮放比例，保持寬高比
        scale_x = screen_w / game_w
        scale_y = screen_h / game_h
        scale = min(scale_x, scale_y)
        
        # 計算縮放後的大小
        scaled_w = int(game_w * scale)
        scaled_h = int(game_h * scale)
        
        # 縮放遊戲畫面
        scaled_surface = pygame.transform.smoothscale(game_surface, (scaled_w, scaled_h))
        
        return scaled_surface, scale
    
    def get_display_offset(self):
        """獲取畫面在螢幕上的偏移量（居中顯示）"""
        if not self.current_screen:
            return (0, 0)
        
        screen_w, screen_h = self.current_screen.get_size()
        game_w, game_h = self.original_size
        
        # 計算縮放比例
        scale_x = screen_w / game_w
        scale_y = screen_h / game_h
        scale = min(scale_x, scale_y)
        
        # 計算縮放後的大小
        scaled_w = int(game_w * scale)
        scaled_h = int(game_h * scale)
        
        # 計算居中偏移
        offset_x = (screen_w - scaled_w) // 2
        offset_y = (screen_h - scaled_h) // 2
        
        return (offset_x, offset_y)
    
    def map_click_position(self, screen_pos):
        """將螢幕點擊座標轉換為遊戲內座標"""
        if not self.current_screen:
            return screen_pos
        
        screen_w, screen_h = self.current_screen.get_size()
        game_w, game_h = self.original_size
        
        # 計算縮放比例
        scale_x = screen_w / game_w
        scale_y = screen_h / game_h
        scale = min(scale_x, scale_y)
        
        # 計算偏移
        offset_x, offset_y = self.get_display_offset()
        
        # 轉換座標
        game_x = (screen_pos[0] - offset_x) / scale
        game_y = (screen_pos[1] - offset_y) / scale
        
        return (game_x, game_y)
    
    def draw_fullscreen_button(self, canvas):
        """繪製全屏切換按鈕在右上角"""
        game_w, game_h = self.original_size
        
        # 按鈕設置
        btn_size = 40
        btn_margin = 15
        btn_x = game_w - btn_size - btn_margin
        btn_y = btn_margin
        
        # 轉換為PIL圖像進行繪圖
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        
        # 繪製半透明淡橘色的圓形按鈕背景
        draw.ellipse([btn_x, btn_y, btn_x + btn_size, btn_y + btn_size], 
                    fill=self.button_bg_color)  # 使用動態顏色設置
        
        # 如果有載入全螢幕圖示，使用它；否則使用備用圖示
        if self.fullscreen_icon is not None:
            # 將全螢幕圖示調整為合適大小
            icon_size = int(btn_size * 0.6)  # 圖示大小為按鈕的 60%
            icon_x = btn_x + (btn_size - icon_size) // 2
            icon_y = btn_y + (btn_size - icon_size) // 2
            
            # 將 canvas 轉回 numpy array 以使用 overlay_image
            canvas = np.array(pil_img)
            icon_resized = cv2.resize(self.fullscreen_icon, (icon_size, icon_size))
            
            # 疊加圖示
            if icon_resized.shape[2] == 4:  # 如果有 alpha 通道
                alpha = icon_resized[:, :, 3] / 255.0
                for c in range(3):
                    canvas[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                        alpha * icon_resized[:, :, c] + \
                        (1 - alpha) * canvas[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
            else:
                canvas[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized[:, :, :3]
            
            # 轉回 PIL 以繪製文字
            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)
        else:
            # 備用圖示：根據全螢幕狀態繪製不同的圖示
            icon_padding = 8
            icon_x1 = btn_x + icon_padding
            icon_y1 = btn_y + icon_padding
            icon_x2 = btn_x + btn_size - icon_padding
            icon_y2 = btn_y + btn_size - icon_padding
            
            if self.is_fullscreen:
                # 全屏狀態：繪製退出全屏圖標（兩個小方框）
                small_size = 8
                draw.rectangle([icon_x1, icon_y1, icon_x1 + small_size, icon_y1 + small_size], 
                              outline=(255, 255, 255), width=2)
                draw.rectangle([icon_x2 - small_size, icon_y2 - small_size, icon_x2, icon_y2], 
                              outline=(255, 255, 255), width=2)
            else:
                # 視窗狀態：繪製全屏圖標（大方框）
                draw.rectangle([icon_x1, icon_y1, icon_x2, icon_y2], 
                              outline=(255, 255, 255), width=2)
        
        # 繪製狀態文字
        try:
            font = ImageFont.truetype("level_Universal/NotoSansTC-Black.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        text = "ESC" if self.is_fullscreen else "F11"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = btn_x + (btn_size - text_w) // 2
        text_y = btn_y + btn_size + 2
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        canvas = np.array(pil_img)
        
        # 返回按鈕區域座標
        return canvas, (btn_x, btn_y, btn_size, btn_size)
    
    def is_button_clicked(self, click_pos, button_rect):
        """檢查是否點擊了全屏按鈕"""
        if button_rect is None:
            return False
        
        btn_x, btn_y, btn_size, _ = button_rect
        click_x, click_y = click_pos
        
        # 檢查點擊是否在圓形按鈕內
        center_x = btn_x + btn_size // 2
        center_y = btn_y + btn_size // 2
        distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
        
        return distance <= btn_size // 2
    
    def render_frame(self, game_surface):
        """渲染並顯示遊戲畫面"""
        if not self.current_screen:
            return
        
        # 清空螢幕
        self.current_screen.fill((0, 0, 0))
        
        # 獲取縮放後的畫面
        scaled_surface, scale = self.get_scaled_surface(game_surface)
        
        # 獲取居中偏移
        offset_x, offset_y = self.get_display_offset()
        
        # 顯示畫面
        self.current_screen.blit(scaled_surface, (offset_x, offset_y))
        pygame.display.flip()
    
    def handle_keydown(self, event):
        """處理鍵盤事件"""
        if event.key == pygame.K_F11:
            # F11 切換全屏
            return self.toggle_fullscreen()
        elif event.key == pygame.K_ESCAPE and self.is_fullscreen:
            # ESC 退出全屏
            self.is_fullscreen = False
            self.current_screen = pygame.display.set_mode(self.original_size)
            self.save_config()
            print("退出全屏模式")
            return self.current_screen
        return None 