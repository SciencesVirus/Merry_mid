import cv2
import json
import os
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class CameraManager:
    def __init__(self, config_file="camera_config.json"):
        self.config_file = config_file
        self.current_camera_index = 0
        self.available_cameras = []
        self.cap = None
        self.camera_icon = None
        self.button_bg_color = (255, 165, 0, 128)  # 預設淡橘色半透明 (RGB)
        try:
            self.camera_icon = cv2.imread("assets/icon/camera.png", cv2.IMREAD_UNCHANGED)
            if self.camera_icon is None:
                print("❌ 無法載入相機圖示")
        except Exception as e:
            print(f"❌ 載入相機圖示失敗: {e}")
        self.load_config()
        self.detect_cameras()
        self.initialize_camera()
    
    def detect_cameras(self):
        """偵測可用的攝像頭"""
        self.available_cameras = []
        # 測試攝像頭索引 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(i)
                cap.release()
        
        if not self.available_cameras:
            print("❌ 未找到可用的攝像頭")
            self.available_cameras = [0]  # 默認添加索引0
        else:
            print(f"✅ 找到攝像頭: {self.available_cameras}")
        
        # 確保當前選擇的攝像頭在可用列表中
        if self.current_camera_index not in self.available_cameras:
            self.current_camera_index = self.available_cameras[0]
    
    def load_config(self):
        """載入攝像頭配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.current_camera_index = config.get('camera_index', 0)
            else:
                self.current_camera_index = 0
        except Exception as e:
            print(f"載入攝像頭配置失敗: {e}")
            self.current_camera_index = 0
    
    def save_config(self):
        """保存攝像頭配置"""
        try:
            config = {
                'camera_index': self.current_camera_index
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存攝像頭配置失敗: {e}")
    
    def initialize_camera(self):
        """初始化攝像頭"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            print(f"❌ 無法開啟攝像頭 {self.current_camera_index}")
            # 嘗試使用第一個可用的攝像頭
            if self.available_cameras:
                self.current_camera_index = self.available_cameras[0]
                self.cap = cv2.VideoCapture(self.current_camera_index)
        
        return self.cap.isOpened()
    
    def switch_camera(self):
        """切換到下一個攝像頭"""
        if len(self.available_cameras) <= 1:
            return False
        
        current_index = self.available_cameras.index(self.current_camera_index)
        next_index = (current_index + 1) % len(self.available_cameras)
        self.current_camera_index = self.available_cameras[next_index]
        
        self.initialize_camera()
        self.save_config()
        print(f"切換到攝像頭 {self.current_camera_index}")
        return True
    
    def get_frame(self):
        """獲取攝像頭幀"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def release(self):
        """釋放攝像頭資源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def draw_switch_button(self, canvas, cam_x, cam_y, cam_w, cam_h):
        """在取景框內繪製攝像頭切換按鈕"""
        if len(self.available_cameras) <= 1:
            return canvas, None
        
        # 按鈕位置 (右上角)
        btn_size = 40
        btn_x = cam_x + cam_w - btn_size - 10
        btn_y = cam_y + 10
        
        # 創建按鈕區域
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        
        # 繪製圓形按鈕背景
        draw.ellipse([btn_x, btn_y, btn_x + btn_size, btn_y + btn_size], 
                    fill=self.button_bg_color)  # 使用動態顏色設置
        
        # 如果有載入相機圖示，使用它；否則使用備用圖示
        if self.camera_icon is not None:
            # 將相機圖示調整為合適大小
            icon_size = int(btn_size * 0.6)  # 圖示大小為按鈕的 60%
            icon_x = btn_x + (btn_size - icon_size) // 2
            icon_y = btn_y + (btn_size - icon_size) // 2
            
            # 將 canvas 轉回 numpy array 以使用 overlay_image
            canvas = np.array(pil_img)
            icon_resized = cv2.resize(self.camera_icon, (icon_size, icon_size))
            
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
            # 備用圖示：簡單的矩形
            icon_padding = 8
            icon_x1 = btn_x + icon_padding
            icon_y1 = btn_y + icon_padding
            icon_x2 = btn_x + btn_size - icon_padding
            icon_y2 = btn_y + btn_size - icon_padding
            draw.rectangle([icon_x1, icon_y1, icon_x2, icon_y2], 
                          outline=(255, 255, 255), width=2)
        
        # 繪製攝像頭編號
        try:
            font = ImageFont.truetype("level_Universal/NotoSansTC-Black.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text = str(self.current_camera_index)
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
        """檢查是否點擊了切換按鈕"""
        if button_rect is None:
            return False
        
        btn_x, btn_y, btn_size, _ = button_rect
        click_x, click_y = click_pos
        
        # 檢查點擊是否在圓形按鈕內
        center_x = btn_x + btn_size // 2
        center_y = btn_y + btn_size // 2
        distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
        
        return distance <= btn_size // 2 