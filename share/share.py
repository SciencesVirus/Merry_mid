import pygame
import cv2
import numpy as np

# 初始化
pygame.init()
screen = pygame.display.set_mode((1440, 960))
pygame.display.set_caption("結束分享畫面")
clock = pygame.time.Clock()

# 載入圖像
bg_image = cv2.imread("share back.png")
group_img = cv2.imread("Group.png", cv2.IMREAD_UNCHANGED)
end_btn = cv2.imread("end.png", cv2.IMREAD_UNCHANGED)

# 圖片檢查
if bg_image is None:
    raise FileNotFoundError("❌ 背景圖 share back.png 未找到")
if group_img is None:
    raise FileNotFoundError("❌ Group.png 未找到")
if end_btn is None:
    raise FileNotFoundError("❌ end.png 未找到")

# 位置與大小設定
group_x, group_y, group_w, group_h = 264, 164, 912, 631
btn_x, btn_y, btn_w, btn_h = 614, 842, 210, 55

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

# 主迴圈
running = True
while running:
    bg = cv2.resize(bg_image, (1440, 960))
    bg = overlay_image(bg, group_img, group_x, group_y, group_w, group_h)
    bg = overlay_image(bg, end_btn, btn_x, btn_y, btn_w, btn_h)

    # 顯示畫面
    canvas_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pygame_surface = pygame.image.frombuffer(canvas_rgb.tobytes(), (1440, 960), "RGB")
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)

    # 處理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                print("🔚 點擊結束分享，退出畫面")
                running = False

# 結束
pygame.quit()
cv2.destroyAllWindows()