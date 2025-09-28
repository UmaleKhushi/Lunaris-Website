import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from tkinter import Tk, filedialog
import os
import cv2

OUTPUT_DIR = "terrain_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PIXELS_PER_CM = 37

# ---------------- Stone Detection ----------------
def detect_stones(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # ignore small noise
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        stones.append((int(x), int(y), int(radius)))
    return stones

# ---------------- Main ----------------
def main():
    root = Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(title="Select Terrain Image",
                                          filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not img_path:
        print("No file selected.")
        return

    img_rgb = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_rgb.shape[:2]

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Smooth / terrain map
    terrain_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    terrain_smooth = gaussian_filter(terrain_gray, sigma=2)
    terrain_norm = (terrain_smooth - terrain_smooth.min()) / (terrain_smooth.max() - terrain_smooth.min())
    plt.imsave(os.path.join(OUTPUT_DIR, "terrain_map.png"), terrain_norm, cmap='terrain')
    print("Saved: terrain_map.png")

    # Slope / hazard map
    grad_x = sobel(terrain_norm, axis=1)
    grad_y = sobel(terrain_norm, axis=0)
    slope = np.hypot(grad_x, grad_y)
    hazard_threshold = np.percentile(slope, 90)
    hazard_map = np.zeros_like(slope)
    hazard_map[slope > hazard_threshold] = 1
    plt.imsave(os.path.join(OUTPUT_DIR, "hazard_map_2d.png"), hazard_map, cmap='Reds')
    print("Saved: hazard_map_2d.png")

    # Detect stones
    stones = detect_stones(img_rgb)

    # Grid map: Green=Safe, Yellow=Hazard, Red=Stones
    grid_img = np.zeros((h, w, 3), dtype=np.uint8)
    grid_img[...,1] = 255  # Green base
    grid_img[hazard_map==1] = [255,255,0]  # Yellow hazard
    for x, y, r in stones:
        cv2.circle(grid_img, (x, y), r, (255,0,0), -1)  # Red stone
    plt.imsave(os.path.join(OUTPUT_DIR, "grid_map_colored.png"), grid_img)
    print("Saved: grid_map_colored.png")

    # 3D terrain plot
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, terrain_norm, rstride=2, cstride=2,
                    facecolors=plt.cm.terrain(terrain_norm), edgecolor='none', alpha=0.9)
    # Overlay hazards
    hazard_overlay = np.zeros((h, w, 4))
    hazard_overlay[...,0] = hazard_map
    hazard_overlay[...,3] = hazard_map*0.5
    ax.plot_surface(X, Y, terrain_norm + 0.01, rstride=1, cstride=1,
                    facecolors=hazard_overlay, edgecolor='none')
    ax.set_title("3D Terrain with Hazard Zones")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Elevation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "terrain_3d.png"))
    plt.show()
    print("Saved: terrain_3d.png")

    # 3D terrain with stones
    fig2 = plt.figure(figsize=(14,8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, terrain_norm, rstride=2, cstride=2,
                     facecolors=plt.cm.terrain(terrain_norm), edgecolor='none', alpha=0.9)
    # Stones as spheres
    for x, y, r in stones:
        u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
        xs = x + r*np.cos(u)*np.sin(v)
        ys = y + r*np.sin(u)*np.sin(v)
        zs = terrain_norm[y,x] + r/PIXELS_PER_CM * np.cos(v)
        ax2.plot_surface(xs, ys, zs, color='red', alpha=0.8)
    ax2.set_title("3D Terrain with Stones")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Elevation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "terrain_stones_3d.png"))
    plt.show()
    print("Saved: terrain_stones_3d.png")

if __name__ == "__main__":
    main()
