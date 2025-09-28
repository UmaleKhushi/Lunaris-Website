import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from tkinter import Tk, filedialog
import os
import csv
from mpl_toolkits.mplot3d import Axes3D

PIXELS_PER_CM = 37  # pixels per cm for scaling
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- 3D FUNCTIONS -------------------
def compute_depth_map(img_rgb, blur_sigma=8):
    gray = 0.2989*img_rgb[...,0] + 0.5870*img_rgb[...,1] + 0.1140*img_rgb[...,2]
    depth = 1.0 - (gray / 255.0)
    depth_smooth = gaussian_filter(depth, sigma=blur_sigma)
    depth_norm = (depth_smooth - depth_smooth.min()) / (depth_smooth.max() - depth_smooth.min())
    return depth_norm

def create_shifted_image(img, depth, max_shift=25, direction='left'):
    h, w = depth.shape
    sign = 1 if direction == 'left' else -1
    shift_px = ((depth - 0.5) * 2.0 * max_shift * sign).astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (grid_x + shift_px).astype(np.float32)
    map_y = grid_y.astype(np.float32)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    shifted = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)

def make_anaglyph(left_rgb, right_rgb):
    anaglyph = np.zeros_like(left_rgb)
    anaglyph[...,0] = left_rgb[...,0]
    anaglyph[...,1:] = right_rgb[...,1:]
    return anaglyph

# ------------------- STONE DETECTION -------------------
def detect_stones_contour(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    stone_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 5:
            diameter_px = 2 * radius
            diameter_cm = diameter_px / PIXELS_PER_CM  # real-world diameter
            results.append((stone_id, center[0], center[1], radius, diameter_cm))
            cv2.circle(img_rgb, center, radius, (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{diameter_cm:.1f}cm", (center[0]-40, center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(img_rgb, f"ID:{stone_id}", (center[0]-40, center[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            stone_id += 1
    return img_rgb, results

# ------------------- SAVE UTILS -------------------
def save_image(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}")

def save_csv(stone_info):
    csv_path = os.path.join(OUTPUT_DIR, "stone_report.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Stone_ID", "X_px", "Y_px", "Radius_px", "Diameter_cm"])
        for stone in stone_info:
            writer.writerow(stone)
    print(f"Saved CSV Report: {csv_path}")

# ------------------- MAIN -------------------
def main():
    root = Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(title="Select an Image",
                                          filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not img_path:
        print("No file selected.")
        return

    img = np.array(Image.open(img_path).convert('RGB'))
    depth = compute_depth_map(img)
    left = create_shifted_image(img, depth, direction='left')
    right = create_shifted_image(img, depth, direction='right')
    anaglyph = make_anaglyph(left, right)
    stones_img, stone_info = detect_stones_contour(img.copy())
    print("Detected stones (cm):", [round(s[4],2) for s in stone_info])

    # 2x3 panel
    fig, axes = plt.subplots(2,3,figsize=(15,10))
    axes[0,0].imshow(img); axes[0,0].set_title("Original"); axes[0,0].axis("off")
    axes[0,1].imshow(left); axes[0,1].set_title("Left View"); axes[0,1].axis("off")
    axes[0,2].imshow(right); axes[0,2].set_title("Right View"); axes[0,2].axis("off")
    axes[1,0].imshow(anaglyph); axes[1,0].set_title("Anaglyph 3D"); axes[1,0].axis("off")
    axes[1,1].imshow(depth, cmap="inferno"); axes[1,1].set_title("Depth Map"); axes[1,1].axis("off")
    axes[1,2].imshow(stones_img); axes[1,2].set_title("Stone Detection"); axes[1,2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_results.png"))
    plt.show()

    # save images
    save_image("original.jpg", img)
    save_image("left.jpg", left)
    save_image("right.jpg", right)
    save_image("anaglyph.jpg", anaglyph)
    save_image("depth.jpg", (depth*255).astype(np.uint8))
    save_image("stone_detection.jpg", stones_img)
    save_csv(stone_info)

    # 3D surface with spheres scaled to real diameter
    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    fig3d = plt.figure(figsize=(10,8))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, depth, rstride=5, cstride=5, facecolors=img/255.0, shade=False)

    for stone in stone_info:
        sid, x, y, radius_px, diameter_cm = stone
        z = depth[y, x]
        color = img[y, x] / 255.0

        # Sphere radius in pixels = diameter in cm * pixels per cm / 2
        radius_3d = (diameter_cm * PIXELS_PER_CM) / 2.0
        u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
        xs = x + radius_3d * np.cos(u) * np.sin(v)
        ys = y + radius_3d * np.sin(u) * np.sin(v)
        zs = z + radius_3d * np.cos(v) / PIXELS_PER_CM  # scale depth slightly

        ax.plot_surface(xs, ys, zs, color=color, alpha=0.8)
        ax.text(x, y, z+0.02, f"ID:{sid}", color='black')

    ax.set_title("3D Surface with Real-Size Colored Stones")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    plt.show()

if __name__ == "__main__":
    main()
