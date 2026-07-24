import os
import glob
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize, rotate
import cv2


MODEL_PATH = "Colorized-Img\Colorized-Img\colorize_autoencoder_first1000.keras"
IMAGE_FOLDER = "Colorized-Img"
IMAGE_EXTENSIONS = ("forest397.jpg", "forest399.jpg", "ZYLOMHW6DWLE.jpg", "YIN8QU2CC97X.jpg")


model = tf.keras.models.load_model(MODEL_PATH)
model.summary()


def load_and_resize(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, target_size, preserve_range=True).astype(np.uint8)
    return img

def grayscale_to_L_method1(gray_img):
    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    lab = rgb2lab(gray_rgb / 255.0)
    return lab[:, :, 0]

def grayscale_to_L_method2(gray_img):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, gray_img)
    gray_read = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    os.unlink(temp_path)
    gray_rgb = cv2.cvtColor(gray_read, cv2.COLOR_BGR2RGB)
    lab = rgb2lab(gray_rgb / 255.0)
    return lab[:, :, 0]

def colorize_image(model, rgb_original, method='method1'):
    gray = cv2.cvtColor(rgb_original, cv2.COLOR_RGB2GRAY)
    if method == 'method1':
        L = grayscale_to_L_method1(gray)
    else:
        L = grayscale_to_L_method2(gray)
    L_input = L.reshape(1, 256, 256, 1)
    ab_pred = model.predict(L_input, verbose=0)[0]
    a_pred = ab_pred[:, :, 0] * 128
    b_pred = ab_pred[:, :, 1] * 128
    lab_pred = np.stack([L, a_pred, b_pred], axis=-1)
    rgb_pred = lab2rgb(lab_pred)
    return (rgb_pred * 255).astype(np.uint8)


image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, f"*{ext}")))

if len(image_paths) == 0:
    print("No images found. Check IMAGE_FOLDER.")
    exit()


###    === PART 3: Compare Method 1 and Method 2 ===
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i, img_path in enumerate(image_paths):
    orig = load_and_resize(img_path, (256, 256))
    colorized_m1 = colorize_image(model, orig, method='method1')
    colorized_m2 = colorize_image(model, orig, method='method2')
    axes[i, 0].imshow(orig)
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')
    axes[i, 1].imshow(colorized_m1)
    axes[i, 1].set_title("Method 1")
    axes[i, 1].axis('off')
    axes[i, 2].imshow(colorized_m2)
    axes[i, 2].set_title("Method 2")
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# Analysis of L Similarity
sample_gray = cv2.cvtColor(load_and_resize(image_paths[0], (256,256)), cv2.COLOR_RGB2GRAY)
L1 = grayscale_to_L_method1(sample_gray)
L2 = grayscale_to_L_method2(sample_gray)
print(f"Max difference between L from method1 and method2: {np.max(np.abs(L1 - L2)):.6f}")


###    === PART 4: Robustness Tests (using Method 1 only) ===
test_images = [load_and_resize(p, (256,256)) for p in image_paths[:3]]

def rotate_and_colorize(model, rgb_img, angle=45):
    rotated = rotate(rgb_img, angle, preserve_range=True).astype(np.uint8)
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    L_rot = grayscale_to_L_method1(gray_rot)
    L_input = L_rot.reshape(1,256,256,1)
    ab_rot = model.predict(L_input, verbose=0)[0]
    a_rot = ab_rot[:,:,0]*128
    b_rot = ab_rot[:,:,1]*128
    lab_rot = np.stack([L_rot, a_rot, b_rot], axis=-1)
    rgb_rot = lab2rgb(lab_rot)
    rgb_rot = (rgb_rot*255).astype(np.uint8)
    rgb_back = rotate(rgb_rot, -angle, preserve_range=True).astype(np.uint8)
    h, w = rgb_img.shape[:2]
    rgb_back = resize(rgb_back, (h,w), preserve_range=True).astype(np.uint8)
    return rgb_back

###    --- Rotation Test (45°) ---
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
for idx, img in enumerate(test_images):
    orig_col = colorize_image(model, img, method='method1')
    rot_col = rotate_and_colorize(model, img, 45)
    axes[idx,0].imshow(orig_col)
    axes[idx,0].set_title(f"Original Colorized {idx+1}")
    axes[idx,0].axis('off')
    axes[idx,1].imshow(rot_col)
    axes[idx,1].set_title(f"After Rotation {idx+1}")
    axes[idx,1].axis('off')
plt.tight_layout()
plt.show()

def add_gaussian_noise(L, std=0.5):
    noise = np.random.normal(0, std, L.shape)
    L_noisy = L + noise
    return np.clip(L_noisy, 0, 100)

def colorize_with_noise(model, rgb_img, std=0.1):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    L_clean = grayscale_to_L_method1(gray)
    L_noisy = add_gaussian_noise(L_clean, std)
    L_input = L_noisy.reshape(1,256,256,1)
    ab_pred = model.predict(L_input, verbose=0)[0]
    a_pred = ab_pred[:,:,0]*128
    b_pred = ab_pred[:,:,1]*128
    lab_pred = np.stack([L_clean, a_pred, b_pred], axis=-1)
    rgb_noisy = lab2rgb(lab_pred)
    return (rgb_noisy*255).astype(np.uint8)

###    --- Gaussian Noise Test (std=0.1) ---
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
for idx, img in enumerate(test_images):
    orig_col = colorize_image(model, img, method='method1')
    noisy_col = colorize_with_noise(model, img, std=0.1)
    axes[idx,0].imshow(orig_col)
    axes[idx,0].set_title(f"Original Colorized {idx+1}")
    axes[idx,0].axis('off')
    axes[idx,1].imshow(noisy_col)
    axes[idx,1].set_title(f"With Noise {idx+1}")
    axes[idx,1].axis('off')
plt.tight_layout()
plt.show()

def adjust_brightness(L, factor):
    L_adj = L * factor
    return np.clip(L_adj, 10, 100)

def colorize_with_brightness(model, rgb_img, factor):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    L_orig = grayscale_to_L_method1(gray)
    L_adj = adjust_brightness(L_orig, factor)
    L_input = L_adj.reshape(1,256,256,1)
    ab_pred = model.predict(L_input, verbose=0)[0]
    a_pred = ab_pred[:,:,0]*128
    b_pred = ab_pred[:,:,1]*128
    lab_pred = np.stack([L_adj, a_pred, b_pred], axis=-1)
    rgb_adj = lab2rgb(lab_pred)
    return (rgb_adj*255).astype(np.uint8)

###    --- Brightness Test (x0.7 and x1.3) ---
for idx, img in enumerate(test_images):
    orig_col = colorize_image(model, img, method='method1')
    dark_col = colorize_with_brightness(model, img, 0.7)
    bright_col = colorize_with_brightness(model, img, 1.3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_col)
    axes[0].set_title(f"Original Brightness {idx+1}")
    axes[0].axis('off')
    axes[1].imshow(dark_col)
    axes[1].set_title("x0.7")
    axes[1].axis('off')
    axes[2].imshow(bright_col)
    axes[2].set_title("x1.3")
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
