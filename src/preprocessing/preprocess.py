import cv2
import numpy as np
import os


def remove_tiny_specks(binary, max_area=50, max_width=6, max_height=6):
    """
    Remove only very small isolated dots / specks.
    Keep anything that looks like a real handwriting stroke.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Keep all normal handwriting components; drop only tiny isolated blobs.
        if area > max_area or w > max_width or h > max_height:
            cleaned[labels == i] = 255

    return cleaned


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flatten the page background first so faint back-side marks fade out.
    background = cv2.GaussianBlur(gray, (61, 61), 0)
    normalized = cv2.divide(gray, background, scale=255)

    # Median blur helps with tiny isolated specks after normalization.
    blur = cv2.medianBlur(normalized, 3)

    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    # Only remove tiny back-side specks.
    clean = remove_tiny_specks(thresh, max_area=50, max_width=6, max_height=6)

    # A tiny opening pass removes leftover isolated dots without changing letters much.
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    # Deskew the image
    coords = np.column_stack(np.where(clean > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 3.0:
            (h, w) = clean.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            clean = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return clean


def process_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_dir, file)
            processed = preprocess_image(path)

            if processed is not None:
                save_path = os.path.join(output_dir, file)
                cv2.imwrite(save_path, processed)
                print(f"Processed: {file}")


if __name__ == "__main__":
    process_all("data/raw", "data/processed")
