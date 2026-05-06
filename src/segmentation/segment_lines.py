import cv2
import numpy as np
import os

def foreground_mask(gray):
    if gray.size == 0:
        return gray

    h, w = gray.shape
    border = max(5, min(h, w) // 20)
    corners = np.concatenate(
        [
            gray[:border, :border].ravel(),
            gray[:border, -border:].ravel(),
            gray[-border:, :border].ravel(),
            gray[-border:, -border:].ravel(),
        ]
    )

    dark_background = float(np.median(corners)) < 127.0
    threshold_type = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU,
    )
    return binary

def is_rule_like_crop(gray):
    if gray.size == 0:
        return False

    mask = foreground_mask(gray)
    ys, xs = np.where(mask > 0)
    if xs.size < 20:
        return False

    # Ruled notebook lines are usually close to a single straight line.
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    design = np.vstack([xs_f, np.ones_like(xs_f)]).T
    slope, intercept = np.linalg.lstsq(design, ys_f, rcond=None)[0]
    residuals = ys_f - (slope * xs_f + intercept)

    tolerance = max(3.0, float(gray.shape[0]) * 0.08)
    line_fraction = float(np.mean(np.abs(residuals) <= tolerance))

    if line_fraction < 0.85:
        return False

    return True

def is_blank_crop(gray):
    if gray.size == 0:
        return True

    mask = foreground_mask(gray)

    # Ruled notebook lines can look like foreground, but they are not text.
    if is_rule_like_crop(gray):
        return True

    fg_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    ratio = fg_pixels / float(total_pixels)

    # 🔥 1. Very low ink → blank
    if ratio < 0.02:
        return True

    # 🔥 2. Weak horizontal signal → blank
    row_sums = np.sum(mask > 0, axis=1)
    if np.max(row_sums) < 10:
        return True

    # 🔥 3. No meaningful components → blank
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return True

    max_area = max(stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels))

    if max_area < 50:
        return True

    return False


def merge_ranges(ranges, max_gap=4):
    if not ranges:
        return []

    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def remove_horizontal_rules(binary, image_width):
    # Notebook ruling lines are long and thin, so a wide horizontal opening isolates them well.
    kernel_width = max(80, int(image_width * 0.35))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    return cv2.subtract(binary, horizontal)


def remove_hough_horizontal_lines(binary):
    if binary.size == 0:
        return binary

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(120, int(binary.shape[1] * 0.35)),
        maxLineGap=25,
    )

    if lines is None:
        return binary

    mask = np.zeros_like(binary)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        slope = abs(dy / dx) if dx != 0 else float("inf")
        if length >= binary.shape[1] * 0.85 and slope <= 0.02:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 5)

    if not np.any(mask):
        return binary

    cleaned = binary.copy()
    cleaned[mask > 0] = 0
    return cleaned


def remove_full_width_rows(binary, coverage_ratio=0.92):
    if binary.size == 0:
        return binary

    row_sums = np.sum(binary > 0, axis=1)
    separator_rows = row_sums >= int(binary.shape[1] * coverage_ratio)
    if not np.any(separator_rows):
        return binary

    cleaned = binary.copy()
    cleaned[separator_rows, :] = 0
    return cleaned


def split_on_separator_rows(crop, coverage_ratio=0.995, min_gap=50):
    if crop.size == 0:
        return [crop]

    binary = foreground_mask(crop)

    row_sums = np.sum(binary > 0, axis=1)
    separator_rows = np.where(row_sums >= int(binary.shape[1] * coverage_ratio))[0]
    if separator_rows.size < 2:
        return [crop]

    clusters = []
    start = int(separator_rows[0])
    prev = int(separator_rows[0])
    for row in map(int, separator_rows[1:]):
        if row == prev + 1:
            prev = row
        else:
            clusters.append((start, prev))
            start = prev = row
    clusters.append((start, prev))

    gaps = []
    for (left_start, left_end), (right_start, right_end) in zip(clusters, clusters[1:]):
        gap_start = left_end + 1
        gap_end = right_start - 1
        gap_len = gap_end - gap_start + 1
        if gap_len >= min_gap:
            gaps.append((gap_start, gap_end, gap_len))

    if not gaps:
        return [crop]

    gap_start, gap_end, gap_len = max(gaps, key=lambda item: item[2])
    cluster_count = len(clusters)
    min_required_gap = max(min_gap, int(crop.shape[0] * 0.2))
    if gap_len < min_required_gap:
        return [crop]

    height = crop.shape[0]
    if cluster_count < 3 and not (cluster_count == 2 and 80 <= height <= 110 and gap_len >= 50):
        return [crop]

    split_y = (gap_start + gap_end) // 2

    if split_y <= int(height * 0.2) or split_y >= int(height * 0.8):
        return [crop]

    return [crop[:split_y + 1, :], crop[split_y + 1:, :]]


def _split_range_recursive(start, end, smoothed, split_limit):
    height = end - start
    if height <= split_limit:
        return [(start, end)]

    band = smoothed[start:end + 1]
    if band.size < 20:
        return [(start, end)]

    # Prefer cutting at actual whitespace between text lines.
    local_threshold = max(6.0, float(band.max()) * 0.18)
    active = band > local_threshold

    gap_start = None
    gaps = []
    for idx, is_active in enumerate(active):
        if not is_active and gap_start is None:
            gap_start = idx
        elif is_active and gap_start is not None:
            gap_end = idx - 1
            if gap_end - gap_start + 1 >= 20:
                gaps.append((gap_start, gap_end))
            gap_start = None

    if gap_start is not None:
        gap_end = band.size - 1
        if gap_end - gap_start + 1 >= 20:
            gaps.append((gap_start, gap_end))

    if gaps:
        band_center = (band.size - 1) / 2.0

        def gap_key(gap):
            gap_start_idx, gap_end_idx = gap
            gap_width = gap_end_idx - gap_start_idx + 1
            gap_center = (gap_start_idx + gap_end_idx) / 2.0
            # Prefer wide gaps near the middle of the band.
            return (-gap_width, abs(gap_center - band_center))

        gap_start_idx, gap_end_idx = min(gaps, key=gap_key)
        split_y = start + (gap_start_idx + gap_end_idx) // 2

        if start + 10 < split_y < end - 10:
            left_parts = _split_range_recursive(start, split_y, smoothed, split_limit)
            right_parts = _split_range_recursive(split_y + 1, end, smoothed, split_limit)
            return left_parts + right_parts

    # Fallback: split at the deepest valley away from the edges.
    margin = max(8, int(height * 0.15))
    left = min(max(margin, 1), band.size - 2)
    right = max(left + 1, band.size - margin - 1)
    if right <= left:
        return [(start, end)]

    valley_offset = int(np.argmin(band[left:right + 1])) + left
    valley_y = start + valley_offset
    band_peak = float(band.max())
    valley_value = float(smoothed[valley_y])

    if valley_y <= start + 15 or valley_y >= end - 15:
        return [(start, end)]

    # Keep the split conservative if the valley is shallow.
    if valley_value > band_peak * 0.55:
        return [(start, end)]

    left_parts = _split_range_recursive(start, valley_y, smoothed, split_limit)
    right_parts = _split_range_recursive(valley_y + 1, end, smoothed, split_limit)
    return left_parts + right_parts


def split_tall_ranges(ranges, smoothed):
    if not ranges:
        return []

    heights = np.array([end - start for start, end in ranges], dtype=np.float32)
    median_height = float(np.median(heights)) if heights.size else 0.0
    split_limit = max(100, int(median_height * 2.0))

    split_ranges = []
    for start, end in ranges:
        split_ranges.extend(_split_range_recursive(start, end, smoothed, split_limit))

    return split_ranges


def add_soft_border(gray, border=8):
    if gray.size == 0 or border <= 0:
        return gray

    core = gray
    if gray.shape[0] > border * 2:
        core = gray[border:-border, :]

    nonzero = core[core > 0]
    if nonzero.size:
        background = int(np.median(nonzero))
    else:
        background = int(np.median(core))

    return cv2.copyMakeBorder(
        gray,
        border,
        border,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=background,
    )


def trim_border(gray, border=8):
    if gray.size == 0 or border <= 0:
        return gray
    if gray.shape[0] <= max(60, border * 2):
        return gray
    return gray[border:-border, :]


def is_text_like_crop(gray):
    if gray.size == 0:
        return False
    
    # NEW: remove blank lines early
    if is_blank_crop(gray):
        return False

    mask = foreground_mask(gray)

    fg_pixels = int(np.count_nonzero(mask))
    if fg_pixels == 0:
        return False

    # Lower threshold for foreground ratio to catch faint text
    ratio = fg_pixels / float(mask.size)
    if ratio >= 0.03:  # Reduced from 0.08
        return True

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Skip connected components analysis if no foreground pixels found
    if num_labels <= 1:
        # For very sparse images, still check if it might be text based on size
        # But we need at least some pixels to make a reasonable judgment
        if fg_pixels < 5:  # Reduced threshold for minimal text
            return False
        # For single component images (like a single word), check dimensions directly
        # Find the bounding box of all foreground pixels
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Reduced size requirements for short text
            if h >= 10 and w >= 8:  # Reduced from 20x20
                # Additional check to filter out horizontal lines
                if w > 0 and h > 0:
                    aspect_ratio = w / h
                    if aspect_ratio <= 20:  # Increased from 15 to be more permissive
                        return True
            return False
        return False
    
    max_height = max(int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, num_labels))
    max_width = max(int(stats[i, cv2.CC_STAT_WIDTH]) for i in range(1, num_labels))
    
    # Reduced size requirements
    if max_height < 10 or max_width < 8:  # Reduced from 20x20
        return False
    
    # Additional check to filter out horizontal lines (like notebook ruled lines)
    # If width is excessively large compared to height, it's likely just a line
    if max_width > 0 and max_height > 0:
        aspect_ratio = max_width / max_height
        # If aspect ratio is too high (very wide and short), it's likely a horizontal line
        if aspect_ratio > 20:  # Increased from 15 to be more permissive
            return False
    
    return True

def add_line_margin(gray, top=8, bottom=6):
    if gray.size == 0:
        return gray
    return cv2.copyMakeBorder(gray, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)


def split_crop_by_projection(crop):
    if crop.size == 0:
        return [crop]

    separator_parts = split_on_separator_rows(crop)
    if len(separator_parts) > 1:
        return separator_parts

    h, w = crop.shape
    binary = foreground_mask(crop)

    # Remove any residual horizontal ruling inside the crop.
    binary = remove_horizontal_rules(binary, w)
    binary = remove_hough_horizontal_lines(binary)
    binary = remove_full_width_rows(binary)

    row_sums = np.sum(binary > 0, axis=1).astype(np.float32)
    if row_sums.size == 0 or float(row_sums.max()) <= 0:
        return [crop]

    window = max(9, int(h * 0.06))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(row_sums, kernel, mode="same")

    active_threshold = max(6.0, float(smoothed.max()) * 0.1)
    active = smoothed > active_threshold

    ranges = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            end = y - 1
            if end - start >= 20:
                ranges.append((start, end))
            start = None

    if start is not None:
        end = h - 1
        if end - start >= 20:
            ranges.append((start, end))

    if len(ranges) <= 1:
        return [crop]

    heights = np.array([end - start for start, end in ranges], dtype=np.float32)
    median_height = float(np.median(heights)) if heights.size else 0.0
    split_limit = max(100, int(median_height * 2.0))

    parts = []
    for start, end in ranges:
        if end - start <= split_limit:
            parts.append(crop[start:end + 1, :])
        else:
            band = smoothed[start:end + 1]
            band_peak = float(band.max())
            margin = max(2, int((end - start) * 0.08))
            left = min(max(margin, 1), band.size - 2)
            right = max(left + 1, band.size - margin - 1)
            if right <= left:
                parts.append(crop[start:end + 1, :])
                continue

            valley_offset = int(np.argmin(band[left:right + 1])) + left
            split_y = start + valley_offset
            valley_depth = band_peak - float(band[valley_offset])
            if split_y <= start + 10 or split_y >= end - 10:
                parts.append(crop[start:end + 1, :])
                continue
            if valley_depth < band_peak * 0.75:
                parts.append(crop[start:end + 1, :])
                continue

            parts.extend(split_crop_by_projection(crop[start:split_y + 1, :]))
            parts.extend(split_crop_by_projection(crop[split_y + 1:end + 1, :]))

    if not parts:
        return [crop]

    refined = []
    for part in parts:
        if part.shape[0] <= 0:
            continue
        if len(refined) > 0:
            prev = refined[-1]
            # Avoid tiny overlaps or accidental duplicates from recursive splits.
            if abs(prev.shape[0] - part.shape[0]) == 0 and prev.shape == part.shape:
                continue
        refined.append(part)

    return refined if refined else [crop]


def segment_lines(image_path, output_dir):
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Error reading {image_path}")
        return []

    h, w = img.shape

    # Binary text mask.
    binary = foreground_mask(img)

    # Remove long notebook rules before looking for text bands.
    binary = remove_horizontal_rules(binary, w)
    binary = remove_hough_horizontal_lines(binary)
    binary = remove_full_width_rows(binary)

    # Horizontal projection of text pixels.
    row_sums = np.sum(binary > 0, axis=1).astype(np.float32)
    smooth_window = max(15, int(h * 0.01))
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.convolve(row_sums, kernel, mode="same")

    if smoothed.size == 0 or float(smoothed.max()) <= 0:
        print(f"No text detected in {image_path}")
        return []

    active_threshold = max(8.0, float(smoothed.max()) * 0.08)
    active = smoothed > active_threshold

    ranges = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            end = y - 1
            if end - start >= 20:
                ranges.append((start, end))
            start = None

    if start is not None:
        end = h - 1
        if end - start >= 20:
            ranges.append((start, end))

    ranges = merge_ranges(ranges, max_gap=1)
    ranges = split_tall_ranges(ranges, smoothed)
    ranges = merge_ranges(sorted(ranges), max_gap=2)

    if not ranges:
        print(f"No line bands detected in {image_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    band_heights = np.array([end - start for start, end in ranges], dtype=np.float32)
    median_band_height = float(np.median(band_heights)) if band_heights.size else 0.0
    refine_limit = max(80, int(median_band_height * 1.55))

    line_paths = []
    prev_y2 = -1
    for idx, (y1_raw, y2_raw) in enumerate(ranges):
        band_height = y2_raw - y1_raw
        pad_top = max(8, int(band_height * 0.15))
        pad_bottom = max(10, int(band_height * 0.15))

        if prev_y2 < 0:
            y1 = y1_raw - 20
        else:
            y1 = max(y1_raw - pad_top, prev_y2 + 1)
        y1 = max(y1, 0)
        y2 = min(y2_raw + pad_bottom, h)

        if y1 >= y2:
            continue

        prev_y2 = y2

        line_img = img[y1:y2, :]
        saved_height = line_img.shape[0] + 16
        should_refine = band_height >= refine_limit or saved_height >= 150
        if should_refine:
            refined_parts = split_crop_by_projection(line_img)
        else:
            refined_parts = [line_img]
        if len(refined_parts) == 1:
            if not is_text_like_crop(refined_parts[0]):
                continue
            path = os.path.join(output_dir, f"{base}_line_{idx}.png")
            cv2.imwrite(path, refined_parts[0])
            line_paths.append(path)
            continue

        for part_idx, part in enumerate(refined_parts):
            if not is_text_like_crop(part):
                continue
            path = os.path.join(output_dir, f"{base}_line_{idx}_part_{part_idx}.png")
            cv2.imwrite(path, part)
            line_paths.append(path)

    print(f"{len(line_paths)} lines detected in {base}")
    return line_paths
