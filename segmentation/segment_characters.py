import os
import cv2
import numpy as np


class CharacterSegmenter:

    def __init__(self, padding=18, row_tolerance=40, clean_output=False):
        self.padding = padding
        self.row_tolerance = row_tolerance
        self.clean_output = clean_output

    def _build_ink_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (25, 1),
        )
        horizontal_lines = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=1,
        )

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        protected_strokes = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=1,
        )
        horizontal_lines = cv2.bitwise_and(
            horizontal_lines,
            cv2.bitwise_not(protected_strokes),
        )

        clean = cv2.subtract(thresh, horizontal_lines)
        clean = cv2.morphologyEx(
            clean,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
            iterations=1,
        )

        return clean, thresh

    def _remove_horizontal_runs(self, mask_crop):
        cleaned = mask_crop.copy()
        min_run = max(8, int(mask_crop.shape[1] * 0.35))

        for row_index in range(mask_crop.shape[0]):
            cols = np.where(mask_crop[row_index] > 0)[0]
            if len(cols) < min_run:
                continue

            runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
            for run in runs:
                if len(run) < min_run:
                    continue

                y1 = max(0, row_index - 6)
                y2 = min(mask_crop.shape[0], row_index + 7)
                support_band = mask_crop[y1:y2, :]
                vertical_support = support_band[:, run].sum(axis=0) > 255 * 2

                removable_cols = run[~vertical_support]
                cleaned[max(0, row_index - 1):min(mask_crop.shape[0], row_index + 2), removable_cols] = 0

        return cleaned

    def _remove_bottom_rule_fragments(self, mask_crop):
        cleaned = mask_crop.copy()
        height, width = mask_crop.shape[:2]

        min_row = int(height * 0.55)
        min_span = int(width * 0.72)
        min_ink = int(width * 0.48)

        for row_index in range(min_row, height):
            cols = np.where(cleaned[row_index] > 0)[0]
            if len(cols) < min_ink:
                continue

            span = cols[-1] - cols[0] + 1
            touches_edge = cols[0] <= 2 or cols[-1] >= width - 3
            if span < min_span or not touches_edge:
                continue

            y1 = max(0, row_index - 1)
            y2 = min(height, row_index + 2)
            support_top = cleaned[max(0, row_index - 9):y1, cols] > 0
            support_bottom = cleaned[y2:min(height, row_index + 10), cols] > 0
            stroke_support = support_top.sum(axis=0) + support_bottom.sum(axis=0)
            removable_cols = cols[stroke_support < 2]

            cleaned[y1:y2, removable_cols] = 0

        return cleaned

    def _remove_bottom_specks(self, mask_crop):
        cleaned = mask_crop.copy()
        height = mask_crop.shape[0]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)

        for label in range(1, count):
            x, y, w, h, area = stats[label]

            if y < height * 0.55:
                continue

            if area <= 8 or h <= 2:
                cleaned[labels == label] = 0

        return cleaned
    
    def _tight_crop(self, image):

        inverted = cv2.bitwise_not(image)

        # -----------------------------------
        # FIND REAL INK PIXELS
        # -----------------------------------

        rows = np.where(np.sum(inverted > 0, axis=1) > 2)[0]
        cols = np.where(np.sum(inverted > 0, axis=0) > 2)[0]

        if len(rows) == 0 or len(cols) == 0:
            return image

        y1, y2 = rows[0], rows[-1]
        x1, x2 = cols[0], cols[-1]

        # -----------------------------------
        # EXTRA EDGE CLEANUP
        # -----------------------------------

        # remove weak left-edge bleed
        while x1 < x2:

            col_ink = np.sum(inverted[:, x1] > 0)

            if col_ink > 3:
                break

            x1 += 1

        # remove weak bottom bleed
        while y2 > y1:

            row_ink = np.sum(inverted[y2, :] > 0)

            if row_ink > 3:
                break

            y2 -= 1

        pad = 4

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        return image[y1:y2, x1:x2]
    
    def _tighten_crop(self, crop):

            coords = cv2.findNonZero(crop)

            if coords is None:
                return crop

            x, y, w, h = cv2.boundingRect(coords)

            # small safe margin
            pad = 6

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(crop.shape[1], x + w + pad)
            y2 = min(crop.shape[0], y + h + pad)

            return crop[y1:y2, x1:x2]

    def _clean_binary_crop(self, raw_mask, bounds):

        x1, y1, x2, y2 = bounds

        mask_crop = raw_mask[y1:y2, x1:x2]

        mask_crop = self._remove_horizontal_runs(mask_crop)

        mask_crop = self._remove_bottom_rule_fragments(mask_crop)

        mask_crop = self._remove_bottom_specks(mask_crop)

        # -----------------------------------
        # REMOVE ONLY TINY EDGE NOISE
        # -----------------------------------

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_crop, 8)

        cleaned = np.zeros_like(mask_crop)

        height, width = mask_crop.shape[:2]

        for label in range(1, count):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            touches_left = x <= 2
            touches_bottom = (y + h) >= (height - 2)

            tiny_noise = area < 18

            # remove only micro edge junk
            if tiny_noise and (touches_left or touches_bottom):
                continue

            cleaned[labels == label] = 255

        # -----------------------------------
        # SECOND PASS CLEANUP
        # -----------------------------------

        final_count, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(
            cleaned,
            8
        )

        for label in range(1, final_count):

            x = final_stats[label, cv2.CC_STAT_LEFT]
            y = final_stats[label, cv2.CC_STAT_TOP]
            w = final_stats[label, cv2.CC_STAT_WIDTH]
            h = final_stats[label, cv2.CC_STAT_HEIGHT]
            area = final_stats[label, cv2.CC_STAT_AREA]

            touches_left = x <= 4
            touches_bottom = (y + h) >= (height - 4)

            tiny_blob = area < 25

            if tiny_blob and (touches_left or touches_bottom):
                cleaned[final_labels == label] = 0

        # -----------------------------------
        # REMOVE LEFT EDGE BLEED
        # -----------------------------------

        height, width = cleaned.shape[:2]

        for col in range(min(10, width // 4)):

            column = cleaned[:, col]

            ink_rows = np.where(column > 0)[0]

            # empty edge column
            if len(ink_rows) == 0:
                cleaned[:, col] = 0
                continue

            vertical_span = ink_rows[-1] - ink_rows[0]

            # tiny left-edge artifact
            if vertical_span < height * 0.22:
                cleaned[:, col] = 0
            else:
                break

        # -----------------------------------
        # REMOVE BOTTOM EDGE BLEED
        # -----------------------------------

        for row in range(height - 1, max(height - 10, 0), -1):

            row_pixels = cleaned[row, :]

            ink = np.sum(row_pixels > 0)

            # weak bottom bleed only
            if ink <= width * 0.10:
                cleaned[row, :] = 0
            else:
                break

        return cv2.bitwise_not(cleaned)


    def _component_boxes(self, mask):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        image_height, image_width = mask.shape[:2]
        boxes = []

        for label in range(1, count):
            x, y, w, h, area = stats[label]

            if area < 20 or w < 3 or h < 5:
                continue

            if w > image_width * 0.75 or h > image_height * 0.08:
                continue

            # Leftover ruled-line fragments are usually wide and very short.
            if h <= 10 and w > 25:
                continue

            boxes.append((int(x), int(y), int(w), int(h)))

        return boxes

    def _group_boxes_by_row(self, boxes):
        rows = []

        for box in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
            center_y = box[1] + box[3] / 2

            if not rows:
                rows.append({"center_y": center_y, "boxes": [box]})
                continue

            row = rows[-1]
            if abs(center_y - row["center_y"]) <= self.row_tolerance:
                row["boxes"].append(box)
                row["center_y"] = sum(
                    b[1] + b[3] / 2 for b in row["boxes"]
                ) / len(row["boxes"])
            else:
                rows.append({"center_y": center_y, "boxes": [box]})

        merged_boxes = []
        for row in rows:
            xs = [b[0] for b in row["boxes"]]
            ys = [b[1] for b in row["boxes"]]
            rights = [b[0] + b[2] for b in row["boxes"]]
            bottoms = [b[1] + b[3] for b in row["boxes"]]
            merged_boxes.append((
                min(xs),
                min(ys),
                max(rights) - min(xs),
                max(bottoms) - min(ys),
            ))

        return merged_boxes

    def segment_column(self, column_image, output_dir, prefix):
        mask, raw_mask = self._build_ink_mask(column_image)
        boxes = self._component_boxes(mask)
        boxes = self._group_boxes_by_row(boxes)

        for count, box in enumerate(boxes):
            x, y, w, h = box

            left_pad = self.padding
            top_pad = self.padding
            right_pad = self.padding
            bottom_pad = self.padding

            # reduce aggressive expansion
            # specifically where bleed happens

            if prefix == "lower":
                left_pad = max(6, self.padding - 10)
                bottom_pad = max(6, self.padding - 10)

            x1 = max(0, x - left_pad)
            y1 = max(0, y - top_pad)

            x2 = min(column_image.shape[1], x + w + right_pad)
            y2 = min(column_image.shape[0], y + h + bottom_pad)

            char_crop = self._clean_binary_crop(
                raw_mask,
                (x1, y1, x2, y2)
            )
            # dynamic re-tightening
            inverted = cv2.bitwise_not(char_crop)

            tight = self._tighten_crop(inverted)

            char_crop = cv2.bitwise_not(tight)

            char_crop = self._tight_crop(char_crop)

            save_path = os.path.join(
                output_dir,
                f"{prefix}_{count:02d}.png"
            )


            cv2.imwrite(save_path, char_crop)

            print(f"[SAVED] {save_path}")

        print(f"[INFO] {prefix}: saved {len(boxes)} crops")

    def segment(self, image_path, output_dir):

        os.makedirs(output_dir, exist_ok=True)
        if self.clean_output:
            for filename in os.listdir(output_dir):
                if filename.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(output_dir, filename))
                    except PermissionError:
                        print(f"[WARN] Could not remove locked file: {filename}")

        image = cv2.imread(image_path)

        if image is None:
            raise Exception(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]

        # -------------------------
        # SPLIT INTO 3 COLUMNS
        # -------------------------

        col_width = width // 3

        uppercase_col = image[:, 0:col_width]

        lowercase_col = image[:, col_width:col_width*2]

        symbols_col = image[:, col_width*2:width]

        print("[INFO] Processing uppercase column...")
        self.segment_column(
            uppercase_col,
            output_dir,
            "upper"
        )

        print("[INFO] Processing lowercase column...")
        self.segment_column(
            lowercase_col,
            output_dir,
            "lower"
        )

        print("[INFO] Processing symbols column...")
        self.segment_column(
            symbols_col,
            output_dir,
            "symbol"
        )
