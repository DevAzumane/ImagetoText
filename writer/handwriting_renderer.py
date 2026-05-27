import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from writer.layout_engine import LayoutEngine


class HandwritingRenderer:

    # Handwriting typography rules for letter sizing
    # Ascenders: tall letters that reach above x-height (should be ~1.15-1.25)
    # Regular x-height: normal lowercase letters (1.0)
    # Descenders: letters that dip below baseline - should be FULL x-height but positioned to extend down
    GLYPH_SCALE_OVERRIDES = {
        # Ascenders
        "b": 1.18,
        "d": 1.18,
        "f": 1.55,
        "h": 1.18,
        "k": 1.18,
        "l": 1.24,
        "t": 1.14,

        # Descenders
        "g": 1.55,
        "j": 1.55,
        "p": 1.55,
        "q": 1.22,
        "y": 1.77,

        # Width balance
        "m": 1.04,
        "w": 1.04,
        "v": 1.04,
    }

    GLYPH_BASELINE_OFFSETS = {
    "ascender": 0,
    "regular": 2,
    "descender": -8,
    }

    def parse_layout_text(self, text):

        blocks = []

        lines = text.splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # -------------------------
            # TITLE
            # -------------------------

            if line.startswith("TITLE"):
                _, value = line.split(":", 1)

                blocks.append({
                    "type": "title",
                    "text": value.strip()
                })

                continue

            # -------------------------
            # IMAGE + SUMMARY
            # -------------------------

            if ":" in line:

                key, value = line.split(":", 1)

                key = key.strip()
                value = value.strip()

                if key.lower().startswith("img"):

                    blocks.append({
                        "type": "image_name",
                        "text": key
                    })

                    blocks.append({
                        "type": "summary",
                        "text": value
                    })

        return blocks    

    def _get_visible_width(self, image):
        alpha = np.array(image.getchannel("A"))
        
        cols = np.where(alpha.max(axis=0) > 0)[0]

        if len(cols) == 0:
            return image.width

        return cols[-1] - cols[0] + 1

    def __init__(self, seed=7, font_path=None):

        self.font_path = font_path or "assets/fonts/Caveat-VariableFont_wght.ttf"

        self.font_size = 64

        self.font = ImageFont.truetype(
            self.font_path,
            self.font_size
        )

        self.random = random.Random(seed)
        self.glyph_dir = Path("glyphs/segmented")
        self.glyphs = self._load_glyphs()
        self.templates = {
            "classic": {
                "paper": (248, 246, 238),
                "grain": 5,
                "line_color": (184, 202, 232),
                "line_shadow": (220, 228, 242),
                "margin_color": (229, 132, 126),
                "margin_x": 92,
                "line_gap": 40,
                "first_line_y": 120,
                "left_edge": 55,
                "right_edge": 35,
                "holes": False,
                "top_rule": False,
                "baseline_nudge": 0,
            },
            "spiral": {
                "paper": (250, 249, 242),
                "grain": 7,
                "line_color": (176, 199, 226),
                "line_shadow": (226, 232, 241),
                "margin_color": (224, 125, 118),
                "margin_x": 118,
                "line_gap": 48,
                "first_line_y": 128,
                "left_edge": 82,
                "right_edge": 35,
                "holes": True,
                "top_rule": False,
                "baseline_nudge": 0,
            },
            "exam": {
                "paper": (246, 248, 244),
                "grain": 4,
                "line_color": (194, 211, 224),
                "line_shadow": (229, 235, 240),
                "margin_color": (214, 151, 143),
                "margin_x": 96,
                "line_gap": 44,
                "first_line_y": 136,
                "left_edge": 58,
                "right_edge": 35,
                "holes": False,
                "top_rule": True,
                "baseline_nudge": 0,
            },
            "aged": {
                "paper": (240, 233, 216),
                "grain": 9,
                "line_color": (174, 190, 211),
                "line_shadow": (211, 214, 217),
                "margin_color": (202, 123, 116),
                "margin_x": 104,
                "line_gap": 44,
                "first_line_y": 124,
                "left_edge": 62,
                "right_edge": 35,
                "holes": False,
                "top_rule": False,
                "baseline_nudge": 0,
            },
        }

    def _load_glyphs(self):
        glyphs = {}

        for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            path = self.glyph_dir / f"upper_{index:02d}.png"
            if path.exists():
                glyphs[letter] = self._prepare_glyph(path)

        for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
            path = self.glyph_dir / f"lower_{index:02d}.png"
            if path.exists():
                glyphs[letter] = self._prepare_glyph(path)

        symbol_map = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4,
            "6": 5,
            "7": 6,
            "8": 7,
            "9": 8,
            "!": 9,
            '"': 10,
            "'": 11,
            ";": 12,
            ":": 13,
            ".": 14,
            "?": 15,
        }

        for character, index in symbol_map.items():
            path = self.glyph_dir / f"symbol_{index:02d}.png"
            if path.exists():
                glyphs[character] = self._prepare_glyph(path)

        return glyphs

    def _prepare_glyph(self, path):

        image = Image.open(path).convert("L")

        mask = image.point(
            lambda value: 255 if value < 245 else 0
        )

        ink_box = self._robust_ink_box(mask)

        if ink_box:
            left, top, right, bottom = ink_box
            image.ink_top = top
            image.ink_bottom = bottom
            image = image.crop(ink_box)

        # ---------------------------------
        # STANDARDIZED GLYPH CANVAS
        # ---------------------------------

        STANDARD_HEIGHT = image.height + 8
        STANDARD_WIDTH = image.width + 8

        canvas = Image.new(
            "L",
            (STANDARD_WIDTH, STANDARD_HEIGHT),
            255
        )

        # center glyph
        x = (STANDARD_WIDTH - image.width) // 2
        y = (STANDARD_HEIGHT - image.height) // 2

        canvas.paste(image, (x, y))

        image = canvas

        alpha = image.point(
            lambda value: max(
                0,
                min(255, int((255 - value) * 1.8))
            )
        )

        glyph = Image.new(
            "RGBA",
            image.size,
            (28, 30, 34, 0)
        )

        glyph.putalpha(alpha)
        glyph.ink_top = getattr(image, "ink_top", 0)
        glyph.ink_bottom = getattr(image, "ink_bottom", image.height)
        glyph.ink_height = image.height

        return glyph

    def _robust_ink_box(self, mask):
        ink = np.array(mask)
        occupied = ink > 0

        if not occupied.any():
            return None

        column_threshold = max(3, int(occupied.shape[0] * 0.05))
        row_threshold = max(3, int(occupied.shape[1] * 0.04))

        active_cols = np.where(occupied.sum(axis=0) >= column_threshold)[0]
        active_rows = np.where(occupied.sum(axis=1) >= row_threshold)[0]

        if len(active_cols) == 0 or len(active_rows) == 0:
            basic_box = mask.getbbox()
            if basic_box is None:
                return None
            return basic_box

        active_cols = self._largest_index_run(active_cols)

        return (
            int(active_cols[0]),
            int(active_rows[0]),
            int(active_cols[-1]) + 1,
            int(active_rows[-1]) + 1,
        )

    def _largest_index_run(self, indices):
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        return max(runs, key=len)

    def _get_letter_type(self, character):
        """Determine if a letter is an ascender, descender, or regular."""
        ascenders = set("bdfhklt")
        descenders = set("gjpqy")
        
        if character in ascenders:
            return "ascender"
        elif character in descenders:
            return "descender"
        else:
            return "regular"

    def _glyph_scale(self, character, glyph, target_height):

        ink_height = getattr(glyph, "ink_height", glyph.height)

        scale = target_height / max(1, ink_height)

        return scale * self.GLYPH_SCALE_OVERRIDES.get(character, 1.0)

    def _glyph_width(self, character, target_height):
        if character == " ":
            return 17

        if character == "\t":
            return 36

        glyph = self.glyphs.get(character)
        if glyph is None:
            bbox = self.font.getbbox(character)
            return max(12, bbox[2] - bbox[0])

        scale = self._glyph_scale(character, glyph, target_height)
        resized_width = int(glyph.width * scale)

        tightness = 0.82

        visible_estimate = resized_width * 0.88
        return int(visible_estimate)  # to increase right margin

    def _wrap_text_with_glyphs(self, text, max_width, target_height):
        wrapped_lines = []

        for paragraph in text.strip().splitlines():
            words = paragraph.split(" ")
            if not words:
                wrapped_lines.append("")
                continue

            line = ""
            line_width = 0

            for word in words:
                word_width = sum(
                    self._glyph_width(character, target_height)
                    for character in word
                )
                space_width = self._glyph_width(" ", target_height)

                if not line:
                    line = word
                    line_width = word_width
                    continue

                if line_width + space_width + word_width <= max_width:
                    line = f"{line} {word}"
                    line_width += space_width + word_width
                else:
                    wrapped_lines.append(line)
                    line = word
                    line_width = word_width

            if line:
                wrapped_lines.append(line)

        return wrapped_lines

    def _draw_fallback_character(self, image, character, x, baseline_y, text_color):
        if character in {"“", "”"}:
            character = '"'
        elif character in {"‘", "’"}:
            character = "'"
        elif character in {"–", "—"}:
            character = "-"

        text_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        bbox = self.font.getbbox(character)
        y = baseline_y - bbox[3] + 11
        # Lower fallback `f` a bit so its crossbar sits nearer the midline
        if character == "f":
            y += max(3, int(bbox[3] * 0.14))

        text_draw.text(
            (x, y),
            character,
            font=self.font,
            fill=(*text_color, self.random.randint(225, 235)),
        )
        return Image.alpha_composite(image.convert("RGBA"), text_layer)

    def _draw_font_line(self, image, line, x, baseline_y, text_color):
        text_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        bbox = self.font.getbbox(line)
        y = baseline_y - bbox[3] + 11

        text_draw.text(
            (x + self.random.randint(-2, 2), y + self.random.randint(0, 1)),
            line,
            font=self.font,
            fill=(*text_color, self.random.randint(220, 238)),
        )

        text_layer = text_layer.filter(
            ImageFilter.GaussianBlur(radius=0.12)
        )
        return Image.alpha_composite(image.convert("RGBA"), text_layer)

    def _draw_glyph_line(self, image, line, x, baseline_y, target_height, text_color):
        for character in line:
            if character == " ":
                # Wider spaces between words (user requested a bit more)
                space_width = int(target_height * 0.66)
                x += self.random.randint(
                    space_width - 1,
                    space_width + 6
                )
                continue

            if character in {".", ",", "'", '"', "!", "?"}:
                draw = ImageDraw.Draw(image)
                dot_y = baseline_y + self.random.randint(1, 4)
                if character == ".":
                    x += 4  # More space before period
                    draw.ellipse([(x, baseline_y - 7), (x + 3, baseline_y - 4)], fill=text_color)
                    x += 5  # More space after period
                    continue
                if character == ",":
                    x += 4  # More space before comma
                    draw.ellipse([(x, baseline_y - 2), (x + 2, baseline_y + 1)], fill=text_color)
                    draw.line([(x + 1, baseline_y + 1), (x - 1, baseline_y + 5)], fill=text_color, width=1)
                    x += 5  # More space after comma
                    continue
                if character == "'":
                    draw.line([(x + 2, baseline_y - 28), (x, baseline_y - 18)], fill=text_color, width=2)
                    x += 6
                    continue
                if character == '"':
                    draw.line([(x + 1, baseline_y - 29), (x, baseline_y - 19)], fill=text_color, width=2)
                    draw.line([(x + 7, baseline_y - 29), (x + 6, baseline_y - 19)], fill=text_color, width=2)
                    x += 15
                    continue
                if character == "!":
                    draw.line([(x + 3, baseline_y - 30), (x + 2, baseline_y - 8)], fill=text_color, width=2)
                    draw.ellipse([(x + 1, dot_y), (x + 4, dot_y + 3)], fill=text_color)
                    x += 12
                    continue
                if character == "?":
                    image = self._draw_fallback_character(image, character, x, baseline_y, text_color)
                    x += 12
                    continue

            if character == "_":
                x += 6  # More space before underscore
                draw = ImageDraw.Draw(image)
                y = baseline_y - 8
                draw.line([(x, y), (x + 16, y)], fill=text_color, width=1)
                x += 16
                continue

            if character == "-":
                # Add a small gap before and after hyphen
                x += 2
                draw = ImageDraw.Draw(image)
                y = baseline_y - self.random.randint(10, 13)
                draw.line([(x, y), (x + 14, y)], fill=text_color, width=2)
                x += 20
                continue

            glyph = self.glyphs.get(character)
            
            # Special positioning for punctuation glyphs (colon, semicolon)
            if character in {":", ";"}:
                x += 2  # Space before punctuation
                glyph = self.glyphs.get(character)
                if glyph:
                    scale = self._glyph_scale(character, glyph, target_height)
                    scale *= self.random.uniform(0.985, 1.015)
                    width = max(1, int(glyph.width * scale))
                    height = max(1, int(glyph.height * scale))
                    glyph_image = glyph.resize((width, height), Image.Resampling.LANCZOS)
                    
                    alpha_factor = self.random.uniform(0.95, 1.00)
                    alpha = glyph_image.getchannel("A").point(
                        lambda value: max(0, min(255, int(value * alpha_factor)))
                    )
                    tinted = Image.new("RGBA", glyph_image.size, (*text_color, 0))
                    tinted.putalpha(alpha)
                    
                    angle = self.random.uniform(-0.35, 0.35)
                    tinted = tinted.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                    tinted = tinted.filter(ImageFilter.GaussianBlur(radius=self.random.uniform(0.0, 0.12)))
                    
                    # Position punctuation above baseline
                    y = baseline_y - 18 + self.random.randint(1, 2)
                    
                    image.alpha_composite(tinted, (int(x), int(y)))
                    x += width - 1
                    continue
            
            if glyph is None:
                image = self._draw_fallback_character(image, character, x, baseline_y, text_color)
                x += self._glyph_width(character, target_height) + self.random.randint(-2, 1)
                continue

            scale = self._glyph_scale(character, glyph, target_height)
            scale *= self.random.uniform(0.985, 1.015)
            width = max(1, int(glyph.width * scale))
            height = max(1, int(glyph.height * scale))

            glyph_image = glyph.resize((width, height), Image.Resampling.LANCZOS)

            # Store height BEFORE rotation for proper positioning
            pre_rotation_height = height
            
            alpha_factor = self.random.uniform(0.95, 1.00)
            alpha = glyph_image.getchannel("A").point(
                lambda value: max(0, min(255, int(value * alpha_factor)))
            )
            tinted = Image.new("RGBA", glyph_image.size, (*text_color, 0))
            tinted.putalpha(alpha)

            angle = self.random.uniform(-0.35, 0.35)
            tinted = tinted.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            tinted = tinted.filter(ImageFilter.GaussianBlur(radius=self.random.uniform(0.0, 0.12)))


            # -----------------------------------
            # Real handwriting baseline alignment
            # -----------------------------------

            scaled_ink_bottom = int(glyph.ink_bottom * scale)

            y = baseline_y - scaled_ink_bottom

            if character in "gjpqyf":
                y += int(pre_rotation_height * 0.30) + self.random.randint(0, 3)

            # Small handwriting jitter
            baseline_offset = self.random.choice([0, 0, 1, -1])
            y += baseline_offset

            # Draw glyph
            image.alpha_composite(tinted, (int(x), int(y)))

            # Natural character spacing
            visible_width = self._get_visible_width(tinted)

            base_advance = visible_width * 0.88

            kerning_jitter = self.random.uniform(-1.2, 1.0)

            x += int(base_advance + kerning_jitter) # letter spacing

        return image

    def _get_template(self, template):
        if template == "random":
            template = self.random.choice(list(self.templates.keys()))

        if template not in self.templates:
            valid_templates = ", ".join(sorted(self.templates.keys()))
            raise ValueError(f"Unknown template '{template}'. Use: {valid_templates}, random")

        return self.templates[template]

    def _make_paper(self, width, height, page_template):
        paper_color = page_template["paper"]
        grain_strength = page_template["grain"]
        image = Image.new("RGB", (width, height), color=paper_color)
        pixels = image.load()

        for y in range(height):
            vertical_tint = int((y / height) * 3)
            for x in range(width):
                grain = self.random.randint(-grain_strength, grain_strength)
                warm = self.random.randint(-2, 2)
                r = max(0, min(255, paper_color[0] + grain + warm - vertical_tint))
                g = max(0, min(255, paper_color[1] + grain - vertical_tint))
                b = max(0, min(255, paper_color[2] + grain - warm - vertical_tint))
                pixels[x, y] = (r, g, b)

        return image.filter(ImageFilter.GaussianBlur(radius=0.35))

    def _add_page_depth(self, image, page_template):
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size

        for x in range(42):
            alpha = max(0, 30 - x)
            draw.line([(x, 0), (x, height)], fill=(80, 70, 55, alpha))

        for y in range(38):
            alpha = max(0, 18 - y // 2)
            draw.line([(0, y), (width, y)], fill=(255, 255, 255, alpha))

        if page_template["paper"][0] < 245:
            draw.rectangle(
                [(0, 0), (width - 1, height - 1)],
                outline=(205, 190, 160, 60),
                width=2,
            )

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _draw_holes(self, draw, height):
        hole_x = 42
        for y in [230, 800, 1370]:
            draw.ellipse(
                [(hole_x - 16, y - 16), (hole_x + 16, y + 16)],
                fill=(226, 223, 212),
                outline=(185, 181, 170),
                width=2,
            )
            draw.arc(
                [(hole_x - 13, y - 13), (hole_x + 13, y + 13)],
                210,
                35,
                fill=(120, 115, 105),
                width=1,
            )

    def _draw_top_rule(self, draw, width):
        draw.line([(58, 82), (width - 58, 82)], fill=(205, 215, 225), width=1)
        draw.line([(58, 92), (width - 58, 92)], fill=(224, 230, 236), width=1)

    def _draw_ruled_page(self, draw, width, height, page_template):
        line_color = page_template["line_color"]
        shadow_line_color = page_template["line_shadow"]
        margin_color = page_template["margin_color"]
        margin_line_x = page_template["margin_x"]
        first_line_y = page_template["first_line_y"]
        line_gap = page_template["line_gap"]
        left_edge = page_template["left_edge"]
        right_edge = page_template["right_edge"]

        y = first_line_y
        while y < height - 65:
            wobble = self.random.choice([-1, 0, 0, 1])
            x_start = left_edge + self.random.choice([-1, 0, 0, 1])
            x_end = width - right_edge + self.random.choice([-1, 0, 0, 1])
            draw.line(
                [(x_start, y + wobble), (x_end, y + wobble)],
                fill=shadow_line_color,
                width=2,
            )
            draw.line(
                [(x_start, y), (x_end, y)],
                fill=line_color,
                width=1,
            )
            y += line_gap

        draw.line(
            [(margin_line_x - 1, 50), (margin_line_x - 1, height - 55)],
            fill=(248, 205, 200),
            width=1,
        )
        draw.line(
            [(margin_line_x, 50), (margin_line_x, height - 55)],
            fill=margin_color,
            width=2,
        )

        if page_template["holes"]:
            self._draw_holes(draw, height)

        if page_template["top_rule"]:
            self._draw_top_rule(draw, width)

    def _wrap_text(self, text, draw, max_width):
        wrapped_lines = []

        for paragraph in text.strip().splitlines():
            words = paragraph.split()
            if not words:
                wrapped_lines.append("")
                continue

            line = ""
            for word in words:
                candidate = word if not line else f"{line} {word}"
                bbox = draw.textbbox((0, 0), candidate, font=self.font)

                if bbox[2] - bbox[0] <= max_width:
                    line = candidate
                else:
                    if line:
                        wrapped_lines.append(line)
                    line = word

            if line:
                wrapped_lines.append(line)

        return wrapped_lines

    def _summary_from_file(self, path):
        content = path.read_text(encoding="utf-8").strip()
        marker = "===== SUMMARY ====="

        if marker in content:
            summary = content.split(marker, 1)[1].strip()
        else:
            summary = content

        summary = " ".join(summary.split())
        summary = (
            summary.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
        )

        if len(summary) <= 260:
            return summary

        cutoff = summary.rfind(".", 0, 260)
        if cutoff == -1:
            cutoff = summary.rfind(" ", 0, 260)
        if cutoff == -1:
            cutoff = 260

        return summary[:cutoff + 1].strip()

    # def summaries_from_directory(self, input_dir="data/output", summaries_template="default"):
    #     """
    #     Load summaries from directory with customizable format.
        
    #     Templates:
    #       - 'default': 'Image name: summary' (newline between entries)
    #       - 'compact': 'Image name: summary' (no blank lines)
    #       - 'labeled': '[Image] Image name [Summary] summary'
    #       - 'custom': Pass your own format string with {image_name} and {summary} placeholders
    #     """
    #     # Predefined templates
    #     templates = {
    #         "default": "{image_name}: {summary}",
    #         "compact": "{image_name}: {summary}",
    #         "labeled": "[Image] {image_name}\n[Summary] {summary}",
    #     }
        
    #     # Get the template format string
    #     if summaries_template in templates:
    #         format_str = templates[summaries_template]
    #     else:
    #         # Treat it as a custom format string
    #         format_str = summaries_template
        
    #     # Determine separator based on template
    #     if summaries_template == "default":
    #         separator = "\n\n"  # Blank line between entries
    #     else:
    #         separator = "\n"   # Just newline for others
        
    #     input_path = Path(input_dir)
    #     summaries = []

    #     for path in sorted(input_path.glob("*.txt")):
    #         summary = self._summary_from_file(path)
    #         formatted = format_str.format(image_name=path.name, summary=summary)
    #         summaries.append(formatted)

    #     return separator.join(summaries)

    def render_summaries(
        self,
        input_dir="data/output",
        output_path="handwritten_output.png",
        template="random",
        handwriting_source="glyphs",
        layout_template="default",
        custom_title=None,
    ):

        blocks = LayoutEngine.build(
            input_dir=input_dir,
            template=layout_template,
            custom_title=custom_title,
        )
        

        return self.render_layout_blocks(
            blocks,
            output_path=output_path,
            template=template,
            handwriting_source=handwriting_source,
        )

    def render_layout_blocks(
        self,
        blocks,
        output_path="output.png",
        template="random",
        handwriting_source="glyphs",
    ):

        width = 1200
        height = 1600

        text_color = (31, 34, 38)

        page_template = self._get_template(template)

        margin_left = page_template["margin_x"] + 18

        line_gap = page_template["line_gap"]

        baseline_nudge = page_template["baseline_nudge"]

        def make_page():
            page = self._make_paper(width, height, page_template)

            page = self._add_page_depth(page, page_template)

            page_draw = ImageDraw.Draw(page)

            self._draw_ruled_page(
                page_draw,
                width,
                height,
                page_template,
            )

            return page

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page_paths = []

        image = make_page()

        max_text_width = width - margin_left - page_template["right_edge"]

        target_glyph_height = min(26, line_gap - 24)
        use_glyphs = handwriting_source == "glyphs" and self.glyphs

        title_zone_y = 55

        first_line_y = page_template["first_line_y"]

        last_line_y = first_line_y

        while last_line_y + line_gap < height - 65:
            last_line_y += line_gap

        line_y = first_line_y

        page_number = 1

        def filtered_page(page):
            return page.filter(
                ImageFilter.UnsharpMask(
                    radius=1.2,
                    percent=65,
                    threshold=3,
                )
            )

        def save_current_page():
            path = output_path

            if page_number > 1:
                path = output_path.with_name(
                    f"{output_path.stem}_page_{page_number}{output_path.suffix}"
                )

            filtered_page(image).save(path)
            page_paths.append(path)
            print(f"[SAVED] {path}")

        def ensure_ruled_line():
            nonlocal image, line_y, page_number

            if line_y <= last_line_y:
                return

            save_current_page()

            page_number += 1
            image = make_page()
            line_y = first_line_y

        def ensure_ruled_lines(count):
            nonlocal image, line_y, page_number

            if line_y + ((count - 1) * line_gap) <= last_line_y:
                return

            save_current_page()

            page_number += 1
            image = make_page()
            line_y = first_line_y

        for index, block in enumerate(blocks):

            block_type = block["type"]

            block_text = block["text"]

            # ---------------------------------
            # TITLE
            # ---------------------------------

            if block_type == "title":

                if use_glyphs:
                    wrapped = self._wrap_text_with_glyphs(
                        block_text,
                        max_text_width,
                        target_glyph_height + 8,
                    )
                else:
                    wrapped = self._wrap_text(
                        block_text,
                        ImageDraw.Draw(image),
                        max_text_width,
                    )

                current_title_y = title_zone_y

                for line in wrapped:

                    if use_glyphs:
                        line_width = sum(
                            self._glyph_width(c, target_glyph_height + 8)
                            for c in line
                        )
                    else:
                        bbox = self.font.getbbox(line)
                        line_width = bbox[2] - bbox[0]

                    center_x = (width - line_width) // 2

                    if use_glyphs:
                        image = self._draw_glyph_line(
                            image.convert("RGBA"),
                            line,
                            center_x,
                            current_title_y,
                            target_glyph_height + 8,
                            text_color,
                        ).convert("RGB")
                    else:
                        image = self._draw_font_line(
                            image.convert("RGBA"),
                            line,
                            center_x,
                            current_title_y,
                            text_color,
                        ).convert("RGB")

                    current_title_y += line_gap

                # PERFECT ALIGNMENT START
                line_y = first_line_y

            # ---------------------------------
            # HEADING
            # ---------------------------------

            elif block_type == "heading":

                next_block = blocks[index + 1] if index + 1 < len(blocks) else None

                if next_block and next_block["type"] == "paragraph":
                    ensure_ruled_lines(2)
                else:
                    ensure_ruled_line()

                if use_glyphs:
                    image = self._draw_glyph_line(
                        image.convert("RGBA"),
                        block_text,
                        margin_left,
                        line_y + baseline_nudge,
                        target_glyph_height + 3,
                        text_color,
                    ).convert("RGB")
                else:
                    image = self._draw_font_line(
                        image.convert("RGBA"),
                        block_text,
                        margin_left,
                        line_y + baseline_nudge,
                        text_color,
                    ).convert("RGB")

                line_y += line_gap

            # ---------------------------------
            # PARAGRAPH
            # ---------------------------------

            elif block_type == "paragraph":

                if use_glyphs:
                    wrapped = self._wrap_text_with_glyphs(
                        block_text,
                        max_text_width,
                        target_glyph_height,
                    )
                else:
                    wrapped = self._wrap_text(
                        block_text,
                        ImageDraw.Draw(image),
                        max_text_width,
                    )

                for line in wrapped:

                    ensure_ruled_line()

                    if use_glyphs:
                        image = self._draw_glyph_line(
                            image.convert("RGBA"),
                            line,
                            margin_left,
                            line_y + baseline_nudge,
                            target_glyph_height,
                            text_color,
                        ).convert("RGB")
                    else:
                        image = self._draw_font_line(
                            image.convert("RGBA"),
                            line,
                            margin_left,
                            line_y + baseline_nudge,
                            text_color,
                        ).convert("RGB")

                    line_y += line_gap

            # ---------------------------------
            # BLANK
            # ---------------------------------

            elif block_type == "blank":

                if line_y < last_line_y:
                    line_y += line_gap
                else:
                    line_y = last_line_y + line_gap

        save_current_page()

        return page_paths


    # def render_text(
    #     self,
    #     text,
    #     output_path="output.png",
    #     template="random",
    #     handwriting_source="glyphs",
    # ):

    #     # -------------------------
    #     # PAGE SETTINGS
    #     # -------------------------

    #     width = 1200
    #     height = 1600

    #     text_color = (31, 34, 38)
    #     page_template = self._get_template(template)
    #     margin_left = page_template["margin_x"] + 18
    #     first_line_y = page_template["first_line_y"]
    #     line_gap = page_template["line_gap"]
    #     baseline_nudge = page_template["baseline_nudge"]

    #     image = self._make_paper(width, height, page_template)
    #     image = self._add_page_depth(image, page_template)
    #     draw = ImageDraw.Draw(image)

    #     # -------------------------
    #     # DRAW NOTEBOOK PAGE
    #     # -------------------------

    #     self._draw_ruled_page(
    #         draw,
    #         width,
    #         height,
    #         page_template,
    #     )

    #     # -------------------------
    #     # WRAP TEXT
    #     # -------------------------

    #     max_text_width = width - margin_left - page_template["right_edge"]
    #     target_glyph_height = min(26, line_gap - 24)

    #     if handwriting_source == "glyphs" and self.glyphs:
    #         wrapped_lines = self._wrap_text_with_glyphs(
    #             text,
    #             max_text_width,
    #             target_glyph_height,
    #         )
    #     else:
    #         wrapped_lines = self._wrap_text(text, draw, max_text_width)

    #     # -------------------------
    #     # DRAW TEXT
    #     # -------------------------

    #     line_y = first_line_y

    #     for line in wrapped_lines:
    #         if handwriting_source == "glyphs" and self.glyphs:
    #             image = self._draw_glyph_line(
    #                 image.convert("RGBA"),
    #                 line,
    #                 margin_left + self.random.randint(-4, 5),
    #                 line_y + baseline_nudge,
    #                 target_glyph_height,
    #                 text_color,
    #             ).convert("RGB")
    #             draw = ImageDraw.Draw(image)
    #         else:
    #             x_jitter = self.random.randint(-4, 5)
    #             y_jitter = self.random.randint(0, 2)
    #             text_bbox = self.font.getbbox(line)
    #             text_y = line_y - text_bbox[3] + baseline_nudge + y_jitter

    #             text_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    #             text_draw = ImageDraw.Draw(text_layer)
    #             text_draw.text(
    #                 (margin_left + x_jitter, text_y),
    #                 line,
    #                 font=self.font,
    #                 fill=(*text_color, self.random.randint(220, 240)),
    #             )

    #             text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.12))
    #             image = Image.alpha_composite(image.convert("RGBA"), text_layer).convert("RGB")
    #             draw = ImageDraw.Draw(image)

    #         line_y += line_gap

    #     # -------------------------
    #     # SAVE
    #     # -------------------------

    #     image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=65, threshold=3))
    #     image.save(output_path)

    #     print(f"[SAVED] {output_path}")

