"""Unit tests: text and image preprocessing (data pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data_pipeline.preprocessing.image_processing import resize_pad_rgb
from data_pipeline.preprocessing.text_processing import (
    clean_and_normalize_text,
    extract_visible_text_from_html,
)


class TestCleanAndNormalizeText:
    def test_empty(self) -> None:
        assert clean_and_normalize_text("") == ""
        assert clean_and_normalize_text("   ") == ""

    def test_collapses_whitespace(self) -> None:
        assert clean_and_normalize_text("hello   world\n\t") == "hello world"

    def test_strips_control_chars(self) -> None:
        t = clean_and_normalize_text("a\x00b\rc")
        assert "\x00" not in t
        assert "a" in t and "b" in t and "c" in t


class TestExtractVisibleTextFromHtml:
    def test_strips_script(self) -> None:
        html = "<html><script>evil()</script><body><p>Hi</p></body></html>"
        out = extract_visible_text_from_html(html)
        assert "evil" not in out
        assert "Hi" in out

    def test_empty_html(self) -> None:
        assert extract_visible_text_from_html("") == ""


class TestResizePadRgb:
    def test_square_output_size(self) -> None:
        img = Image.new("RGB", (100, 50), color=(10, 20, 30))
        out = resize_pad_rgb(img, 336)
        assert out.size == (336, 336)
        assert out.mode == "RGB"

    def test_preserves_aspect_inside_canvas(self) -> None:
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        out = resize_pad_rgb(img, 100)
        arr = np.asarray(out)
        assert arr.shape == (100, 100, 3)
        assert arr[50, 50, 0] == 255
