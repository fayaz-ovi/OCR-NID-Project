"""
image_preprocessor.py
=====================
Tesseract-optimised preprocessing pipeline for Bangladesh NID card photos.

Why Tesseract needs a completely different pipeline than EasyOCR
----------------------------------------------------------------
EasyOCR is a neural network that tolerates noise, gradients, and partial
binarisation.  Tesseract is a classical engine that expects:
  • Clean black-ink-on-white binary image
  • ≥ 300 DPI resolution (≥ 1600 px wide for a standard NID card)
  • Uniform background (no shadows, no gradients)
  • Sharp text edges (no blur)
  • White padding on all sides (prevents edge-character clipping)
  • NO dilation  (causes Bangla conjunct characters to merge)

Pipeline (Tesseract-specific)
------------------------------
  1.  Load
  2.  Alpha → RGB
  3.  Downsample if > 20 MP
  4.  Minimum size guard
  5.  Crop to card (perspective warp)   – removes background/hand
  6.  Grayscale
  7.  Skew correction                   – fixed rotation sign
  8.  Upscale to ≥ 1600 px             – EARLIER than before; before filters
  9.  Unsharp mask (sharpen)            – NEW; crisp edges for Tesseract
 10.  De-shadow (morphological top-hat) – NEW; flattens lighting gradient
 11.  Otsu global threshold             – replaces adaptive; no noise dots
 12.  White border padding              – NEW; prevents edge clipping
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np

try:
    from .exceptions import PreprocessingError  # type: ignore[import]
except ImportError:
    class PreprocessingError(Exception):        # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_PIXEL_COUNT: int   = 20_000_000
_MIN_SIDE_PX:     int   = 200
_TARGET_MIN_W:    int   = 1600   # wider target for Tesseract 300 DPI
_TARGET_MIN_H:    int   = 1000
_MAX_SKEW_DEG:    float = 45.0
_BORDER_PX:       int   = 30     # white padding around processed image

_CARD_ASPECT_MIN: float = 1.3
_CARD_ASPECT_MAX: float = 2.2
_CARD_AREA_FRAC:  float = 0.15


class NIDImagePreprocessor:
    """
    Prepares a raw NID card photograph for pytesseract OCR.

    Usage::

        pre = NIDImagePreprocessor()
        img = pre.preprocess("/path/to/nid.jpg")
        # img is a grayscale binary uint8 ndarray, ready for pytesseract
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image_path: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """Full pipeline. Returns a binary (0/255) single-channel ndarray."""
        img    = self._load(image_path)
        img    = self._handle_alpha(img)
        img    = self._downsample_if_oversized(img)
        self._assert_minimum_size(img)
        img    = self._crop_to_card(img)
        gray   = self._to_grayscale(img)
        gray   = self._correct_skew(gray)
        gray   = self._upscale_for_tesseract(gray)  # early – before filters
        gray   = self._unsharp_mask(gray)            # sharpen edges
        gray   = self._remove_shadow(gray)           # flatten background
        binary = self._otsu_threshold(gray)          # clean global binarise
        binary = self._add_border(binary)            # prevent edge clipping
        return binary

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def _load(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            logger.debug("_load: ndarray received directly")
            return source
        path = Path(source)
        if not path.exists():
            raise PreprocessingError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise PreprocessingError(
                f"OpenCV could not decode {path}. "
                "File may be corrupted or unsupported."
            )
        logger.debug("_load: %s  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
        return img

    # ------------------------------------------------------------------
    # Step 2 — Alpha channel
    # ------------------------------------------------------------------

    def _handle_alpha(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2 or img.shape[2] == 3:
            return img
        if img.shape[2] == 4:
            bgr   = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            white = np.ones_like(bgr) * 255.0
            return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        if img.shape[2] == 2:
            return img[:, :, 0]
        return img

    # ------------------------------------------------------------------
    # Step 3 — Downsample
    # ------------------------------------------------------------------

    def _downsample_if_oversized(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h * w <= _MAX_PIXEL_COUNT:
            return img
        scale  = math.sqrt(_MAX_PIXEL_COUNT / (h * w))
        new_w  = max(1, int(w * scale))
        new_h  = max(1, int(h * scale))
        logger.info("_downsample: %dx%d → %dx%d (%.2f×)", w, h, new_w, new_h, scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Step 4 — Minimum size guard
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_minimum_size(img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if h < _MIN_SIDE_PX or w < _MIN_SIDE_PX:
            raise PreprocessingError(
                f"Image too small: {w}×{h} px "
                f"(minimum {_MIN_SIDE_PX}×{_MIN_SIDE_PX} px)."
            )

    # ------------------------------------------------------------------
    # Step 5 — Crop to card (perspective warp)
    # ------------------------------------------------------------------

    def _crop_to_card(self, img: np.ndarray) -> np.ndarray:
        """
        Detect the NID card rectangle and apply a perspective warp.

        Designed for fayaz_nid_.jpg conditions: dark/textured background,
        hand partially visible, card ~70 % of frame area.
        Falls back to the full image if no suitable contour is found —
        this step must NEVER raise an exception that blocks the pipeline.
        """
        try:
            h, w  = img.shape[:2]
            gray  = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Gentle blur to suppress JPEG noise before edge detection
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 180)

            # Dilate edges to close small gaps in the card outline
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=2)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return img

            image_area = h * w
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                if cv2.contourArea(cnt) < image_area * _CARD_AREA_FRAC:
                    break
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    rect_r  = cv2.minAreaRect(approx)
                    rw, rh  = sorted(rect_r[1])
                    if rh == 0:
                        continue
                    aspect  = rw / rh if rw > rh else rh / rw
                    if _CARD_ASPECT_MIN <= aspect <= _CARD_ASPECT_MAX:
                        warped = self._four_point_warp(img, approx.reshape(4, 2))
                        logger.info("_crop_to_card: detected (aspect=%.2f)", aspect)
                        return warped

            logger.debug("_crop_to_card: no card rectangle found, using full image")
        except Exception as exc:
            logger.warning("_crop_to_card: error '%s', skipping crop", exc)

        return img

    @staticmethod
    def _four_point_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Perspective-warp *img* using the four corner points *pts*."""
        rect    = np.zeros((4, 2), dtype="float32")
        s       = pts.sum(axis=1)
        d       = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        rect[1] = pts[np.argmin(d)]   # top-right
        rect[3] = pts[np.argmax(d)]   # bottom-left

        tl, tr, br, bl = rect
        max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (max_w, max_h))

    # ------------------------------------------------------------------
    # Step 6 — Grayscale
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Step 7 — Skew correction
    # ------------------------------------------------------------------

    def _correct_skew(self, gray: np.ndarray) -> np.ndarray:
        """Detect and correct document skew via Hough lines (sign-fixed)."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=gray.shape[1] // 4, maxLineGap=20,
        )
        if lines is None:
            return gray

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            ang = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            if abs(ang) <= _MAX_SKEW_DEG:
                angles.append(ang)

        if not angles:
            return gray

        median = float(np.median(angles))
        if abs(median) < 1.0:
            return gray   # negligible skew – skip interpolation

        h, w    = gray.shape
        M       = cv2.getRotationMatrix2D((w // 2, h // 2), -median, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info("_correct_skew: corrected %.2f°", median)
        return rotated

    # ------------------------------------------------------------------
    # Step 8 — Upscale for Tesseract
    # ------------------------------------------------------------------

    @staticmethod
    def _upscale_for_tesseract(img: np.ndarray) -> np.ndarray:
        """
        Ensure the image is at least 1600 × 1000 px.

        Tesseract requires ~300 DPI for accurate Bangla Unicode output.
        A standard NID card is ~85.6 mm wide; at 300 DPI that is ≈1009 px.
        Most phone photos are in the 900–1200 px range, so a 1.5–2× upscale
        is typically needed.

        INTER_LANCZOS4 gives the sharpest result for upscaling text images.
        """
        h, w = img.shape[:2]
        if w >= _TARGET_MIN_W and h >= _TARGET_MIN_H:
            return img

        scale  = max(_TARGET_MIN_W / w, _TARGET_MIN_H / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        logger.info("_upscale_for_tesseract: %dx%d → %dx%d (%.2f×)", w, h, new_w, new_h, scale)
        return result

    # ------------------------------------------------------------------
    # Step 9 (NEW) — Unsharp mask
    # ------------------------------------------------------------------

    @staticmethod
    def _unsharp_mask(gray: np.ndarray) -> np.ndarray:
        """
        Sharpen text edges with an unsharp mask.

        Phone photos are softened by lens optics and JPEG compression.
        Sharpening before binarisation makes text boundaries crisp so
        Tesseract's LSTM edge-detection step finds clean character outlines.

        Parameters chosen empirically on Bangladesh NID card photos:
          amount  = 1.5  (moderate sharpening, not aggressive)
          sigma   = 2.0  (picks up text-stroke-width blur)
        """
        blurred   = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        return sharpened

    # ------------------------------------------------------------------
    # Step 10 (NEW) — Remove shadow / background gradient
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_shadow(gray: np.ndarray) -> np.ndarray:
        """
        Normalise uneven lighting using a morphological background estimate.

        This solves the specific problem in phone-photo scans where the hand
        holding the card creates a shadow gradient, causing Otsu to choose a
        threshold that leaves regions too dark or over-exposed.

        Method (morphological normalisation):
          1. Dilate the image with a large (45 × 45) ellipse kernel.
             Because MORPH_DILATE computes the **maximum** over each
             neighbourhood, and text pixels are dark (low value), the kernel
             slides over character strokes and the output is dominated by the
             surrounding bright paper — producing a smooth estimate of the
             background illumination.
          2. diff = background − gray.
             • Bright-paper pixels → diff ≈ 0   (background ≈ pixel)
             • Dark-ink pixels    → diff ≈ high (background >> pixel)
          3. Normalise diff to 0–255 so the result has full contrast.

        After this step:
          • diff is HIGH where there is dark ink / dark elements.
          • diff is LOW  where there is white paper / bright background.
        The subsequent _otsu_threshold (THRESH_BINARY_INV) maps
          HIGH diff (ink)  → 0   (black ink)
          LOW  diff (paper)→ 255 (white paper)
        yielding the conventional black-text-on-white-background output
        that Tesseract expects.
        """
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        diff       = cv2.subtract(background, gray)
        normalised = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return normalised

    # ------------------------------------------------------------------
    # Step 11 — Otsu global threshold
    # ------------------------------------------------------------------

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Binarise with Otsu's automatic global threshold.

        After _remove_shadow the pixel values encode ink-presence:
          • HIGH value (≈200+) → dark ink / text
          • LOW  value (≈0)    → bright paper / background

        Using THRESH_BINARY_INV maps:
          pixel > threshold → 0   (black ink)   — ink detected
          pixel ≤ threshold → 255 (white paper) — background

        This gives the standard Tesseract input format:
        **black text on white background**.

        WHY NOT THRESH_BINARY (old behaviour)
        --------------------------------------
        THRESH_BINARY would give white-ink-on-black-background (inverted).
        While Tesseract 4 LSTM can auto-detect polarity, the standard
        black-on-white format is what the Bangla training data was produced
        from, and it avoids the engine wasting time on polarity detection.

        WHY NOT adaptive threshold
        --------------------------
        After de-shadowing there is no reason for local thresholding, and
        adaptive thresholding introduces noise around Bangla matras and
        dots (ো, ু, ূ) which Tesseract's LSTM misreads as extra characters.
        """
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        return binary

    # ------------------------------------------------------------------
    # Step 12 (NEW) — White border padding
    # ------------------------------------------------------------------

    @staticmethod
    def _add_border(img: np.ndarray) -> np.ndarray:
        """
        Add a 30 px white border on all four sides.

        Tesseract's layout analysis engine shrinks its internal text-block
        detection region inward from the image edges.  Without padding,
        characters that appear near the card boundary (especially the NID
        number on the right side, or the last letter of long names) are
        silently dropped from the output.
        """
        return cv2.copyMakeBorder(
            img,
            top=_BORDER_PX, bottom=_BORDER_PX,
            left=_BORDER_PX, right=_BORDER_PX,
            borderType=cv2.BORDER_CONSTANT,
            value=255,   # white fill
        )
