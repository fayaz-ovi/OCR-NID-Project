"""
ocr_service.py
==============
Tesseract-primary OCR service for Bangladesh NID card images.

Architecture
------------
Tesseract is the PRIMARY engine.  EasyOCR is the FALLBACK (reverse of before).

Dual-pass Tesseract strategy
-----------------------------
Running Tesseract once with ``lang="ben+eng"`` on a mixed-script document
gives mediocre results because the LSTM engine tries to switch language models
mid-line.  Accuracy is significantly higher with two separate passes:

  Pass 1 — Bangla pass  (lang="ben", PSM 6)
    Captures Bangla names, father/mother names, header text.
    Returns word-level (bbox, text, confidence) tuples.

  Pass 2 — English pass (lang="eng", PSM 6)
    Captures English name, date of birth, NID number, blood group.
    Latin-script recognition is much more reliable without the Bengali
    language model competing.

  Merge step
    Results from both passes are combined.  For each word bbox we keep
    the pass that produced the higher confidence score.  Words that only
    appeared in one pass are kept as-is.

  Line reconstruction
    Tesseract's ``image_to_data`` output groups words by
    (block_num, par_num, line_num).  We respect this grouping to
    reconstruct text lines that preserve the original reading order —
    much more reliable than sorting by Y coordinate alone (which fails
    on multi-column layouts or when two text rows have similar Y centres).

Tesseract configuration strings used
--------------------------------------
  --oem 3   : Use LSTM engine (most accurate for Bangla Unicode)
  --psm 6   : Uniform block of text — best for structured form layouts
  --dpi 300 : Tell Tesseract the image is 300 DPI (matches our upscale)

Field parsing improvements for this card layout
-------------------------------------------------
All the FIX A–H improvements from the previous version are retained and
combined with Tesseract-specific robustness fixes:

  T1: Tesseract often splits "Date of Birth:" and "28 Feb 2003" across
      consecutive words on the same physical line.  The line reconstruction
      step joins them before regex matching.

  T2: Tesseract frequently OCRs "ID NO:" as "ID NO :" (space before colon)
      or "10 NO:" (digit-one confusion).  The NID regex now handles both.

  T3: Tesseract on Bangla sometimes outputs "নাম :" with a space before
      the colon.  The keyword patterns are loosened to allow optional
      spaces around the colon separator.

  T4: Tesseract's Bengali model outputs correct Unicode for standard
      conjuncts but may confuse ো/া vowel signs on low-res images.
      We strip low-confidence tokens (conf < 30/100 for Tesseract,
      which maps to 0.30) instead of the 0.05 floor used for EasyOCR.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------
try:
    from .exceptions import OCRServiceError, OCRTimeoutError, PreprocessingError  # type: ignore
    from .image_preprocessor import NIDImagePreprocessor                           # type: ignore
except ImportError:
    class PreprocessingError(Exception): pass   # type: ignore[no-redef]
    class OCRTimeoutError(Exception):    pass   # type: ignore[no-redef]
    class OCRServiceError(Exception):    pass   # type: ignore[no-redef]
    from image_preprocessor import NIDImagePreprocessor                             # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract configuration strings
# ---------------------------------------------------------------------------
# OEM 3 = LSTM engine; PSM 6 = uniform block of text; DPI 300 = matches upscale
_TESS_CONFIG_BN  = r"--oem 3 --psm 6 --dpi 300 -c preserve_interword_spaces=1"
_TESS_CONFIG_EN  = r"--oem 3 --psm 6 --dpi 300 -c preserve_interword_spaces=1"

# Minimum Tesseract per-word confidence to include in results (0–100 scale)
_MIN_TESS_CONF: int = 30

# Timeout for the OCR thread
_OCR_TIMEOUT_SECONDS: int = 90   # Tesseract is slower than EasyOCR

# ---------------------------------------------------------------------------
# Month normalisation map
# ---------------------------------------------------------------------------
_MONTH_MAP: dict[str, str] = {
    "1": "Jan",  "01": "Jan",  "2": "Feb",  "02": "Feb",
    "3": "Mar",  "03": "Mar",  "4": "Apr",  "04": "Apr",
    "5": "May",  "05": "May",  "6": "Jun",  "06": "Jun",
    "7": "Jul",  "07": "Jul",  "8": "Aug",  "08": "Aug",
    "9": "Sep",  "09": "Sep",  "10": "Oct", "11": "Nov", "12": "Dec",
    "jan": "Jan", "january": "Jan",   "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",     "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",      "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",    "sep": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",   "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Bangla Unicode block
_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")

# NID number
# T2 fix: handles "ID NO :", "10 NO:", "1D NO:", "ID-NO", "ID: 4663704700"
_NID_BARE_RE  = re.compile(r"\b(\d{17}|\d{10})\b")
_NID_LABEL_RE = re.compile(
    r"(?:NID|ID\s*[NO\.]+|[1I][D0]\s*[NO\.]+)\s*[:\s#]*(\d{9,17})",
    re.IGNORECASE,
)

# Date patterns  (T1 fix: allow space between date parts that Tesseract splits)
_DATE_WRITTEN_RE = re.compile(
    r"\b(\d{1,2})\s*"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?t?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s*(\d{4})\b",
    re.IGNORECASE,
)
_DATE_DMY_SEP_RE = re.compile(
    r"\b(\d{1,2})[/\-.](\d{1,2}|\w+)[/\-.](\d{4})\b"
)

# Blood group
_BLOOD_BARE_RE = re.compile(r"\b(AB|A|B|O)[+\-]\b", re.IGNORECASE)
_BLOOD_WORD_RE = re.compile(
    r"\b(AB|A|B|O)\s*(?:পজিটিভ|positive|নেগেটিভ|negative)\b",
    re.IGNORECASE,
)

# Keyword patterns  (T3 fix: optional spaces around colon)
# English name: "Name : FAYAZ" or "Name: FAYAZ" or just "Name"
# No ^ anchor — phone photos often have OCR noise before the keyword on the same line
_KW_NAME_EN   = re.compile(r"\bName\s*[:\.]?\s*", re.IGNORECASE)

# Bangla name: "নাম :" or "নাম:" or "নাম"
# Also handles common Tesseract OCR substitutions on phone photos:
#   ন→দ/ত/ল (consonant substitution) → "দাম:", "তাম:", "লাম:" all caught
_KW_NAME_BN   = re.compile(r"নাম\s*[:\.]?\s*|[নদতল][াি]ম\s*[:\.]?\s*")

# Father: "পিতা :" or "পিতা:" or common OCR variants পিওা/ফিতা on blurry scans
_KW_FATHER_BN = re.compile(
    r"পিতা\s*[:\.]?\s*|father\s*[:\.]?\s*|[পফব][িী]তা\s*[:\.]?\s*",
    re.IGNORECASE,
)
# Mother: "মাতা :" or "মাতা:" or common OCR variant "আাতা:" on phone photos
_KW_MOTHER_BN = re.compile(
    r"মাতা\s*[:\.]?\s*|mother\s*[:\.]?\s*|[মআ][াি][তট]া\s*[:\.]?\s*",
    re.IGNORECASE,
)
# Date of birth
_KW_DOB       = re.compile(
    r"জন্ম\s*তারিখ\s*[:\.]?\s*|date\s*of\s*birth\s*[:\.]?\s*|dob\s*[:\.]?\s*",
    re.IGNORECASE,
)
# Address
_KW_ADDRESS   = re.compile(r"ঠিকানা\s*[:\.]?\s*|address\s*[:\.]?\s*", re.IGNORECASE)

# Header lines to skip when searching for Bangla person name
_HEADER_RE = re.compile(
    r"সরকার|জাতীয়|পরিচয়|বাংলাদেশ|গণপ্রজাতন্ত্রী|national|government|republic",
    re.IGNORECASE,
)

# Latin header phrases to skip in English name fallback
_LATIN_HEADER_PHRASES = {
    "government", "national", "republic", "peoples", "people",
    "bangladesh", "card", "identity", "date", "birth",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_bangla(text: str) -> bool:
    return bool(_BANGLA_RE.search(text))


def _is_uppercase_name(text: str) -> bool:
    """Return True if text looks like a Bangladesh NID English name.

    All NID English names are printed in ALL CAPS (e.g. "FAYAZ BIN FARUK",
    "MD FARUK HOSSAIN").  Requirements:

    1. ≥ 75 % of alphabetic characters are uppercase (allows for occasional
       single-character OCR substitutions like "FAYAZ BiN FARUK").
    2. At least one word must be ≥ 4 characters long to reject very short
       garbage sequences like "OO A" or "AB C" found in noisy phone photos.

    This guards against the English-name fallback picking up mixed-case
    UI text or random uppercase noise from heavily distorted images.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    if sum(c.isupper() for c in alpha) / len(alpha) < 0.75:
        return False
    # Require at least one substantial word (≥4 chars) — rejects "OO A", "AB C"
    words = text.split()
    return any(len(w) >= 4 for w in words)


def _trim_to_uppercase_name(text: str) -> str:
    """
    Trim a candidate English name string to its leading ALL-CAPS token sequence.

    Bangladesh NID English names are always in ALL CAPS.  Tesseract sometimes
    appends lowercase OCR garbage tokens at the end of the name line, e.g.
    "FAYAZ BIN FARUK bf" — the "bf" is noise from adjacent card text.

    Strategy: scan tokens left-to-right; stop when:
    • the token contains a non-name character (like "=", "[", digits mixed
      with letters — catches OCR artefacts like "=F", "0F"), OR
    • more than 50 % of alphabetic characters are lowercase.

    Pure non-alpha tokens (digits like "2", punctuation like "|") are skipped
    rather than acting as stop words, so they do not break a valid name.

    Examples::
        "FAYAZ BIN FARUK bf"   → "FAYAZ BIN FARUK"
        "MD FARUK HOSSAIN 3"   → "MD FARUK HOSSAIN"   (digit skipped)
        "=F 0 FARUK HOSSAIN"   → ""                   (stops at "=F")
        "FAYAZ BiN FARUK"      → "FAYAZ BiN FARUK"    (OCR error tolerated)
        "FARUK HOSSAIN"        → "FARUK HOSSAIN"       (unchanged)
    """
    result: list[str] = []
    for token in text.split():
        alpha = [c for c in token if c.isalpha()]
        if not alpha:
            continue  # skip pure-digit / punctuation tokens
        # Stop at tokens that contain characters that cannot appear in a name
        if not all(_is_clean_name_char(c) for c in token):
            break
        # Stop at the first predominantly-lowercase word
        if sum(c.islower() for c in alpha) / len(alpha) > 0.5:
            break
        result.append(token)
    return " ".join(result)


def _is_clean_name_char(c: str) -> bool:
    """Return True if *c* is acceptable inside a personal name token.

    Covers:
    • All Unicode Letter categories  (Lu, Ll, Lt, Lm, Lo) — basic letters
    • All Unicode Mark categories    (Mn, Mc, Me)  — Bengali vowel signs,
      hasanta (্), nukta (়), Arabic diacritics, etc.
    • Specific punctuation used in names: hyphen, period, apostrophe,
      Bengali visarga-like mark (ঃ).
    """
    cat = unicodedata.category(c)
    return cat[0] in ('L', 'M') or c in ".-'ঃ"


def _strip_trailing_noise(text: str) -> str:
    """
    Remove trailing non-name tokens from an OCR-extracted **name** string.

    Tesseract sometimes appends noise characters after a valid name —
    for example ``"FAYAZ BIN FARUK =="`` or ``"মোঃ ফারুক হোসেন !"``
    or ``"চামেলী আক্তার ."`` (stray period from watermark).

    A token is considered *clean* (part of the name) if:
    • It contains **at least one** Unicode Letter or Mark character, AND
    • Every character in the token is either a Unicode Letter, a Unicode
      Mark (Bengali combining vowels / hasanta count as Mark), or one of
      the single-character punctuation marks ``.-'ঃ``.

    Requiring at least one letter/mark filters out standalone noise tokens
    like ``"."`` ``"!"`` ``"=="`` ``"3"`` that would otherwise pass the
    per-character check because those characters are in the allowed set.

    Trailing tokens that fail this test are stripped one at a time from
    the right until a clean token is found.

    **This function is for personal names only.**  Do NOT call it on dates
    or ID numbers — digits are considered noise in a name context.

    Examples::
        "FAYAZ BIN FARUK =="     → "FAYAZ BIN FARUK"
        "মোঃ ফারুক হোসেন !"     → "মোঃ ফারুক হোসেন"
        "চামেলী আক্তার ."        → "চামেলী আক্তার"
        "চামেলী আক্তার"          → "চামেলী আক্তার"   (unchanged)
        "ফাইয়াজ বিন ফারুক"       → "ফাইয়াজ বিন ফারুক" (unchanged)
    """
    def _is_clean_name_token(token: str) -> bool:
        has_letter_or_mark = any(unicodedata.category(c)[0] in ('L', 'M') for c in token)
        return has_letter_or_mark and all(_is_clean_name_char(c) for c in token)

    words = text.strip().split()
    while words:
        if _is_clean_name_token(words[-1]):
            break
        words.pop()
    return " ".join(words)


def _is_latin_name(text: str, min_words: int = 1) -> bool:
    """Return True if text looks like a Latin personal name (not a header).

    Parameters
    ----------
    min_words:
        Minimum number of words required.  Pass ``min_words=2`` for the
        fallback path (no keyword found) to avoid accepting short garbage
        tokens like ``"AAS"`` or ``"oo"`` as the person's name.
    """
    s = text.strip()
    if not (
        len(s) >= 4
        and all(c.isalpha() or c.isspace() or c in ".-'" for c in s)
        and not any(c.isdigit() for c in s)
    ):
        return False
    words = s.split()
    if len(words) < min_words:
        return False
    words_lower = {w.lower().strip("'.") for w in words}
    return not (words_lower & _LATIN_HEADER_PHRASES)


def _normalize_date(raw: str) -> str:
    """Normalise any date string to DD Mon YYYY, returning raw on failure."""
    m = _DATE_WRITTEN_RE.search(raw)
    if m:
        day   = m.group(1).zfill(2)
        month = _MONTH_MAP.get(m.group(2).lower(), m.group(2).capitalize())
        return f"{day} {month} {m.group(3)}"
    m = _DATE_DMY_SEP_RE.search(raw)
    if m:
        month = _MONTH_MAP.get(m.group(2).lower(), m.group(2).capitalize())
        return f"{m.group(1).zfill(2)} {month} {m.group(3)}"
    return raw.strip()


def _normalize_blood_group(group: str, sign_word: str = "") -> str:
    g = group.upper()
    s = sign_word.lower().strip()
    if s in ("+", "positive", "পজিটিভ"):
        return f"{g}+"
    if s in ("-", "negative", "নেগেটিভ"):
        return f"{g}-"
    return f"{g}{s}"


def _run_with_timeout(func, args: tuple, timeout: int = _OCR_TIMEOUT_SECONDS) -> Any:
    result: list[Any]                     = [None]
    exc:    list[Optional[BaseException]] = [None]

    def _target():
        try:
            result[0] = func(*args)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise OCRTimeoutError(
            f"OCR exceeded {timeout}s time limit. Try a smaller image."
        )
    if exc[0] is not None:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Tesseract word record type
# ---------------------------------------------------------------------------
# Each record: (bbox [[x,y],[x+w,y],[x+w,y+h],[x,y+h]], text, confidence_0_1, line_key)
# line_key groups words by Tesseract's internal line detection (block-par-line)
TessResult = tuple[list[list[int]], str, float, str]


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class NIDOCRService:
    """
    Extracts structured NID card fields using Tesseract (primary)
    with EasyOCR as an optional fallback.

    Tesseract dual-pass strategy
    ----------------------------
    Pass 1: lang="ben"  — Bangla name, father, mother, header text
    Pass 2: lang="eng"  — English name, NID number, date of birth, blood group

    Results are merged by bounding-box overlap; higher-confidence word wins.
    """

    # Singleton EasyOCR reader (only loaded if Tesseract is unavailable)
    _easyocr_reader: Any            = None
    _reader_lock: threading.Lock    = threading.Lock()

    def __init__(self) -> None:
        self._preprocessor = NIDImagePreprocessor()


    def extract_text(
        self,
        image_path: Union[str, Path, np.ndarray],
    ) -> dict[str, Any]:
        """
        Full pipeline: preprocess → dual-pass Tesseract → parse → return.

        Returns
        -------
        {
          "raw_text"           : str,   full OCR dump (merged passes)
          "confidence"         : float, mean word confidence 0–1
          "fields"             : dict,  all 8 NID fields
          "processing_time_ms" : int,
          "ocr_engine"         : str,   "tesseract" or "easyocr"
        }
        """
        start = time.monotonic()

        # --- Preprocess ---------------------------------------------------
        preprocessed = self._preprocessor.preprocess(image_path)

        if self._is_blank_image(preprocessed):
            logger.warning("extract_text: blank image detected")
            return self._empty_result(int((time.monotonic() - start) * 1000))

        # --- OCR ----------------------------------------------------------
        try:
            ocr_results, engine = _run_with_timeout(
                self._run_ocr,
                (preprocessed,),
                timeout=_OCR_TIMEOUT_SECONDS,
            )
        except OCRTimeoutError:
            raise
        except MemoryError:
            return self._error_result(
                "Out of memory. Try a lower-resolution image.",
                int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            raise OCRServiceError(f"OCR failure: {exc}") from exc

        # --- Aggregate ----------------------------------------------------
        merged     = ocr_results["merged"]
        raw_text   = self._build_raw_text(ocr_results)
        confidence = self._calculate_confidence(merged)
        fields     = self._parse_fields_dual(ocr_results)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "extract_text: %dms  conf=%.2f  engine=%s  "
            "nid=%r  name_en=%r  name_bn=%r",
            elapsed_ms, confidence, engine,
            fields["nid_number"], fields["name_english"], fields["name_bangla"],
        )

        return {
            "raw_text":           raw_text,
            "confidence":         round(confidence, 4),
            "fields":             fields,
            "processing_time_ms": elapsed_ms,
            "ocr_engine":         engine,
        }

    # ------------------------------------------------------------------
    # OCR runner — Tesseract primary, EasyOCR fallback
    # ------------------------------------------------------------------

    def _run_ocr(
        self,
        img: np.ndarray,
    ) -> tuple[dict[str, list[TessResult]], str]:
        """
        Try Tesseract first; fall back to EasyOCR if unavailable.

        Returns
        -------
        (result_dict, engine_name)

        result_dict has keys:
          "bn"     — Bangla-pass word results
          "en"     — English-pass word results
          "merged" — combined for raw_text / confidence (all words)
        """
        try:
            import pytesseract  # noqa: PLC0415
            results = self._run_tesseract_dual_pass(img, pytesseract)
            return results, "tesseract"
        except ImportError:
            logger.warning("pytesseract not installed — falling back to EasyOCR")

        try:
            easyocr_results = self._run_easyocr(img)
            # EasyOCR runs bn+en together; use same list for all keys
            return {"bn": easyocr_results, "en": easyocr_results,
                    "merged": easyocr_results}, "easyocr"
        except ImportError:
            pass

        raise OCRServiceError(
            "No OCR backend found. Install pytesseract "
            "(pip install pytesseract) and Tesseract binary, "
            "or install easyocr (pip install easyocr)."
        )

    # ------------------------------------------------------------------
    # Tesseract dual-pass
    # ------------------------------------------------------------------

    def _run_tesseract_dual_pass(
        self,
        img: np.ndarray,
        pytesseract: Any,
    ) -> dict[str, list[TessResult]]:
        """
        Run Tesseract twice: Bengali pass + English pass.

        Returns a dict with separate results per pass plus a merged list
        for raw_text/confidence.  Field parsing uses per-pass results
        so that Bangla-pass line grouping is used for Bangla fields and
        English-pass line grouping is used for English fields — this avoids
        the spatial ordering corruption that occurs when merging bboxes
        from two separate passes.
        """
        logger.info("Tesseract: starting Bangla pass (ben)…")
        bn_results = self._tess_pass(img, pytesseract, lang="ben",
                                     config=_TESS_CONFIG_BN)

        logger.info("Tesseract: starting English pass (eng)…")
        en_results = self._tess_pass(img, pytesseract, lang="eng",
                                     config=_TESS_CONFIG_EN)

        merged = self._merge_tess_results(bn_results, en_results)
        logger.info(
            "Tesseract: bn=%d words, en=%d words, merged=%d words",
            len(bn_results), len(en_results), len(merged),
        )
        return {"bn": bn_results, "en": en_results, "merged": merged}

    @staticmethod
    def _tess_pass(
        img: np.ndarray,
        pytesseract: Any,
        lang: str,
        config: str,
    ) -> list[TessResult]:
        """
        Run one Tesseract pass and return word-level results.

        Tesseract's image_to_data groups output by:
          level 1=page, 2=block, 3=paragraph, 4=line, 5=word

        We use its built-in line grouping (block_num + par_num + line_num)
        to reconstruct text lines.  This is more reliable than sorting by
        Y coordinate because Tesseract already resolves multi-column layouts
        and non-uniform line spacing internally.
        """
        try:
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.error("_tess_pass(%s): %s", lang, exc)
            return []

        results: list[TessResult] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf_raw = data["conf"][i]
            if not isinstance(conf_raw, (int, float)) or conf_raw < _MIN_TESS_CONF:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            conf = float(conf_raw) / 100.0   # normalise to 0–1
            line_key = f"{data['block_num'][i]}-{data['par_num'][i]}-{data['line_num'][i]}"
            results.append((bbox, word, conf, line_key))

        return results

    @staticmethod
    def _merge_tess_results(
        bn_results: list[TessResult],
        en_results: list[TessResult],
    ) -> list[TessResult]:
        """
        Merge Bangla-pass and English-pass results.

        Strategy
        ---------
        • For each bbox in the English pass, check whether the Bangla
          pass has a result at the same location (IoU > 0.5).
          - If YES: keep whichever has higher confidence; also keep
            the Bangla result if it contains Bangla chars (always prefer
            Bangla text from the Bangla pass).
          - If NO: keep the English result (unique to this pass).
        • All Bangla-pass results that weren't matched are appended.

        This ensures:
          • English words (name, NID, date) come from the English pass.
          • Bangla words (নাম, পিতা, মাতা values) come from the Bangla pass.
          • Neither pass silently drops the other's results.
        """
        def bbox_to_xywh(bbox: list) -> tuple[int, int, int, int]:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1 = min(xs), min(ys)
            return x1, y1, max(xs) - x1, max(ys) - y1

        def iou(a: list, b: list) -> float:
            ax, ay, aw, ah = bbox_to_xywh(a)
            bx, by, bw, bh = bbox_to_xywh(b)
            ix1 = max(ax, bx); iy1 = max(ay, by)
            ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        merged: list[TessResult] = []
        used_bn_indices: set[int] = set()

        for en_bbox, en_text, en_conf, en_lk in en_results:
            best_bn_idx   = -1
            best_bn_iou   = 0.0
            for bn_idx, (bn_bbox, bn_text, bn_conf, bn_lk) in enumerate(bn_results):
                overlap = iou(en_bbox, bn_bbox)
                if overlap > best_bn_iou:
                    best_bn_iou = overlap
                    best_bn_idx = bn_idx

            if best_bn_iou > 0.5 and best_bn_idx >= 0:
                bn_bbox, bn_text, bn_conf, bn_lk = bn_results[best_bn_idx]
                used_bn_indices.add(best_bn_idx)
                # Prefer Bangla result if it has Bangla chars
                if _is_bangla(bn_text):
                    merged.append((bn_bbox, bn_text, bn_conf, bn_lk))
                else:
                    # Both are Latin: take higher confidence
                    if bn_conf >= en_conf:
                        merged.append((bn_bbox, bn_text, bn_conf, bn_lk))
                    else:
                        merged.append((en_bbox, en_text, en_conf, en_lk))
            else:
                merged.append((en_bbox, en_text, en_conf, en_lk))

        # Append unmatched Bangla-pass results
        for bn_idx, bn_result in enumerate(bn_results):
            if bn_idx not in used_bn_indices:
                merged.append(bn_result)

        return merged

    # ------------------------------------------------------------------
    # EasyOCR fallback
    # ------------------------------------------------------------------

    def _run_easyocr(self, img: np.ndarray) -> list[TessResult]:
        """Load EasyOCR singleton and run bn+en pass."""
        if NIDOCRService._easyocr_reader is None:
            with NIDOCRService._reader_lock:
                if NIDOCRService._easyocr_reader is None:
                    import easyocr  # noqa: PLC0415
                    logger.info("Initialising EasyOCR (bn + en)…")
                    NIDOCRService._easyocr_reader = easyocr.Reader(
                        lang_list=["bn", "en"], gpu=False, verbose=False
                    )
        raw = NIDOCRService._easyocr_reader.readtext(img, detail=1, paragraph=False)
        # Convert 3-tuples to 4-tuples with synthetic line keys (one word per line)
        return [(bbox, text, conf, f"1-1-{i}") for i, (bbox, text, conf) in enumerate(raw)]

    # ------------------------------------------------------------------
    # Field parser
    # ------------------------------------------------------------------

    def _parse_fields(
        self,
        ocr_results: list[TessResult],
    ) -> dict[str, str]:
        """
        Extract all 8 NID fields from merged OCR word results.

        Line reconstruction
        -------------------
        Words are first sorted by (Y-centre, X-centre) so the text reads
        naturally left-to-right, top-to-bottom.  Adjacent words whose
        Y-centres are within 15 px of each other are grouped into the same
        reconstructed line.  This tolerates baseline jitter from Tesseract
        and gives us full text lines like "Date of Birth: 28 Feb 2003"
        rather than individual tokens.
        """
        fields: dict[str, str] = {
            "name_bangla":        "",
            "name_english":       "",
            "father_name_bangla": "",
            "mother_name_bangla": "",
            "date_of_birth":      "",
            "nid_number":         "",
            "blood_group":        "",
            "address_bangla":     "",
        }

        if not ocr_results:
            return fields

        # ---------- Reconstruct text lines from word bboxes ----------
        def _y_centre(entry: TessResult) -> float:
            pts = entry[0]
            return sum(p[1] for p in pts) / len(pts)

        def _x_centre(entry: TessResult) -> float:
            pts = entry[0]
            return sum(p[0] for p in pts) / len(pts)

        sorted_words = sorted(ocr_results, key=lambda e: (_y_centre(e), _x_centre(e)))

        # Group into lines: words within 15 px vertically → same line
        lines: list[str] = []
        if sorted_words:
            current_line_words: list[str] = [sorted_words[0][1]]
            current_y = _y_centre(sorted_words[0])

            for entry in sorted_words[1:]:
                y = _y_centre(entry)
                if abs(y - current_y) <= 15:
                    current_line_words.append(entry[1])
                else:
                    lines.append(" ".join(current_line_words))
                    current_line_words = [entry[1]]
                    current_y = y

            if current_line_words:
                lines.append(" ".join(current_line_words))

        full_text = "\n".join(lines)
        logger.debug("_parse_fields: reconstructed %d lines:\n%s", len(lines), full_text)

        # ---------- Global regex extractions (not line-dependent) ----------
        fields["nid_number"]    = self._extract_nid_number(full_text)
        fields["date_of_birth"] = self._extract_date_of_birth(full_text)
        fields["blood_group"]   = self._extract_blood_group(full_text)

        # ---------- Keyword proximity loop ----------
        idx = 0
        while idx < len(lines):
            line = lines[idx]

            # English name — inline or next line
            if _KW_NAME_EN.match(line) and not fields["name_english"]:
                inline = _KW_NAME_EN.sub("", line).strip()
                if inline and _is_latin_name(inline):
                    fields["name_english"] = inline
                elif idx + 1 < len(lines) and _is_latin_name(lines[idx + 1]):
                    fields["name_english"] = lines[idx + 1].strip()
                    idx += 1

            # Bangla name — inline or next line
            if _KW_NAME_BN.search(line) and not fields["name_bangla"]:
                inline = _KW_NAME_BN.sub("", line).strip().lstrip(":। ")
                if inline and _is_bangla(inline) and len(inline) >= 3:
                    fields["name_bangla"] = inline
                elif idx + 1 < len(lines) and _is_bangla(lines[idx + 1]):
                    fields["name_bangla"] = lines[idx + 1].strip()
                    idx += 1

            # Father name — inline or next line
            if _KW_FATHER_BN.search(line) and not fields["father_name_bangla"]:
                value = self._extract_after_keyword(line, _KW_FATHER_BN)
                if not value and idx + 1 < len(lines):
                    value = lines[idx + 1]
                    idx += 1
                if value:
                    fields["father_name_bangla"] = value.strip()

            # Mother name — inline or next line
            if _KW_MOTHER_BN.search(line) and not fields["mother_name_bangla"]:
                value = self._extract_after_keyword(line, _KW_MOTHER_BN)
                if not value and idx + 1 < len(lines):
                    value = lines[idx + 1]
                    idx += 1
                if value:
                    fields["mother_name_bangla"] = value.strip()

            # Date of birth — inline or next line  (T1: handles split tokens)
            if _KW_DOB.search(line) and not fields["date_of_birth"]:
                value = self._extract_after_keyword(line, _KW_DOB)
                if not value and idx + 1 < len(lines):
                    value = lines[idx + 1]
                    idx += 1
                if value:
                    fields["date_of_birth"] = _normalize_date(value)

            # Address — up to 3 lines after keyword
            if _KW_ADDRESS.search(line) and not fields["address_bangla"]:
                addr_lines: list[str] = []
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    cand = lines[j]
                    if (
                        _KW_FATHER_BN.search(cand)
                        or _KW_MOTHER_BN.search(cand)
                        or _KW_DOB.search(cand)
                    ):
                        break
                    if cand:
                        addr_lines.append(cand)
                fields["address_bangla"] = "\n".join(addr_lines)

            idx += 1

        # ---------- Fallbacks ----------

        # Bangla name fallback: first substantial Bangla line, skip headers
        if not fields["name_bangla"]:
            for line in lines:
                if (
                    _is_bangla(line)
                    and len(line.strip()) >= 3
                    and not any(c.isdigit() for c in line)
                    and not _KW_FATHER_BN.search(line)
                    and not _KW_MOTHER_BN.search(line)
                    and not _KW_ADDRESS.search(line)
                    and not _HEADER_RE.search(line)
                ):
                    fields["name_bangla"] = line.strip()
                    break

        # English name fallback: first clean Latin line
        if not fields["name_english"]:
            for line in lines:
                if _is_latin_name(line, min_words=2) and not _KW_ADDRESS.search(line):
                    fields["name_english"] = line.strip()
                    break

        return fields

    # ------------------------------------------------------------------
    # Dual-pass field parser (primary)
    # ------------------------------------------------------------------

    def _parse_fields_dual(
        self,
        ocr_results: dict[str, list[TessResult]],
    ) -> dict[str, str]:
        """
        Parse NID fields using per-pass line reconstruction.

        Bangla fields (name_bangla, father, mother, address) are extracted
        from the **Bangla-pass** line reconstruction, which preserves the
        correct spatial ordering of Bangla words.

        English fields (name_english, nid, date_of_birth, blood_group) are
        extracted from the **English-pass** line reconstruction, which
        correctly reads Latin text without Bangla interference.

        This avoids the spatial ordering corruption that happened when both
        passes were merged into one flat list and then re-sorted by Y,X —
        words from different passes at similar Y positions would interleave
        incorrectly.
        """
        bn_lines = self._reconstruct_lines(ocr_results.get("bn", []))
        en_lines = self._reconstruct_lines(ocr_results.get("en", []))

        logger.debug(
            "_parse_fields_dual: bn_lines=%d, en_lines=%d",
            len(bn_lines), len(en_lines),
        )
        for i, line in enumerate(bn_lines):
            logger.debug("  BN line %d: %r", i, line)
        for i, line in enumerate(en_lines):
            logger.debug("  EN line %d: %r", i, line)

        fields: dict[str, str] = {
            "name_bangla":        "",
            "name_english":       "",
            "father_name_bangla": "",
            "mother_name_bangla": "",
            "date_of_birth":      "",
            "nid_number":         "",
            "blood_group":        "",
            "address_bangla":     "",
        }

        # ---- English fields from English-pass lines ----
        en_full = "\n".join(en_lines)
        fields["nid_number"]    = self._extract_nid_number(en_full)
        fields["date_of_birth"] = self._extract_date_of_birth(en_full)
        fields["blood_group"]   = self._extract_blood_group(en_full)

        # English name — keyword search in English-pass lines
        # _KW_NAME_EN now uses \b (word boundary) instead of ^ so it
        # matches "পর name FAYAZ BIN FARUK" (phone-photo OCR noise before keyword)
        for idx, line in enumerate(en_lines):
            if _KW_NAME_EN.search(line) and not fields["name_english"]:
                inline = self._extract_after_keyword(line, _KW_NAME_EN)
                # Strip trailing noise, then trim to the uppercase prefix
                inline = _trim_to_uppercase_name(_strip_trailing_noise(inline))
                if inline and _is_latin_name(inline) and _is_uppercase_name(inline):
                    fields["name_english"] = inline
                elif idx + 1 < len(en_lines):
                    nxt = _trim_to_uppercase_name(
                        _strip_trailing_noise(en_lines[idx + 1].strip())
                    )
                    if _is_latin_name(nxt) and _is_uppercase_name(nxt):
                        fields["name_english"] = nxt

        # English name fallback: first clean multi-word ALL-CAPS Latin line
        # require_uppercase guards against screenshot UI garbage like "Rate ay SES"
        if not fields["name_english"]:
            for line in en_lines:
                stripped = _trim_to_uppercase_name(
                    _strip_trailing_noise(line.strip())
                )
                if (
                    _is_latin_name(stripped, min_words=2)
                    and _is_uppercase_name(stripped)
                    and not _KW_ADDRESS.search(stripped)
                ):
                    fields["name_english"] = stripped
                    break

        # ---- Bangla fields from Bangla-pass lines ----
        bn_full = "\n".join(bn_lines)

        idx = 0
        while idx < len(bn_lines):
            line = bn_lines[idx]

            # Bangla name — extract text AFTER the keyword (drops any
            # noise that precedes it on the same line, e.g. "( দাম ফাইয়াজ…"
            # where "দাম" is the phone-photo OCR variant of "নাম:")
            if _KW_NAME_BN.search(line) and not fields["name_bangla"]:
                inline = self._extract_after_keyword(line, _KW_NAME_BN)
                inline = _strip_trailing_noise(inline)
                if inline and _is_bangla(inline) and len(inline) >= 3:
                    fields["name_bangla"] = inline
                elif idx + 1 < len(bn_lines) and _is_bangla(bn_lines[idx + 1]):
                    fields["name_bangla"] = _strip_trailing_noise(bn_lines[idx + 1].strip())
                    idx += 1

            # Father name
            if _KW_FATHER_BN.search(line) and not fields["father_name_bangla"]:
                value = self._extract_after_keyword(line, _KW_FATHER_BN)
                if not value and idx + 1 < len(bn_lines):
                    value = bn_lines[idx + 1]
                    idx += 1
                if value:
                    fields["father_name_bangla"] = _strip_trailing_noise(value.strip())

            # Mother name
            if _KW_MOTHER_BN.search(line) and not fields["mother_name_bangla"]:
                value = self._extract_after_keyword(line, _KW_MOTHER_BN)
                if not value and idx + 1 < len(bn_lines):
                    value = bn_lines[idx + 1]
                    idx += 1
                if value:
                    fields["mother_name_bangla"] = _strip_trailing_noise(value.strip())

            # Date of birth from Bangla lines (fallback if EN pass missed it)
            if _KW_DOB.search(line) and not fields["date_of_birth"]:
                value = self._extract_after_keyword(line, _KW_DOB)
                if not value and idx + 1 < len(bn_lines):
                    value = bn_lines[idx + 1]
                    idx += 1
                if value:
                    fields["date_of_birth"] = _normalize_date(value)

            # Address — up to 3 lines after keyword
            if _KW_ADDRESS.search(line) and not fields["address_bangla"]:
                addr_lines: list[str] = []
                for j in range(idx + 1, min(idx + 4, len(bn_lines))):
                    cand = bn_lines[j]
                    if (
                        _KW_FATHER_BN.search(cand)
                        or _KW_MOTHER_BN.search(cand)
                        or _KW_DOB.search(cand)
                    ):
                        break
                    if cand:
                        addr_lines.append(cand)
                fields["address_bangla"] = "\n".join(addr_lines)

            idx += 1

        # ---- Bangla name fallback ----
        if not fields["name_bangla"]:
            for line in bn_lines:
                candidate = _strip_trailing_noise(line.strip())
                if (
                    _is_bangla(candidate)
                    and len(candidate.strip()) >= 3
                    and not any(c.isdigit() for c in candidate)
                    and not _KW_FATHER_BN.search(candidate)
                    and not _KW_MOTHER_BN.search(candidate)
                    and not _KW_ADDRESS.search(candidate)
                    and not _HEADER_RE.search(candidate)
                ):
                    fields["name_bangla"] = candidate
                    break

        # ---- Cross-pass fallbacks ----
        # If NID not found in English pass, try Bangla pass raw text
        if not fields["nid_number"]:
            fields["nid_number"] = self._extract_nid_number(bn_full)
        # If DOB not found, try Bangla pass
        if not fields["date_of_birth"]:
            fields["date_of_birth"] = self._extract_date_of_birth(bn_full)

        return fields

    # ------------------------------------------------------------------
    # Line reconstruction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_lines(
        ocr_results: list[TessResult],
    ) -> list[str]:
        """
        Group OCR word results into text lines using Tesseract's built-in
        line detection (block_num-par_num-line_num).

        Within each line, words are sorted left-to-right by their X position.
        Lines are ordered top-to-bottom by mean Y position.

        Using Tesseract's line grouping is far more reliable than Y-centre
        proximity because Tesseract already handles multi-column layouts,
        non-uniform spacing, and bbox jitter internally.
        """
        if not ocr_results:
            return []

        def _x_left(entry: TessResult) -> int:
            return entry[0][0][0]  # top-left x

        def _y_centre(entry: TessResult) -> float:
            pts = entry[0]
            return sum(p[1] for p in pts) / len(pts)

        # Group words by their Tesseract line_key
        from collections import OrderedDict
        line_groups: dict[str, list[TessResult]] = OrderedDict()
        for entry in ocr_results:
            lk = entry[3]  # line_key
            if lk not in line_groups:
                line_groups[lk] = []
            line_groups[lk].append(entry)

        # Sort each line's words left-to-right by X position
        line_entries: list[tuple[float, str]] = []
        for lk, words in line_groups.items():
            words.sort(key=_x_left)
            mean_y = sum(_y_centre(w) for w in words) / len(words)
            text = " ".join(w[1] for w in words)
            line_entries.append((mean_y, text))

        # Sort lines top-to-bottom
        line_entries.sort(key=lambda e: e[0])
        return [text for _, text in line_entries]

    # ------------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_nid_number(text: str) -> str:
        """
        T2: handles Tesseract OCR errors like "10 NO:" and "ID NO :"
        Prefers labelled match; falls back to bare 10/17-digit number.
        """
        candidates: list[str] = []
        for m in _NID_LABEL_RE.finditer(text):
            candidates.append(m.group(1))
        for m in _NID_BARE_RE.finditer(text):
            candidates.append(m.group(1))
        if not candidates:
            return ""
        return max(candidates, key=len)

    @staticmethod
    def _extract_date_of_birth(full_text: str) -> str:
        m = _DATE_WRITTEN_RE.search(full_text)
        if m:
            return _normalize_date(m.group(0))
        m = _DATE_DMY_SEP_RE.search(full_text)
        if m:
            return _normalize_date(m.group(0))
        return ""

    @staticmethod
    def _extract_blood_group(text: str) -> str:
        m = _BLOOD_BARE_RE.search(text)
        if m:
            return m.group(0).upper()
        m = _BLOOD_WORD_RE.search(text)
        if m:
            return _normalize_blood_group(m.group(1), m.group(2))
        return ""

    @staticmethod
    def _extract_after_keyword(line: str, keyword_re: re.Pattern) -> str:
        """Return the text that follows the keyword on the same line."""
        m = keyword_re.search(line)
        if not m:
            return ""
        return line[m.end():].lstrip(" :.-\t।").strip()

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(ocr_results: list[TessResult]) -> float:
        """
        Average word-level confidence.
        T4: uses _MIN_TESS_CONF (0.30) floor since Tesseract's 0–1 scale
        has different noise characteristics than EasyOCR.
        """
        if not ocr_results:
            return 0.0
        scores = [
            c for (_, _, c, *_rest) in ocr_results
            if isinstance(c, (int, float)) and c >= 0.30
        ]
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_text(ocr_results: dict[str, list[TessResult]] | list[TessResult]) -> str:
        """
        Build a human-readable raw text dump from OCR results.

        Uses the merged pass with _reconstruct_lines so the output
        contains proper text lines (not one token per line) matching
        the original card layout.  Falls back to flat word list if
        given a plain list (EasyOCR path).
        """
        if isinstance(ocr_results, dict):
            word_list = ocr_results.get("merged", [])
        else:
            word_list = ocr_results  # legacy / EasyOCR flat list

        lines = NIDOCRService._reconstruct_lines(word_list)
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _is_blank_image(img: np.ndarray) -> bool:
        return float(np.std(img)) < 10.0

    @staticmethod
    def _empty_result(elapsed_ms: int) -> dict[str, Any]:
        return {
            "raw_text":           "",
            "confidence":         0.0,
            "fields": {
                "name_bangla":        "",
                "name_english":       "",
                "father_name_bangla": "",
                "mother_name_bangla": "",
                "date_of_birth":      "",
                "nid_number":         "",
                "blood_group":        "",
                "address_bangla":     "",
            },
            "processing_time_ms": elapsed_ms,
            "ocr_engine":         "tesseract",
        }

    @staticmethod
    def _error_result(message: str, elapsed_ms: int) -> dict[str, Any]:
        r = NIDOCRService._empty_result(elapsed_ms)
        r["raw_text"] = f"ERROR: {message}"
        return r
