"""
Excel export utility for NID card records.

Produces a formatted ``.xlsx`` workbook with two sheets:
  - Sheet 1 "NID Records": tabular data with styled headers and alternating rows.
  - Sheet 2 "Summary": aggregate statistics and blood group distribution.

Handles Bangla Unicode text natively via openpyxl.
Large querysets (>1 000 rows) are processed with the ``queryset.iterator()``
API to avoid loading the entire result set into memory.
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import Optional

from django.db.models import Count, QuerySet
from django.http import HttpResponse
from django.utils import timezone
from openpyxl import Workbook
from openpyxl.styles import (
    Alignment,
    Font,
    PatternFill,
)
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_HEADER_BG     = "1E5C8A"          # Dark blue — row 1 background
_ALT_ROW_BG    = "DCE9F7"          # Light blue — even-row fill
_HEADER_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_BODY_FONT     = Font(name="Calibri", size=10)
_WRAP_ALIGN    = Alignment(wrap_text=True, vertical="top")
_CENTER_ALIGN  = Alignment(horizontal="center", vertical="top")
_MAX_COL_WIDTH = 60                 # Characters — prevents absurdly wide columns

# ---------------------------------------------------------------------------
# Column specification  (field name, display header, max_width_hint)
# ---------------------------------------------------------------------------
_COLUMNS: list[tuple[str, str, int]] = [
    ("id",                  "ID",               6),
    ("name_english",        "Name (English)",   30),
    ("name_bangla",         "Name (Bangla)",     30),
    ("nid_number",          "NID Number",       20),
    ("date_of_birth",       "Date of Birth",    16),
    ("father_name_bangla",  "Father Name",      30),
    ("mother_name_bangla",  "Mother Name",      30),
    ("blood_group",         "Blood Group",      12),
    ("address_bangla",      "Address",          45),
    ("ocr_confidence",      "OCR Confidence",   15),
    ("processing_status",   "Status",           14),
    ("created_at",          "Created At",       18),
]


class NIDExcelExporter:
    """
    Builds an Excel workbook from a ``NIDRecord`` queryset and returns
    it wrapped in a Django ``HttpResponse``.

    Usage::

        exporter = NIDExcelExporter()
        response = exporter.export(queryset)
        return response
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        queryset: QuerySet,
        filename: Optional[str] = None,
    ) -> HttpResponse:
        """
        Generate an ``.xlsx`` file and return it as an ``HttpResponse``.

        The queryset is consumed with ``iterator()`` to keep memory usage
        flat regardless of result count.

        Args:
            queryset: A ``NIDRecord`` queryset (may be filtered/ordered).
            filename: Override the auto-generated ``nid_records_<ts>.xlsx``
                      filename if provided.

        Returns:
            ``HttpResponse`` with Excel content-type and attachment header.
        """
        if filename is None:
            ts = timezone.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nid_records_{ts}.xlsx"

        logger.info("NIDExcelExporter.export: starting export to %s", filename)

        wb = Workbook()
        ws_data = wb.active
        ws_data.title = "NID Records"

        row_count = self._write_data_sheet(ws_data, queryset)
        self._write_summary_sheet(wb, queryset, row_count)

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        response = HttpResponse(
            buffer.getvalue(),
            content_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        logger.info(
            "NIDExcelExporter.export: wrote %d data rows to %s",
            row_count, filename,
        )
        return response

    # ------------------------------------------------------------------
    # Sheet 1 — NID Records
    # ------------------------------------------------------------------

    def _write_data_sheet(self, ws, queryset: QuerySet) -> int:
        """
        Populate the "NID Records" worksheet.

        Steps:
          1. Write the bold, colour-coded header row.
          2. Freeze the header row so it stays visible during scrolling.
          3. Iterate the queryset (``iterator()`` keeps memory flat).
          4. Apply alternate row shading.
          5. Auto-fit column widths based on content (capped at ``_MAX_COL_WIDTH``).

        Args:
            ws:       openpyxl Worksheet to write into.
            queryset: Source data queryset.

        Returns:
            Number of data rows written (excluding the header).
        """
        header_fill = PatternFill(
            fill_type="solid", fgColor=_HEADER_BG
        )
        alt_fill = PatternFill(
            fill_type="solid", fgColor=_ALT_ROW_BG
        )

        # --- Header row ---------------------------------------------------
        headers = [col[1] for col in _COLUMNS]
        ws.append(headers)
        for col_idx, cell in enumerate(ws[1], start=1):
            cell.font      = _HEADER_FONT
            cell.fill      = header_fill
            cell.alignment = _CENTER_ALIGN

        # Freeze header row
        ws.freeze_panes = "A2"

        # Track max content widths per column for auto-fit
        col_widths: list[int] = [len(h) for h in headers]

        # --- Data rows ----------------------------------------------------
        row_num = 0
        for record in queryset.iterator():
            row_num += 1
            excel_row_idx = row_num + 1   # 1-based, offset by header

            row_values = self._record_to_row(record)
            ws.append(row_values)

            # Alternate row shading on even data rows
            if row_num % 2 == 0:
                for cell in ws[excel_row_idx]:
                    cell.fill = alt_fill

            # Apply fonts and alignment
            for col_idx, cell in enumerate(ws[excel_row_idx], start=0):
                cell.font = _BODY_FONT
                cell.alignment = (
                    _WRAP_ALIGN
                    if _COLUMNS[col_idx][0] == "address_bangla"
                    else Alignment(vertical="top")
                )
                # Update max width tracker
                val_len = len(str(cell.value)) if cell.value is not None else 0
                if val_len > col_widths[col_idx]:
                    col_widths[col_idx] = val_len

        # --- Auto-fit column widths ---------------------------------------
        for col_idx, (_, _, hint_width) in enumerate(_COLUMNS, start=1):
            computed = min(col_widths[col_idx - 1] + 2, _MAX_COL_WIDTH)
            # Never shrink below the hint minimum
            width = max(computed, hint_width)
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        # Enable text-wrap for address column
        addr_col_idx = next(
            i + 1 for i, (field, _, _) in enumerate(_COLUMNS) if field == "address_bangla"
        )
        for row in ws.iter_rows(min_row=2, min_col=addr_col_idx, max_col=addr_col_idx):
            for cell in row:
                cell.alignment = _WRAP_ALIGN

        return row_num

    @staticmethod
    def _record_to_row(record) -> list:
        """
        Convert a single ``NIDRecord`` instance to a list of cell values
        matching the ``_COLUMNS`` specification.

        - ``None`` values become empty strings (never ``"None"``).
        - ``ocr_confidence`` formatted as a percentage string (``"87.3%"``),
          or ``"N/A"`` when null.
        - ``created_at`` formatted as ``DD/MM/YYYY HH:MM``.

        Args:
            record: A ``NIDRecord`` model instance.

        Returns:
            List of Python scalars / strings for openpyxl ``append()``.
        """
        def _safe(value) -> str:
            """Return empty string instead of None or the string 'None'."""
            if value is None:
                return ""
            s = str(value)
            return "" if s == "None" else s

        # OCR confidence
        if record.ocr_confidence is not None:
            conf_str = f"{record.ocr_confidence * 100:.1f}%"
        else:
            conf_str = "N/A"

        # Created at
        if record.created_at:
            # Convert from UTC/aware to local time before formatting
            local_dt = timezone.localtime(record.created_at)
            created_str = local_dt.strftime("%d/%m/%Y %H:%M")
        else:
            created_str = ""

        return [
            record.id,
            _safe(record.name_english),
            _safe(record.name_bangla),
            _safe(record.nid_number),
            _safe(record.date_of_birth),
            _safe(record.father_name_bangla),
            _safe(record.mother_name_bangla),
            _safe(record.blood_group),
            _safe(record.address_bangla),
            conf_str,
            _safe(record.processing_status),
            created_str,
        ]

    # ------------------------------------------------------------------
    # Sheet 2 — Summary
    # ------------------------------------------------------------------

    def _write_summary_sheet(
        self,
        wb: Workbook,
        queryset: QuerySet,
        exported_count: int,
    ) -> None:
        """
        Create and populate the "Summary" worksheet in *wb*.

        Sections:
          - Export metadata (count, timestamp).
          - Blood group distribution table.
          - Processing status success rate.

        Args:
            wb:             The parent Workbook.
            queryset:       Same queryset used for Sheet 1 (for aggregate queries).
            exported_count: Row count already computed by ``_write_data_sheet``.
        """
        ws = wb.create_sheet(title="Summary")

        header_fill = PatternFill(fill_type="solid", fgColor=_HEADER_BG)
        title_font  = Font(name="Calibri", bold=True, size=12, color="FFFFFF")
        label_font  = Font(name="Calibri", bold=True, size=10)
        value_font  = Font(name="Calibri", size=10)

        def _header_row(text: str) -> None:
            ws.append([text])
            cell = ws.cell(row=ws.max_row, column=1)
            cell.font = title_font
            cell.fill = header_fill
            cell.alignment = _CENTER_ALIGN
            ws.merge_cells(
                start_row=ws.max_row, start_column=1,
                end_row=ws.max_row, end_column=2,
            )

        def _kv_row(label: str, value) -> None:
            ws.append([label, value])
            ws.cell(row=ws.max_row, column=1).font = label_font
            ws.cell(row=ws.max_row, column=2).font = value_font

        # --- Export info --------------------------------------------------
        _header_row("Export Information")
        _kv_row("Total records exported", exported_count)
        _kv_row(
            "Export date/time",
            timezone.localtime(timezone.now()).strftime("%d/%m/%Y %H:%M:%S"),
        )
        ws.append([])

        # --- Blood group distribution -------------------------------------
        _header_row("Blood Group Distribution")
        dist = (
            queryset
            .exclude(blood_group="")
            .values("blood_group")
            .annotate(count=Count("id"))
            .order_by("blood_group")
        )

        if dist:
            ws.append(["Blood Group", "Count"])
            ws.cell(row=ws.max_row, column=1).font = label_font
            ws.cell(row=ws.max_row, column=2).font = label_font
            for item in dist:
                _kv_row(item["blood_group"], item["count"])
        else:
            ws.append(["No blood group data available."])
        ws.append([])

        # --- Processing status summary ------------------------------------
        _header_row("Processing Status")
        total     = queryset.count()
        success   = queryset.filter(processing_status="SUCCESS").count()
        failed    = queryset.filter(processing_status="FAILED").count()
        pending   = queryset.filter(processing_status__in=["PENDING", "PROCESSING"]).count()
        rate      = (success / total * 100) if total else 0.0

        _kv_row("Total",           total)
        _kv_row("Successful",      success)
        _kv_row("Failed",          failed)
        _kv_row("Pending/Processing", pending)
        _kv_row("Success rate",    f"{rate:.1f}%")

        # --- Column widths ------------------------------------------------
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 22
