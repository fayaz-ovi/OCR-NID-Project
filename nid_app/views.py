"""
API views for the NID card OCR application.

Endpoints
---------
POST   /api/upload/           NIDUploadView       — upload & OCR an NID image
GET    /api/records/          NIDRecordListView   — paginated list with filters
GET    /api/records/<id>/     NIDRecordDetailView — retrieve full record
PUT    /api/records/<id>/     NIDRecordDetailView — manually correct OCR fields
DELETE /api/records/<id>/     NIDRecordDetailView — soft-delete
GET    /api/export/           NIDExportView       — download filtered Excel file
GET    /api/stats/            NIDStatsView        — dashboard statistics
"""

from __future__ import annotations

import logging
import os
import re
from datetime import timedelta
from typing import Optional

from django.db.models import Avg, Count, Q ,QuerySet
from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .exceptions import OCRServiceError, OCRTimeoutError, PreprocessingError
from .models import NIDRecord
from .ocr_service import NIDOCRService
from .serializers import (
    NIDRecordListSerializer,
    NIDRecordSerializer,
    NIDRecordUpdateSerializer,
)
from .utils.excel_exporter import NIDExcelExporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upload validation constants
# ---------------------------------------------------------------------------
_MAX_UPLOAD_BYTES       = 10 * 1024 * 1024          # 10 MB
_ALLOWED_PILLOW_FORMATS = {"JPEG", "PNG", "WEBP"}
_HEIC_EXTENSIONS        = {".heic", ".heif"}
_PDF_EXTENSIONS         = {".pdf"}
_LOW_CONFIDENCE         = 0.5                      
_ALLOWED_ORDER_FIELDS = {
    "created_at", "-created_at",
    "name_english", "-name_english",
    "nid_number", "-nid_number",
}


def _validate_upload(
    request: Request,
) -> tuple[Optional[object], list[str], Optional[Response]]:
    """
    Validate the image file attached to *request*.

    Checks (in order):
      1. Field ``image`` is present.
      2. File is not a PDF → 400 with helpful convert hint.
      3. File is not HEIC/HEIF (iPhone) → 415 with conversion advice.
      4. File does not exceed 10 MB → 400.
      5. Pillow can decode it as JPEG / PNG / WEBP → 400 for anything else.

    Args:
        request: DRF ``Request`` containing ``request.FILES``.

    Returns:
        ``(file_object, warnings, error_response)`` where
        ``error_response`` is ``None`` on success or a ready-made DRF
        ``Response`` that the caller should return immediately.
    """
    file = request.FILES.get("image") or request.FILES.get("uploaded_image")

    if not file:
        return None, [], Response(
            {
                "success": False,
                "error": (
                    'No image file provided. '
                    'Send the file in a multipart field named "image".'
                ),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    ext = os.path.splitext(file.name.lower())[1]

    # PDF check
    if ext in _PDF_EXTENSIONS or (file.content_type or "").startswith("application/pdf"):
        return None, [], Response(
            {
                "success": False,
                "error": (
                    "PDF files are not supported. "
                    "Please upload a photo of the NID card (JPEG or PNG)."
                ),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # HEIC check (iPhone default format)
    if ext in _HEIC_EXTENSIONS or (file.content_type or "") in ("image/heic", "image/heif"):
        return None, [], Response(
            {
                "success": False,
                "error": (
                    "HEIC format (iPhone photos) is not directly supported. "
                    "Open the photo in your phone gallery, share it as JPEG, "
                    "or use an online HEIC-to-JPEG converter before uploading."
                ),
            },
            status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    # Size check
    if file.size > _MAX_UPLOAD_BYTES:
        mb = file.size / 1024 / 1024
        return None, [], Response(
            {
                "success": False,
                "error": (
                    f"File too large ({mb:.1f} MB). "
                    "Maximum allowed size is 10 MB."
                ),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Pillow format check — guards against fake extensions
    try:
        from PIL import Image as _PilImage  # noqa: PLC0415

        file.seek(0)
        pil_img    = _PilImage.open(file)
        img_format = pil_img.format or ""
        file.seek(0)
    except Exception as exc:
        logger.warning("_validate_upload: Pillow could not open file — %s", exc)
        return None, [], Response(
            {
                "success": False,
                "error": (
                    "Could not read the uploaded file. "
                    "It may be corrupted or not a valid image."
                ),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    if img_format.upper() not in _ALLOWED_PILLOW_FORMATS:
        return None, [], Response(
            {
                "success": False,
                "error": (
                    f'Unsupported image format "{img_format}". '
                    "Accepted formats: JPEG, PNG, WEBP."
                ),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    return file, [], None


def _apply_queryset_filters(qs, params) -> "QuerySet[NIDRecord]":
    """
    Apply standard list-view query parameters to *qs*.

    Handles: ``search``, ``status``, ``blood_group``, ``date_from``,
    ``date_to``, ``ordering``.  Invalid values are silently ignored so
    that a single bad filter never breaks a GET request.

    Args:
        qs:     Base ``NIDRecord`` queryset.
        params: ``request.query_params`` dict-like mapping.

    Returns:
        Filtered and ordered queryset.
    """
    search = params.get("search", "").strip()
    if search:
        qs = qs.filter(
            Q(name_english__icontains=search)
            | Q(name_bangla__icontains=search)
            | Q(nid_number__icontains=search)
        )

    status_filter = params.get("status", "").strip()
    if status_filter:
        qs = qs.filter(processing_status=status_filter)

    blood_group = params.get("blood_group", "").strip()
    if blood_group:
        qs = qs.filter(blood_group=blood_group)

    date_from = params.get("date_from", "").strip()
    if date_from:
        try:
            qs = qs.filter(created_at__date__gte=date_from)
        except Exception:
            logger.debug(
                "_apply_queryset_filters: ignored invalid date_from=%r", date_from
            )

    date_to = params.get("date_to", "").strip()
    if date_to:
        try:
            qs = qs.filter(created_at__date__lte=date_to)
        except Exception:
            logger.debug(
                "_apply_queryset_filters: ignored invalid date_to=%r", date_to
            )

    ordering = params.get("ordering", "-created_at").strip()
    if ordering in _ALLOWED_ORDER_FIELDS:
        qs = qs.order_by(ordering)
    else:
        qs = qs.order_by("-created_at")

    return qs


# ---------------------------------------------------------------------------
# View 1 — Upload + OCR
# ---------------------------------------------------------------------------

class NIDUploadView(APIView):
    """
    POST /api/upload/

    Accepts a multipart form with an ``image`` field, runs the full
    preprocessing + OCR pipeline synchronously, and persists the result.

    The record is always saved — even when OCR fails — so the uploaded
    image is never lost.  Partial OCR results are stored in the DB and
    returned with ``success: false``.

    Response (201 Created)::

        {
          "success": true,
          "message": "NID card processed successfully",
          "data":    { ...NIDRecordSerializer fields... },
          "warnings": []
        }

    On OCR failure the HTTP status is 200 (the *upload* succeeded)::

        {
          "success": false,
          "message": "NID image uploaded but OCR processing failed.",
          "data":    { ...partial record... },
          "warnings": ["OCR error: ..."]
        }
    """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Handle NID card image upload and synchronous OCR extraction."""
        warnings: list[str] = []

        # 1. Validate the uploaded file ----------------------------------------
        upload_file, file_warnings, error_response = _validate_upload(request)
        warnings.extend(file_warnings)
        if error_response is not None:
            return error_response

        # 2. Persist a PENDING record so the image is never lost ---------------
        record = NIDRecord.objects.create(
            uploaded_image=upload_file,
            processing_status=NIDRecord.ProcessingStatus.PENDING,
        )
        logger.info(
            "NIDUploadView: saved record #%d (%s), starting OCR",
            record.id, upload_file.name,
        )

        # 3. Mark PROCESSING ---------------------------------------------------
        record.processing_status = NIDRecord.ProcessingStatus.PROCESSING
        record.save(update_fields=["processing_status", "updated_at"])

        # 4. Preprocess + OCR --------------------------------------------------
        ocr_result: Optional[dict] = None
        try:
            ocr_result = NIDOCRService().extract_text(record.uploaded_image.path)

            fields          = ocr_result["fields"]
            valid_bg        = {c[0] for c in NIDRecord.BloodGroup.choices}
            raw_bg          = fields.get("blood_group", "")

            record.name_bangla        = fields.get("name_bangla", "")
            record.name_english       = fields.get("name_english", "")
            record.father_name_bangla = fields.get("father_name_bangla", "")
            record.mother_name_bangla = fields.get("mother_name_bangla", "")
            record.date_of_birth      = fields.get("date_of_birth", "")
            record.nid_number         = fields.get("nid_number", "")
            record.address_bangla     = fields.get("address_bangla", "")
            record.blood_group        = raw_bg if raw_bg in valid_bg else ""
            record.raw_ocr_text       = ocr_result["raw_text"]
            record.ocr_confidence     = ocr_result["confidence"]
            record.processing_time_ms = ocr_result["processing_time_ms"]
            record.processing_status  = NIDRecord.ProcessingStatus.SUCCESS
            record.error_message      = ""
            record.save()

            logger.info(
                "NIDUploadView: record #%d SUCCESS  nid=%s  confidence=%.2f  %dms",
                record.id,
                record.nid_number or "(not detected)",
                ocr_result["confidence"],
                ocr_result["processing_time_ms"],
            )

        except (PreprocessingError, OCRTimeoutError, OCRServiceError) as exc:
            logger.error(
                "NIDUploadView: record #%d FAILED — %s: %s",
                record.id, type(exc).__name__, exc,
            )
            record.processing_status = NIDRecord.ProcessingStatus.FAILED
            record.error_message     = str(exc)
            if ocr_result:
                record.raw_ocr_text       = ocr_result.get("raw_text", "")
                record.ocr_confidence     = ocr_result.get("confidence")
                record.processing_time_ms = ocr_result.get("processing_time_ms")
            record.save()

            # HTTP 200 — the *upload* worked; just the OCR step failed
            serializer = NIDRecordSerializer(record, context={"request": request})
            return Response(
                {
                    "success":  False,
                    "message":  "NID image uploaded but OCR processing failed.",
                    "data":     serializer.data,
                    "warnings": [f"OCR error: {exc}"],
                },
                status=status.HTTP_200_OK,
            )

        # 5. Quality warnings --------------------------------------------------
        if not record.nid_number:
            warnings.append(
                "NID number was not detected. Please verify and correct manually."
            )
        if not record.name_english:
            warnings.append("English name was not detected.")
        if not record.date_of_birth:
            warnings.append("Date of birth was not detected.")
        if (
            record.ocr_confidence is not None
            and record.ocr_confidence < _LOW_CONFIDENCE
        ):
            warnings.append(
                f"Low OCR confidence ({record.ocr_confidence:.1%}). "
                "Results may be inaccurate — please review all fields."
            )

        # 6. Duplicate NID warning (never blocks save) -------------------------
        if record.nid_number:
            prev = (
                NIDRecord.objects
                .filter(nid_number=record.nid_number, is_deleted=False)
                .exclude(pk=record.pk)
                .order_by("-created_at")
                .first()
            )
            if prev is not None:
                warnings.append(
                    f"Duplicate NID number detected. Previous record ID: {prev.pk}"
                )
                logger.info(
                    "NIDUploadView: duplicate NID %s — new #%d / previous #%d",
                    record.nid_number, record.pk, prev.pk,
                )

        serializer = NIDRecordSerializer(record, context={"request": request})
        return Response(
            {
                "success":  True,
                "message":  "NID card processed successfully",
                "data":     serializer.data,
                "warnings": warnings,
            },
            status=status.HTTP_201_CREATED,
        )


# ---------------------------------------------------------------------------
# View 2 — Record List
# ---------------------------------------------------------------------------

class NIDRecordListView(APIView):
    """
    GET /api/records/

    Returns a paginated, filterable list of non-deleted NID records.

    Query parameters
    ~~~~~~~~~~~~~~~~
    ``page``         int   default 1
    ``page_size``    int   default 20, max 100
    ``search``       str   searches name_english / name_bangla / nid_number
    ``status``       str   filter by processing_status
    ``blood_group``  str   filter by blood_group
    ``date_from``    ISO   lower bound on created_at date
    ``date_to``      ISO   upper bound on created_at date
    ``ordering``     str   one of created_at, -created_at, name_english,
                           -name_english, nid_number, -nid_number
    """

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Return paginated, filtered list of NID records."""
        qs = NIDRecord.objects.filter(is_deleted=False)
        qs = _apply_queryset_filters(qs, request.query_params)

        # Summary computed on the *filtered* queryset before slicing
        total_count = qs.count()
        summary = {
            "total_records": total_count,
            "successful": qs.filter(
                processing_status=NIDRecord.ProcessingStatus.SUCCESS
            ).count(),
            "failed": qs.filter(
                processing_status=NIDRecord.ProcessingStatus.FAILED
            ).count(),
        }

        # Pagination
        try:
            page = max(1, int(request.query_params.get("page", 1)))
        except (ValueError, TypeError):
            page = 1

        try:
            page_size = min(
                100, max(1, int(request.query_params.get("page_size", 20)))
            )
        except (ValueError, TypeError):
            page_size = 20

        total_pages  = max(1, (total_count + page_size - 1) // page_size)
        page         = min(page, total_pages)
        offset       = (page - 1) * page_size
        page_qs      = qs[offset: offset + page_size]

        serializer = NIDRecordListSerializer(
            page_qs, many=True, context={"request": request}
        )
        logger.debug(
            "NIDRecordListView: page=%d/%d  page_size=%d  total=%d",
            page, total_pages, page_size, total_count,
        )
        return Response(
            {
                "count":        total_count,
                "total_pages":  total_pages,
                "current_page": page,
                "page_size":    page_size,
                "results":      serializer.data,
                "summary":      summary,
            }
        )


# ---------------------------------------------------------------------------
# View 3 — Record Detail (GET / PUT / DELETE)
# ---------------------------------------------------------------------------

class NIDRecordDetailView(APIView):
    """
    GET    /api/records/<id>/  — retrieve full record
    PUT    /api/records/<id>/  — manually correct OCR text fields
    DELETE /api/records/<id>/  — soft-delete (sets is_deleted=True)
    """

    def _get_object(
        self, pk: int
    ) -> tuple[Optional[NIDRecord], Optional[Response]]:
        """
        Fetch a non-deleted NIDRecord or produce a 404 Response.

        Args:
            pk: Record primary key.

        Returns:
            ``(record, None)`` on success or ``(None, 404_response)``.
        """
        try:
            return NIDRecord.objects.get(pk=pk, is_deleted=False), None
        except NIDRecord.DoesNotExist:
            return None, Response(
                {
                    "success": False,
                    "error": f"NID record #{pk} not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

    def get(self, request: Request, pk: int, *args, **kwargs) -> Response:
        """Return the full detail of a single NID record."""
        record, err = self._get_object(pk)
        if err:
            return err

        serializer = NIDRecordSerializer(record, context={"request": request})
        return Response({"success": True, "data": serializer.data})

    def put(self, request: Request, pk: int, *args, **kwargs) -> Response:
        """
        Manually correct OCR-extracted text fields.

        Writeable: name_bangla, father_name_bangla, mother_name_bangla,
        address_bangla, name_english, date_of_birth, nid_number,
        blood_group.

        Blocked: uploaded_image, raw_ocr_text, ocr_confidence,
        processing_time_ms (enforced by NIDRecordUpdateSerializer).

        Returns a ``warnings`` list if the new nid_number collides with an
        existing record (not an error — duplicates are intentionally allowed).
        """
        record, err = self._get_object(pk)
        if err:
            return err

        serializer = NIDRecordUpdateSerializer(
            record,
            data=request.data,
            partial=True,
            context={"request": request},
        )
        if not serializer.is_valid():
            logger.info(
                "NIDRecordDetailView.put: validation errors for #%d — %s",
                pk, serializer.errors,
            )
            return Response(
                {"success": False, "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()
        logger.info("NIDRecordDetailView.put: updated record #%d", pk)

        # Duplicate NID warning (after save)
        warnings: list[str] = []
        new_nid = serializer.validated_data.get("nid_number", "")
        if new_nid:
            clash = (
                NIDRecord.objects
                .filter(nid_number=new_nid, is_deleted=False)
                .exclude(pk=pk)
                .first()
            )
            if clash:
                warnings.append(
                    f"NID number {new_nid} already exists on record #{clash.pk}."
                )

        return Response(
            {
                "success":  True,
                "message":  "Record updated successfully.",
                "data":     serializer.data,
                "warnings": warnings,
            }
        )

    def delete(self, request: Request, pk: int, *args, **kwargs) -> Response:
        """
        Soft-delete: set ``is_deleted=True``.

        The DB row and image file are retained; the record is hidden from
        all API responses and counts.
        """
        record, err = self._get_object(pk)
        if err:
            return err

        record.is_deleted = True
        record.save(update_fields=["is_deleted", "updated_at"])
        logger.info("NIDRecordDetailView.delete: soft-deleted record #%d", pk)

        return Response(
            {
                "success": True,
                "message": f"Record #{pk} has been deleted.",
            },
            status=status.HTTP_200_OK,
        )


# ---------------------------------------------------------------------------
# View 4 — Excel Export
# ---------------------------------------------------------------------------

class NIDExportView(APIView):
    """
    GET /api/export/

    Generates and streams a formatted ``.xlsx`` file.

    Query parameters
    ~~~~~~~~~~~~~~~~
    All standard list filters (search, status, blood_group, date_from,
    date_to, ordering) are supported.

    ``ids``  Comma-separated list of specific record IDs to export.
             Takes precedence over other filters when provided.
             Invalid IDs are silently skipped.

    Returns an empty workbook with column headers if no records match.
    """

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Build and return the Excel file as an attachment."""
        qs = NIDRecord.objects.filter(is_deleted=False)
        qs = _apply_queryset_filters(qs, request.query_params)

        # ``ids`` override
        ids_param = request.query_params.get("ids", "").strip()
        if ids_param:
            id_list: list[int] = []
            for raw_id in ids_param.split(","):
                try:
                    id_list.append(int(raw_id.strip()))
                except (ValueError, TypeError):
                    pass  # skip non-integer tokens silently

            qs = qs.filter(pk__in=id_list) if id_list else qs.none()
            logger.info(
                "NIDExportView: ids param resolved to %d valid IDs", len(id_list)
            )

        count = qs.count()
        logger.info("NIDExportView: exporting %d records", count)

        return NIDExcelExporter().export(qs)


# ---------------------------------------------------------------------------
# View 5 — Dashboard Statistics
# ---------------------------------------------------------------------------

class NIDStatsView(APIView):
    """
    GET /api/stats/

    Returns aggregate statistics for the admin dashboard.

    All figures exclude soft-deleted records.
    Time-based counts use the server timezone (Asia/Dhaka).
    """

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Compute and return dashboard statistics."""
        base_qs = NIDRecord.objects.filter(is_deleted=False)

        total      = base_qs.count()
        successful = base_qs.filter(
            processing_status=NIDRecord.ProcessingStatus.SUCCESS
        ).count()
        failed = base_qs.filter(
            processing_status=NIDRecord.ProcessingStatus.FAILED
        ).count()

        success_rate = round((successful / total * 100), 1) if total else 0.0

        # Average metrics — only meaningful on completed records
        agg = base_qs.filter(
            processing_status=NIDRecord.ProcessingStatus.SUCCESS
        ).aggregate(
            avg_confidence=Avg("ocr_confidence"),
            avg_time_ms=Avg("processing_time_ms"),
        )
        avg_confidence = round(agg["avg_confidence"] or 0.0, 4)
        avg_time_ms    = int(agg["avg_time_ms"] or 0)

        # Blood group distribution (successful records only, skip blanks)
        dist = (
            base_qs
            .filter(processing_status=NIDRecord.ProcessingStatus.SUCCESS)
            .exclude(blood_group="")
            .values("blood_group")
            .annotate(count=Count("id"))
            .order_by("blood_group")
        )
        blood_group_dist: dict[str, int] = {
            row["blood_group"]: row["count"] for row in dist
        }

        # Time-windowed counts
        now      = timezone.now()
        week_ago = now - timedelta(days=7)

        records_today = base_qs.filter(created_at__date=now.date()).count()
        records_week  = base_qs.filter(created_at__gte=week_ago).count()

        logger.debug(
            "NIDStatsView: total=%d success=%d failed=%d today=%d week=%d",
            total, successful, failed, records_today, records_week,
        )

        return Response(
            {
                "total_records":              total,
                "successful_extractions":     successful,
                "failed_extractions":         failed,
                "success_rate":               success_rate,
                "average_confidence":         avg_confidence,
                "average_processing_time_ms": avg_time_ms,
                "blood_group_distribution":   blood_group_dist,
                "records_today":              records_today,
                "records_this_week":          records_week,
            }
        )
