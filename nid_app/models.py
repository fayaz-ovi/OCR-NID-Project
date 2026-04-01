from django.db import models


class NIDRecord(models.Model):

    # ------------------------------------------------------------------
    # Blood group choices
    # ------------------------------------------------------------------
    class BloodGroup(models.TextChoices):
        A_POS  = 'A+',  'A+'
        A_NEG  = 'A-',  'A-'
        B_POS  = 'B+',  'B+'
        B_NEG  = 'B-',  'B-'
        AB_POS = 'AB+', 'AB+'
        AB_NEG = 'AB-', 'AB-'
        O_POS  = 'O+',  'O+'
        O_NEG  = 'O-',  'O-'

    # ------------------------------------------------------------------
    # Processing status choices
    # ------------------------------------------------------------------
    class ProcessingStatus(models.TextChoices):
        PENDING    = 'PENDING',    'Pending'
        PROCESSING = 'PROCESSING', 'Processing'
        SUCCESS    = 'SUCCESS',    'Success'
        FAILED     = 'FAILED',     'Failed'

    # ------------------------------------------------------------------
    # Bangla fields
    # ------------------------------------------------------------------
    name_bangla        = models.CharField(max_length=255, blank=True)
    father_name_bangla = models.CharField(max_length=255, blank=True)
    mother_name_bangla = models.CharField(max_length=255, blank=True)
    address_bangla     = models.TextField(blank=True)

    # ------------------------------------------------------------------
    # English fields
    # ------------------------------------------------------------------
    name_english  = models.CharField(max_length=255, blank=True)
    date_of_birth = models.CharField(
        max_length=50,
        blank=True,
        help_text='Format: DD Mon YYYY  e.g. 01 Jan 1990',
    )
    nid_number = models.CharField(
        max_length=30,
        blank=True,
        db_index=True,
        # NOT unique=True — same card may be uploaded multiple times;
        # deduplication lives in business logic, not the DB constraint.
        help_text='Bangladesh NID number (10 or 17 digits)',
    )
    blood_group = models.CharField(
        max_length=5,
        blank=True,
        choices=BloodGroup.choices,
    )

    # ------------------------------------------------------------------
    # Image & OCR metadata
    # ------------------------------------------------------------------
    uploaded_image = models.ImageField(
        upload_to='nid_uploads/%Y/%m/%d/',
        help_text='Original NID card image uploaded by the user',
    )
    raw_ocr_text = models.TextField(
        blank=True,
        help_text='Complete raw text dump from OCR engine (for debugging)',
    )
    ocr_confidence     = models.FloatField(null=True, blank=True)
    processing_status  = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.PENDING,
    )
    error_message      = models.TextField(blank=True)
    processing_time_ms = models.IntegerField(null=True, blank=True)

    # ------------------------------------------------------------------
    # Soft-delete flag
    # ------------------------------------------------------------------
    is_deleted = models.BooleanField(
        default=False,
        db_index=True,
        help_text='Soft-delete flag. Deleted records are hidden from API but kept in DB.',
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'NID Record'
        verbose_name_plural = 'NID Records'

    def __str__(self) -> str:
        return f"{self.name_english} | {self.nid_number} | {self.created_at}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_complete(self) -> bool:
        """True only when the three core English fields are all populated."""
        return bool(self.name_english and self.nid_number and self.date_of_birth)
