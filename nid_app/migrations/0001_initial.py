import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='NIDRecord',
            fields=[
                (
                    'id',
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name='ID',
                    ),
                ),
                # ── Bangla fields ────────────────────────────────────────
                (
                    'name_bangla',
                    models.CharField(blank=True, max_length=255),
                ),
                (
                    'father_name_bangla',
                    models.CharField(blank=True, max_length=255),
                ),
                (
                    'mother_name_bangla',
                    models.CharField(blank=True, max_length=255),
                ),
                (
                    'address_bangla',
                    models.TextField(blank=True),
                ),
                # ── English fields ───────────────────────────────────────
                (
                    'name_english',
                    models.CharField(blank=True, max_length=255),
                ),
                (
                    'date_of_birth',
                    models.CharField(
                        blank=True,
                        help_text='Format: DD Mon YYYY  e.g. 01 Jan 1990',
                        max_length=50,
                    ),
                ),
                (
                    'nid_number',
                    models.CharField(
                        blank=True,
                        db_index=True,
                        help_text='Bangladesh NID number (10 or 17 digits)',
                        max_length=30,
                    ),
                ),
                (
                    'blood_group',
                    models.CharField(
                        blank=True,
                        choices=[
                            ('A+',  'A+'),
                            ('A-',  'A-'),
                            ('B+',  'B+'),
                            ('B-',  'B-'),
                            ('AB+', 'AB+'),
                            ('AB-', 'AB-'),
                            ('O+',  'O+'),
                            ('O-',  'O-'),
                        ],
                        max_length=5,
                    ),
                ),
                # ── Image & OCR metadata ─────────────────────────────────
                (
                    'uploaded_image',
                    models.ImageField(
                        help_text='Original NID card image uploaded by the user',
                        upload_to='nid_uploads/%Y/%m/%d/',
                    ),
                ),
                (
                    'raw_ocr_text',
                    models.TextField(
                        blank=True,
                        help_text='Complete raw text dump from OCR engine (for debugging)',
                    ),
                ),
                (
                    'ocr_confidence',
                    models.FloatField(blank=True, null=True),
                ),
                (
                    'processing_status',
                    models.CharField(
                        choices=[
                            ('PENDING',    'Pending'),
                            ('PROCESSING', 'Processing'),
                            ('SUCCESS',    'Success'),
                            ('FAILED',     'Failed'),
                        ],
                        default='PENDING',
                        max_length=20,
                    ),
                ),
                (
                    'error_message',
                    models.TextField(blank=True),
                ),
                (
                    'processing_time_ms',
                    models.IntegerField(blank=True, null=True),
                ),
                # ── Timestamps ───────────────────────────────────────────
                (
                    'created_at',
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    'updated_at',
                    models.DateTimeField(auto_now=True),
                ),
            ],
            options={
                'verbose_name': 'NID Record',
                'verbose_name_plural': 'NID Records',
                'ordering': ['-created_at'],
            },
        ),
    ]
