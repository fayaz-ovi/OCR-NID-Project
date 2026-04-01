from django.contrib import admin
from django.utils.html import format_html

from .models import NIDRecord


@admin.register(NIDRecord)
class NIDRecordAdmin(admin.ModelAdmin):

    # ------------------------------------------------------------------
    # List view
    # ------------------------------------------------------------------
    list_display = (
        'name_english',
        'nid_number',
        'blood_group',
        'processing_status',
        'created_at',
    )
    list_filter  = ('processing_status', 'blood_group', 'created_at')
    search_fields = ('name_english', 'name_bangla', 'nid_number')

    # ------------------------------------------------------------------
    # Detail view layout
    # ------------------------------------------------------------------
    fieldsets = (
        ('English Fields', {
            'fields': (
                'name_english',
                'date_of_birth',
                'nid_number',
                'blood_group',
            ),
        }),
        ('Bangla Fields', {
            'fields': (
                'name_bangla',
                'father_name_bangla',
                'mother_name_bangla',
                'address_bangla',
            ),
        }),
        ('Image', {
            'fields': ('uploaded_image', 'image_preview'),
        }),
        ('OCR Metadata', {
            'fields': (
                'processing_status',
                'error_message',
                'raw_ocr_text',
                'ocr_confidence',
                'processing_time_ms',
            ),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    readonly_fields = (
        'raw_ocr_text',
        'ocr_confidence',
        'processing_time_ms',
        'created_at',
        'updated_at',
        'uploaded_image',   # prevent accidental overwrites of source images
        'image_preview',
    )

    # ------------------------------------------------------------------
    # Custom field: image thumbnail in detail view
    # ------------------------------------------------------------------
    def image_preview(self, obj: NIDRecord):
        if obj.uploaded_image:
            return format_html(
                '<img src="{}" style="max-width:480px; max-height:300px; '
                'border:1px solid #ccc; border-radius:4px;" />',
                obj.uploaded_image.url,
            )
        return '— no image —'

    image_preview.short_description = 'Preview'
