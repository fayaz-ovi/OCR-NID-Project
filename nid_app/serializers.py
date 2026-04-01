from rest_framework import serializers

from .models import NIDRecord

_OCR_READONLY_FIELDS = (
    'raw_ocr_text',
    'ocr_confidence',
    'processing_status',
    'error_message',
    'processing_time_ms',
    'created_at',
    'updated_at',
)


class NIDRecordSerializer(serializers.ModelSerializer):

    image_url = serializers.SerializerMethodField(read_only=True)
    is_complete = serializers.BooleanField(read_only=True)

    class Meta:
        model = NIDRecord
        fields = (
            'id',
            # Bangla
            'name_bangla',
            'father_name_bangla',
            'mother_name_bangla',
            'address_bangla',
            # English
            'name_english',
            'date_of_birth',
            'nid_number',
            'blood_group',
            # Image
            'uploaded_image',
            'image_url',
            # OCR metadata
            'raw_ocr_text',
            'ocr_confidence',
            'processing_status',
            'error_message',
            'processing_time_ms',
            # Derived
            'is_complete',
            # Timestamps
            'created_at',
            'updated_at',
        )
        read_only_fields = _OCR_READONLY_FIELDS
        extra_kwargs = {
            'uploaded_image': {'write_only': True},
        }

    def get_image_url(self, obj: NIDRecord) -> str | None:
        request = self.context.get('request')
        if obj.uploaded_image and request:
            return request.build_absolute_uri(obj.uploaded_image.url)
        if obj.uploaded_image:
            return obj.uploaded_image.url
        return None


class NIDRecordListSerializer(serializers.ModelSerializer):


    class Meta:
        model = NIDRecord
        fields = (
            'id',
            'name_english',
            'name_bangla',
            'nid_number',
            'blood_group',
            'processing_status',
            'created_at',
        )
        read_only_fields = fields


import re  # noqa: E402

_NID_RE = re.compile(r'^\d{10}$|^\d{17}$')
_VALID_BLOOD_GROUPS = {c[0] for c in NIDRecord.BloodGroup.choices}


class NIDRecordUpdateSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = NIDRecord
        fields = (
            'id',
       
            'name_bangla',
            'father_name_bangla',
            'mother_name_bangla',
            'address_bangla',
            'name_english',
            'date_of_birth',
            'nid_number',
            'blood_group',
    
            'processing_status',
            'ocr_confidence',
            'processing_time_ms',
            'created_at',
            'updated_at',
        )
        read_only_fields = (
            'processing_status',
            'ocr_confidence',
            'processing_time_ms',
            'created_at',
            'updated_at',
        )

    def validate_nid_number(self, value: str) -> str:
        if value and not _NID_RE.match(value):
            raise serializers.ValidationError(
                'NID number must be exactly 10 or 17 digits with no spaces or dashes.'
            )
        return value

    def validate_blood_group(self, value: str) -> str:
        
        if value and value not in _VALID_BLOOD_GROUPS:
            raise serializers.ValidationError(
                f'Blood group must be one of: {", ".join(sorted(_VALID_BLOOD_GROUPS))}.'
            )
        return value
