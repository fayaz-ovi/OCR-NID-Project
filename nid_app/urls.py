
from django.urls import path

from .views import (
    NIDExportView,
    NIDRecordDetailView,
    NIDRecordListView,
    NIDStatsView,
    NIDUploadView,
)

app_name = "nid_app"

urlpatterns = [
    # Upload & OCR
    path("upload/",             NIDUploadView.as_view(),       name="upload"),

    # Record CRUD
    path("records/",            NIDRecordListView.as_view(),   name="record-list"),
    path("records/<int:pk>/",   NIDRecordDetailView.as_view(), name="record-detail"),

    # Bulk operations
    path("export/",             NIDExportView.as_view(),        name="export"),
    path("stats/",              NIDStatsView.as_view(),         name="stats"),
]
