from django.urls import path
from . import views

urlpatterns = [
    path('documents/', views.document_list, name='document-list'),
    path('documents/upload/', views.document_upload, name='document-upload'),
    path('documents/<int:pk>/', views.document_detail, name='document-detail'),
    path('documents/<int:pk>/pages/<int:page_num>/', views.document_page, name='document-page'),
    path('query/', views.query_ask, name='query-ask'),
    path('history/', views.query_history, name='query-history'),
    path('health/', views.health_check, name='health-check'),
]
