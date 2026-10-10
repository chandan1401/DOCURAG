from django.db import models
from django.conf import settings
class Document(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    ]
    
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=50)
    pages = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    error = models.TextField(null=True, blank=True)
    chunk_counts = models.JSONField(default=dict)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='documents', null=True, blank=True,
    )
    def __str__(self):
        return self.filename

class QueryLog(models.Model):
    question = models.TextField()
    answer = models.TextField()
    answerable = models.BooleanField(default=True)
    confidence = models.FloatField()
    total_ms = models.IntegerField()
    timing = models.JSONField()
    sources = models.JSONField()
    agent_trace = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='queries', null=True, blank=True,
    )
    def __str__(self):
        return f"{self.question[:50]}... ({self.created_at})"
