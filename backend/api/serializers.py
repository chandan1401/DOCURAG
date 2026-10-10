from rest_framework import serializers
from .models import Document, QueryLog


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        exclude = ['owner']


class QueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryLog
        exclude = ['owner']