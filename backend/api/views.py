import logging
import mimetypes
import os
import threading

from django.apps import apps
from django.conf import settings
from django.core.files.storage import default_storage
from django.db import connection
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import Document, QueryLog
from .serializers import DocumentSerializer, QueryLogSerializer

logger = logging.getLogger(__name__)

IMAGE_TYPES = {'png', 'jpg', 'jpeg', 'webp'}


def get_pipeline():
    return apps.get_app_config('api').engine


def _my_documents(request):
    """Documents owned by the signed-in user. Every document lookup goes through this."""
    return Document.objects.filter(owner=request.user)


def _page_image_path(doc, page_num):
    """Rendered page image for a document (or the image itself for image uploads)."""
    pages_dir = os.path.join(settings.MEDIA_ROOT, 'pages')
    candidate = os.path.join(pages_dir, f'{doc.id}_p{page_num}.png')
    if os.path.exists(candidate):
        return candidate
    if (doc.file_type in IMAGE_TYPES and int(page_num) == 1
            and doc.file and os.path.exists(doc.file.path)):
        return doc.file.path
    return None


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def document_upload(request):
    file_obj = request.FILES.get('file')
    if not file_obj:
        return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file_obj.name)[1].lower()
    if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.webp']:
        return Response({'error': 'Invalid file extension'}, status=status.HTTP_400_BAD_REQUEST)

    if file_obj.size > 50 * 1024 * 1024:
        return Response({'error': 'File size exceeds 50MB'}, status=status.HTTP_400_BAD_REQUEST)

    doc = Document.objects.create(
        owner=request.user,
        filename=file_obj.name,
        file=file_obj,
        file_type=ext[1:] if ext else 'unknown',
        status='queued',
    )

    def background_ingest(doc_id):
        doc_instance = None
        try:
            pipeline = get_pipeline()
            doc_instance = Document.objects.get(id=doc_id)
            file_path = doc_instance.file.path

            doc_instance.status = 'processing'
            doc_instance.save()

            # ingest_file replaces any existing chunks for this document id
            result = pipeline.ingest_file(file_path, document_id=str(doc_id))
            total = result.get('chunks_indexed', 0)
            doc_instance.pages = result.get('pages')
            doc_instance.chunk_counts = {'total': total}
            if total == 0:
                doc_instance.status = 'failed'
                doc_instance.error = 'No text could be extracted (needs OCR or image captioning)'
            else:
                doc_instance.status = 'ready'
                doc_instance.error = None
        except Exception as e:
            logger.exception("Ingestion failed for document %s", doc_id)
            if doc_instance is None:
                doc_instance = Document.objects.filter(id=doc_id).first()
            if doc_instance is not None:
                doc_instance.status = 'failed'
                doc_instance.error = str(e)
        finally:
            try:
                if doc_instance is not None:
                    doc_instance.save()
            finally:
                connection.close()

    threading.Thread(target=background_ingest, args=(doc.id,), daemon=True).start()

    return Response(DocumentSerializer(doc).data, status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def document_list(request):
    docs = _my_documents(request).order_by('-uploaded_at')
    return Response(DocumentSerializer(docs, many=True).data)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def document_detail(request, pk):
    doc = get_object_or_404(_my_documents(request), pk=pk)
    if request.method == 'DELETE':
        try:
            get_pipeline().remove_document(str(doc.id))
        except Exception:
            logger.exception("Could not remove chunks for document %s", doc.id)
        if doc.file:
            default_storage.delete(doc.file.name)
        doc.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    return Response(DocumentSerializer(doc).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def document_page(request, pk, page_num):
    doc = get_object_or_404(_my_documents(request), pk=pk)
    path = _page_image_path(doc, page_num)
    if not path:
        return Response({'error': 'Page image not found'}, status=status.HTTP_404_NOT_FOUND)
    content_type = mimetypes.guess_type(path)[0] or 'image/png'
    response = FileResponse(open(path, 'rb'), content_type=content_type)
    response['Cache-Control'] = 'private, max-age=3600'
    return response


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def query_ask(request):
    data = request.data
    question = data.get('question', '').strip()
    requested_ids = data.get('document_ids', []) or []
    content_types = data.get('content_types', [])
    history = data.get('history', [])

    if not question:
        return Response({'error': 'Question is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Only this user's finished documents can ever be searched
    my_ready = list(_my_documents(request).filter(status='ready'))
    docs_by_id = {str(d.id): d for d in my_ready}

    clean_q = question.strip().lower().rstrip("!?. ")
    capability_phrases = ["what can you do", "who are you", "what are your features", "how does this work",
                          "how do you work", "help me", "what is docurag", "can you help"]
    greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}

    is_greeting = (
        clean_q in greeting_words
        or any(phrase in clean_q for phrase in capability_phrases)
        or (any(clean_q.startswith(w) for w in greeting_words) and len(clean_q.split()) <= 6)
    )

    if is_greeting:
        doc_count = len(my_ready)
        doc_names = ", ".join(f"`{d.filename}`" for d in my_ready[:3]) if doc_count else "None yet"
        if doc_count > 3:
            doc_names += f" and {doc_count - 3} more"

        greeting_answer = (
            f"**Hello! I am DocuRAG Studio**, your document analysis assistant.\n\n"
            f"Your library has **{doc_count} ready document(s)**: {doc_names}.\n\n"
            f"### What I can help you with:\n"
            f"* **Fact-checking and retrieval:** answers with source citations `[E#]` and page previews.\n"
            f"* **Glossary and concept definitions:** clarify terminology from your documents.\n\n"
            f"Ask a question or try one of the quick prompts below."
        )
        return Response({
            'id': 0,
            'answer': greeting_answer,
            'answerable': True,
            'confidence': 1.0,
            'total_ms': 45,
            'timing': {'total_ms': 45, 'stages': {'retrieval': 0, 'generation': 45}},
            'sources': [],
            'agent_trace': [],
        })

    # Never trust document ids from the browser: intersect with what this user owns.
    # No selection means "all of MY documents", never "everything in the index".
    if requested_ids:
        selected = sorted({str(i) for i in requested_ids} & set(docs_by_id), key=int)
    else:
        selected = sorted(docs_by_id, key=int)

    if not selected:
        return Response({
            'id': 0,
            'answer': "There are no ready documents to search. Upload a file in the Library first, "
                      "or wait for processing to finish.",
            'answerable': False,
            'confidence': 0.0,
            'total_ms': 0,
            'timing': {'total_ms': 0, 'stages': {}},
            'sources': [],
            'agent_trace': [],
        })

    instructions = None
    if history:
        history_blocks = []
        for turn in history[-4:]:
            if turn.get('user') and turn.get('assistant'):
                history_blocks.append(f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}")
        if history_blocks:
            instructions = "Consider the conversation history below for context:\n" + "\n".join(history_blocks)

    pipeline = get_pipeline()
    result = pipeline.ask(question, document_ids=selected, content_types=content_types,
                          instructions=instructions)

    formatted_sources = []
    for idx, s in enumerate(result.sources):
        s_data = s.model_dump() if hasattr(s, 'model_dump') else dict(s)
        ev = result.evidence[idx] if idx < len(result.evidence) else None
        doc = docs_by_id.get(str(ev.chunk.document_id)) if ev is not None else None
        page_num = s_data.get('page') or 1
        s_data['document_id'] = doc.id if doc else None
        s_data['filename'] = doc.filename if doc else None
        s_data['thumbnail_url'] = (
            f'/api/documents/{doc.id}/pages/{page_num}/'
            if doc and _page_image_path(doc, page_num) else None
        )
        formatted_sources.append(s_data)

    confidence_score = result.confidence.score if hasattr(result.confidence, 'score') else 0.0
    total_ms = result.timing.total_ms if hasattr(result.timing, 'total_ms') else 0
    timing_dict = result.timing.model_dump() if hasattr(result.timing, 'model_dump') else {}
    agent_trace_list = [t.model_dump() if hasattr(t, 'model_dump') else t for t in result.agent_trace]

    log = QueryLog.objects.create(
        owner=request.user,
        question=question,
        answer=result.answer,
        answerable=result.answerable,
        confidence=confidence_score,
        total_ms=total_ms,
        timing=timing_dict,
        sources=formatted_sources,
        agent_trace=agent_trace_list,
    )

    return Response({
        'id': log.id,
        'answer': log.answer,
        'answerable': log.answerable,
        'confidence': log.confidence,
        'total_ms': log.total_ms,
        'timing': log.timing,
        'sources': formatted_sources,
        'agent_trace': log.agent_trace,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def query_history(request):
    logs = QueryLog.objects.filter(owner=request.user).order_by('-created_at')[:50]
    return Response(QueryLogSerializer(logs, many=True).data)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    return Response({'status': 'ok'})