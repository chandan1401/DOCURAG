from django.apps import AppConfig
import docurag.pipeline as pipeline


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    engine = None

    def ready(self):
        if self.engine:
            return

        class PipelineWrapper:
            """Thin facade so the views only depend on ask / ingest_file / remove_document."""
            device = 'cpu'
            model_flags = {}

            def ask(self, *args, **kwargs):
                return pipeline.ask(*args, **kwargs)

            def ingest_file(self, *args, **kwargs):
                return pipeline.ingest_file(*args, **kwargs)

            def remove_document(self, document_id):
                return pipeline.remove_document(str(document_id))

            @property
            def total_chunks(self):
                try:
                    return len(pipeline.get_engine()["store"].chunks)
                except Exception:
                    return 0

        self.engine = PipelineWrapper()