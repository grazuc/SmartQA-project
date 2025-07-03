import pytest
import tempfile
from pathlib import Path
from smartqa import DocumentProcessor, SmartQA, QAResult

class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_load_document_success(self):
        """Test successful document loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "Hello world!"
            f.write(test_content)
            f.flush()
            fname = f.name
        result = self.processor.load_document(Path(fname))
        assert result == test_content
        Path(fname).unlink()

    def test_load_document_empty_file(self):
        """Test loading empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("") 
            f.flush()
            fname = f.name
        with pytest.raises(ValueError, match="Document is empty"):
            self.processor.load_document(Path(fname))
        Path(fname).unlink()

    def test_chunk_document_basic(self):
        """Test chunking produces at least one chunk"""
        text = "Hello. This is a test. Another sentence. Goodbye. More text. Sophometrics is a great company."
        chunks = self.processor.chunk_document(text)
        assert len(chunks) > 0
        assert all(hasattr(c, 'page_content') for c in chunks)

class TestSmartQA:
    def setup_method(self):
        from unittest.mock import patch, MagicMock
        self.patcher_embeddings = patch('smartqa.HuggingFaceEmbeddings', autospec=True)
        self.patcher_pipeline = patch('smartqa.HuggingFacePipeline', autospec=True)
        self.patcher_prompt = patch('smartqa.PromptTemplate', autospec=True)
        self.mock_embed = self.patcher_embeddings.start()
        self.mock_pipe = self.patcher_pipeline.start()
        self.mock_prompt = self.patcher_prompt.start()
        self.qa = SmartQA()

    def teardown_method(self):
        self.patcher_embeddings.stop()
        self.patcher_pipeline.stop()
        self.patcher_prompt.stop()

    def test_init_smartqa(self):
        assert isinstance(self.qa, SmartQA)

    def test_index_document(self):
        from unittest.mock import MagicMock
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for chunking.")
            f.flush()
            fname = f.name
        embed_instance = self.mock_embed.return_value
        embed_instance.embed_documents.side_effect = lambda docs: [[0.0]*384 for _ in docs]
        n_chunks = self.qa.index_document(Path(fname))
        assert n_chunks > 0
        Path(fname).unlink()

    def test_qares_result_dataclass(self):
        r = QAResult("A", ["0"], 0.1, 0.8)
        assert r.answer == "A"
        assert r.citations == ["0"]
        assert r.latency == 0.1
        assert r.confidence == 0.8
