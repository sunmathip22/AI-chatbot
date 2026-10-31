import os
import sys
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import rag


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        rag._store.clear()
        rag.clear_conversation_history()

    def tearDown(self):
        rag._store.clear()
        rag.clear_conversation_history()

    def test_document_store_indexing_and_search(self):
        """Test BM25 indexing and similarity search."""
        chunks = [
            rag.DocChunk(text="Python is a high-level general-purpose programming language.", source="python_intro.txt", page=0),
            rag.DocChunk(text="Docker packages software into standardized units called containers.", source="docker_intro.txt", page=1),
            rag.DocChunk(text="Gemma is a family of lightweight open models built by Google.", source="gemma_intro.txt", page=0)
        ]
        rag._store.add_chunks("batch1", chunks)
        self.assertEqual(rag._store.count(), 3)

        results = rag._store.similarity_search("What is Python?", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].text)
        self.assertEqual(results[0].source, "python_intro.txt")

    def test_process_text_document(self):
        """Test processing a temporary document file."""
        sample_path = os.path.join(BASE_DIR, "sample_test.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Artificial intelligence is transforming modern software engineering. "
                    "Retrieval augmented generation connects LLMs to local data.")
        try:
            chunks_created = rag.process_document(sample_path)
            self.assertGreater(chunks_created, 0)
            self.assertEqual(rag._store.count(), chunks_created)

            # Test similarity search on the indexed document
            results = rag._store.similarity_search("retrieval augmented generation", k=1)
            self.assertTrue(len(results) > 0)
            self.assertIn("Retrieval augmented generation", results[0].text)
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)

    def test_language_detection(self):
        """Test accurate detection of English, Tamil (தமிழ் script), and Thanglish."""
        # 1. English
        self.assertEqual(rag.detect_language_style("Hi, how are you?"), "english")
        self.assertEqual(rag.detect_language_style("Which port does the server run on?"), "english")
        self.assertEqual(rag.detect_language_style("Please explain the installation steps."), "english")

        # 2. Tamil script
        self.assertEqual(rag.detect_language_style("வணக்கம், இந்த project எதுக்கு?"), "tamil")
        self.assertEqual(rag.detect_language_style("இந்த project எந்த port-ல் இயங்குகிறது?"), "tamil")
        self.assertEqual(rag.detect_language_style("நன்றி"), "tamil")

        # 3. Thanglish
        self.assertEqual(rag.detect_language_style("indha project edhuku use aagum?"), "thanglish")
        self.assertEqual(rag.detect_language_style("nee epdi oru project pannanumnu sollu"), "thanglish")
        self.assertEqual(rag.detect_language_style("vanakkam bro, project pathi sollunga"), "thanglish")
        self.assertEqual(rag.detect_language_style("server entha port-la run aagum?"), "thanglish")

    def test_spelling_variation_in_thanglish(self):
        """Test that natural spelling variations and suffixes in Thanglish are recognized and retrieve chunks."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="The project server runs on port 8000 using FastAPI and uvicorn.", source="doc.txt", page=0),
            rag.DocChunk(text="To install dependencies, run pip install -r requirements.txt in the backend folder.", source="doc.txt", page=1)
        ])

        variations = [
            ("server entha portla run aagum?", "port 8000"),
            ("port number enna bro?", "port 8000"),
            ("server yentha portla run pannalam?", "port 8000"),
            ("eppadi install pandrathu?", "pip install")
        ]
        for query, expected_text in variations:
            self.assertEqual(rag.detect_language_style(query), "thanglish")
            results = rag._store.similarity_search(query, k=1)
            self.assertGreater(len(results), 0, f"Failed retrieval for variation: {query}")
            self.assertIn(expected_text, results[0].text)

    def test_mixed_tamil_english_technical_question(self):
        """Test mixed Tamil-English questions with case markers like port-ல் or port-la."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="The project server runs on port 8000 using FastAPI.", source="doc.txt", page=0)
        ])

        # Tamil script with mixed English technical term 'port-ல்'
        tamil_query = "இந்த project எந்த port-ல் இயங்குகிறது?"
        self.assertEqual(rag.detect_language_style(tamil_query), "tamil")
        res_tamil = rag._store.similarity_search(tamil_query, k=1)
        self.assertEqual(len(res_tamil), 1)
        self.assertIn("port 8000", res_tamil[0].text)

        # Thanglish with mixed technical term 'port-la'
        thang_query = "server entha port-la run aagum?"
        self.assertEqual(rag.detect_language_style(thang_query), "thanglish")
        res_thang = rag._store.similarity_search(thang_query, k=1)
        self.assertEqual(len(res_thang), 1)
        self.assertIn("port 8000", res_thang[0].text)

    def test_question_present_in_documentation_all_three_languages(self):
        """Test asking about information present in documentation in English, Tamil, and Thanglish."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="The project server runs on port 8000 using FastAPI.", source="doc.txt", page=0)
        ])

        # English
        with patch("rag._invoke_ollama", return_value=("The server runs on port 8000 using FastAPI.", None)):
            res_en = rag.ask_question("Which port does the server run on?")
            self.assertEqual(res_en["mode"], "rag")
            self.assertIn("8000", res_en["answer"])
            self.assertEqual(res_en["sources"], ["doc.txt (page 1)"])

        # Tamil script
        with patch("rag._invoke_ollama", return_value=("இந்த சேவையகம் போர்ட் 8000-ல் FastAPI மூலம் இயங்குகிறது.", None)):
            res_ta = rag.ask_question("இந்த project எந்த port-ல் இயங்குகிறது?")
            self.assertEqual(res_ta["mode"], "rag")
            self.assertIn("8000", res_ta["answer"])
            self.assertEqual(res_ta["sources"], ["doc.txt (page 1)"])

        # Thanglish
        with patch("rag._invoke_ollama", return_value=("Indha server port 8000-la FastAPI use panni run aagudhu.", None)):
            res_th = rag.ask_question("server entha port-la run aagum?")
            self.assertEqual(res_th["mode"], "rag")
            self.assertIn("8000", res_th["answer"])
            self.assertEqual(res_th["sources"], ["doc.txt (page 1)"])

    def test_question_absent_from_documentation_all_three_languages(self):
        """Test that questions absent from documentation return honest missing-information responses in all 3 styles."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="The project server runs on port 8000 using FastAPI.", source="doc.txt", page=0)
        ])

        # 1. English absent question
        with patch("rag._invoke_ollama", return_value=("I couldn't find information regarding Kubernetes deployment in the available project documentation. Please upload the relevant document.", None)):
            res_en = rag.ask_question("How to deploy to Kubernetes on AWS EKS?")
            self.assertEqual(res_en["mode"], "rag")
            self.assertTrue(any(w in res_en["answer"].lower() for w in ["couldn't find", "not available", "don't have"]))
            self.assertEqual(res_en["sources"], [])

        # 2. Tamil script absent question
        with patch("rag._invoke_ollama", return_value=("கிடைக்கக்கூடிய project documentation-ல் அந்தத் தகவல் கிடைக்கவில்லை. சம்பந்தப்பட்ட document-ஐ upload செய்யுங்கள்.", None)):
            res_ta = rag.ask_question("குபெர்னெட்டீஸில் எவ்வாறு இந்த திட்டத்தை நிறுவுவது?")
            self.assertEqual(res_ta["mode"], "rag")
            self.assertIn("தகவல் கிடைக்கவில்லை", res_ta["answer"])
            self.assertEqual(res_ta["sources"], [])

        # 3. Thanglish absent question
        with patch("rag._invoke_ollama", return_value=("Available project documentation-la andha information kidaikkala bro. Relevant document-a upload pannunga.", None)):
            res_th = rag.ask_question("kubernetes-la epdi deploy pandradhu bro?")
            self.assertEqual(res_th["mode"], "rag")
            self.assertIn("kidaikkala", res_th["answer"].lower())
            self.assertEqual(res_th["sources"], [])

    def test_follow_up_question_referring_to_previous_answer(self):
        """Test that follow-up questions use conversation history and recheck documentation."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="The project backend runs on port 8000 with FastAPI. It uses ChromaDB for document storage.", source="doc.txt", page=0)
        ])

        # Turn 1
        with patch("rag._invoke_ollama", return_value=("The server runs on port 8000.", None)):
            res1 = rag.ask_question("Which port does the server run on?")
            self.assertEqual(res1["mode"], "rag")
            self.assertIn("8000", res1["answer"])

        # Turn 2: Follow-up referring to 'it'
        with patch("rag._invoke_ollama", return_value=("It uses ChromaDB for document storage.", None)):
            history = [
                {"role": "user", "content": "Which port does the server run on?"},
                {"role": "assistant", "content": "The server runs on port 8000."}
            ]
            res2 = rag.ask_question("What storage does it use?", history=history)
            self.assertEqual(res2["mode"], "rag")
            self.assertIn("ChromaDB", res2["answer"])
            self.assertEqual(res2["sources"], ["doc.txt (page 1)"])

    def test_document_retrieval_failure_handling(self):
        """Test that if document retrieval fails, system explains the error and suggests retrying."""
        rag._store.add_chunks("doc.txt", [
            rag.DocChunk(text="Sample content.", source="doc.txt", page=0)
        ])

        # Simulate retrieval error
        with patch.object(rag._store, "search_with_status", return_value=([], 0.0, "Corrupted index file")):
            # English query
            res_en = rag.ask_question("What is the port?")
            self.assertIn("could not be accessed due to a retrieval error", res_en["answer"])
            self.assertIn("retry", res_en["answer"].lower())

            # Tamil query
            res_ta = rag.ask_question("இந்த project எதுக்கு?")
            self.assertIn("அணுக முடியவில்லை", res_ta["answer"])

            # Thanglish query
            res_th = rag.ask_question("indha project edhuku use aagum?")
            self.assertIn("mudiyala", res_th["answer"].lower())

    def test_ollama_status_check(self):
        """Test check_ollama_status against local Ollama service."""
        status = rag.check_ollama_status()
        self.assertIn("online", status)
        self.assertIn("models", status)
        self.assertTrue(isinstance(status["models"], list))

    def test_active_model_logic_exact_and_fallback(self):
        """Test model matching: exact, prefix, fallback, and empty list."""
        with patch("rag.check_ollama_status", return_value={"online": True, "models": []}):
            model, available = rag.get_active_model()
            self.assertFalse(available)

        with patch("rag.check_ollama_status", return_value={"online": True, "models": ["gemma3n:e2b"]}):
            model, available = rag.get_active_model()
            self.assertTrue(available)
            self.assertEqual(model, "gemma3n:e2b")

        with patch("rag.check_ollama_status", return_value={"online": True, "models": ["gemma3n"]}):
            with patch.dict(os.environ, {"MODEL_NAME": "gemma3n"}):
                model, available = rag.get_active_model()
                self.assertTrue(available)

        with patch("rag.check_ollama_status", return_value={"online": True, "models": ["mistral:latest", "llama3:latest"]}):
            with patch.dict(os.environ, {"MODEL_NAME": "gemma3n:e2b"}):
                model, available = rag.get_active_model()
                self.assertTrue(available)
                self.assertEqual(model, "mistral:latest")

    def test_direct_mode_when_no_models(self):
        """Test ask_question in direct mode when 0 models are installed."""
        with patch("rag.check_ollama_status", return_value={"online": True, "models": [], "error": None}):
            res = rag.ask_question("What is machine learning?")
            self.assertEqual(res["mode"], "direct")
            self.assertIn("DIRECT MODE", res["answer"])
            self.assertIn("ollama pull", res["answer"])

    def test_groq_auth_failure_handling(self):
        """Test that Groq auth failure is properly reported."""
        http_err = urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"error":{"message":"Invalid API Key"}}')
        )
        with patch.dict(os.environ, {"GROQ_API_KEY": "invalid_key"}):
            with patch("urllib.request.urlopen", side_effect=http_err):
                res, err = rag._invoke_groq("Hello")
                self.assertIsNone(res)
                self.assertIsNotNone(err)
                self.assertIn("401 Unauthorized", err)

    def test_system_status_endpoint(self):
        """Test get_system_status structure and fields."""
        status = rag.get_system_status()
        self.assertEqual(status["status"], "ok")
        self.assertIn("mode", status)
        self.assertIn("chunks_indexed", status)
        self.assertIn("files", status)
        self.assertIn("model_configured", status)
        self.assertIn("ollama_base_url", status)
        self.assertIn("ollama_online", status)
        self.assertIn("available_models", status)
        self.assertIn("groq_fallback_active", status)
        self.assertIn("project_name", status)
        self.assertIn("model_type", status)
        self.assertIn("SLM", status["model_type"])

    def test_slm_identity_response_in_multiple_languages(self):
        """Test that identity questions deterministically identify the model as an SLM in user's language."""
        with patch.dict(os.environ, {"PROJECT_NAME": "AlphaBot Docs"}):
            # English
            res_en = rag.ask_question("Who are you?")
            self.assertIn("Small Language Model", res_en["answer"])
            self.assertIn("SLM", res_en["answer"])
            self.assertIn("AlphaBot Docs", res_en["answer"])

            # Tamil
            res_ta = rag.ask_question("நீங்கள் யார்?")
            self.assertIn("Small Language Model", res_ta["answer"])
            self.assertIn("AlphaBot Docs", res_ta["answer"])

            # Thanglish
            res_th = rag.ask_question("nee yaaru?")
            self.assertIn("Small Language Model", res_th["answer"])
            self.assertIn("AlphaBot Docs", res_th["answer"])

    def test_greetings_in_multiple_languages(self):
        """Test that greetings respond in the user's detected language and style."""
        # English greeting
        res_en = rag.ask_question("Hi, how are you?")
        self.assertIn("Hello! I'm doing well", res_en["answer"])

        # Tamil greeting
        res_ta = rag.ask_question("வணக்கம்")
        self.assertIn("வணக்கம்!", res_ta["answer"])

        # Thanglish greeting
        res_th = rag.ask_question("vanakkam bro")
        self.assertIn("Vanakkam!", res_th["answer"])

    def test_req1_normal_chat_routing_direct_mode(self):
        """Req 1: Verify normal messages ('Hi', 'Hello', 'Good morning') go directly to Gemma 3n E2B without retrieval."""
        rag._store.add_chunks("project_doc.txt", [
            rag.DocChunk(text="The project server uses FastAPI.", source="project_doc.txt", page=0)
        ])
        
        # Test detection
        self.assertTrue(rag.is_general_conversation("Good morning"))
        self.assertTrue(rag.is_general_conversation("Hello there!"))
        self.assertTrue(rag.is_general_conversation("Hi"))
        self.assertTrue(rag.is_general_conversation("how are you doing?"))
        
        # With active Ollama model, verify direct path execution
        with patch("rag._invoke_ollama", return_value=("Good morning! How can I assist you today?", None)) as mock_ollama:
            res = rag.ask_question("Good morning")
            self.assertEqual(res["mode"], "direct")
            self.assertEqual(res["sources"], [])
            call_prompt = mock_ollama.call_args[0][0]
            self.assertNotIn("PROJECT DOCUMENTATION CONTEXT", call_prompt)

    def test_req2_document_processing_pdf_and_txt(self):
        """Req 2: Verify PDF and TXT text extraction, chunking, and source/page metadata."""
        # 1. TXT Document
        txt_path = os.path.join(BASE_DIR, "req_test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("FastAPI is a modern web framework for building APIs with Python 3.8+.")
        try:
            chunks_txt = rag.process_document(txt_path)
            self.assertGreater(chunks_txt, 0)
            txt_chunks = [c for c in rag._store.chunks if c.source == "req_test.txt"]
            self.assertTrue(len(txt_chunks) > 0)
            self.assertEqual(txt_chunks[0].page, 0)
        finally:
            if os.path.exists(txt_path):
                os.remove(txt_path)

        # 2. PDF Document
        pdf_path = os.path.join(BASE_DIR, "req_test.pdf")
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
            b"4 0 obj << /Length 115 >> stream\n"
            b"BT /F1 12 Tf 100 700 Td (This is a sample PDF document for RAG testing. It verifies that readable text is extracted properly from PDF files.) Tj ET\n"
            b"endstream endobj\n"
            b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"xref\n"
            b"0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000244 00000 n \n"
            b"0000000410 00000 n \n"
            b"trailer << /Size 6 /Root 1 0 R >>\n"
            b"startxref\n"
            b"486\n"
            b"%%EOF\n"
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        try:
            chunks_pdf = rag.process_document(pdf_path)
            self.assertGreater(chunks_pdf, 0)
            pdf_chunks = [c for c in rag._store.chunks if c.source == "req_test.pdf"]
            self.assertTrue(len(pdf_chunks) > 0)
            self.assertIn("sample PDF document", pdf_chunks[0].text)
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_req3_embeddinggemma2_dense_embeddings(self):
        """Req 3: Verify EmbeddingGemma 2 generates 768-dimensional dense embeddings."""
        embs, err = rag.generate_embeddings(["This is a test sentence for EmbeddingGemma 2."])
        self.assertIsNone(err)
        self.assertIsNotNone(embs)
        self.assertEqual(len(embs), 1)
        self.assertEqual(len(embs[0]), 768)
        self.assertIsInstance(embs[0][0], float)

    def test_req4_dense_vector_similarity_retrieval(self):
        """Req 4: Verify vector_search retrieves relevant chunks using cosine similarity."""
        vec1 = [0.0] * 768
        vec1[0] = 1.0
        vec2 = [0.0] * 768
        vec2[1] = 1.0

        c1 = rag.DocChunk(text="Chunk 1: Machine learning", source="doc1.txt", page=0, embedding=vec1)
        c2 = rag.DocChunk(text="Chunk 2: Web development", source="doc2.txt", page=0, embedding=vec2)
        rag._store.add_chunks("batch_v", [c1, c2])

        query_vec = [0.0] * 768
        query_vec[0] = 0.95
        query_vec[1] = 0.05

        matched_chunks, best_sim, err = rag._store.vector_search(query_vec, k=1, threshold=0.5)
        self.assertIsNone(err)
        self.assertEqual(len(matched_chunks), 1)
        self.assertEqual(matched_chunks[0].text, "Chunk 1: Machine learning")
        self.assertGreater(best_sim, 0.9)

    def test_req5_gemma3n_grounded_answer_generation(self):
        """Req 5: Verify Gemma 3n E2B generates answers grounded with source and page references."""
        rag._store.add_chunks("arch.txt", [
            rag.DocChunk(text="Antigravity system architecture consists of a neural pipeline on port 9090.", source="arch.txt", page=2)
        ])

        with patch("rag._invoke_ollama", return_value=("The system architecture consists of a neural pipeline on port 9090.", None)):
            res = rag.ask_question("What does the system architecture consist of?")
            self.assertEqual(res["mode"], "rag")
            self.assertIn("9090", res["answer"])
            self.assertEqual(res["sources"], ["arch.txt (page 3)"])

    def test_req6_honest_fallback_when_unanswerable(self):
        """Req 6: Verify honest fallback without hallucination when question cannot be answered from document."""
        rag._store.add_chunks("known.txt", [
            rag.DocChunk(text="This project is written in Python.", source="known.txt", page=0)
        ])

        fallback_msg = "I couldn't find information regarding the database password in the available project documentation."
        with patch("rag._invoke_ollama", return_value=(fallback_msg, None)):
            res = rag.ask_question("What is the production database root password?")
            self.assertEqual(res["mode"], "rag")
            self.assertIn("couldn't find", res["answer"].lower())
            self.assertEqual(res["sources"], [])

    def test_req7_multi_document_isolation_and_sources(self):
        """Req 7: Verify multiple documents can be indexed and queried with context isolation."""
        c_a = rag.DocChunk(text="App A uses SQLite database.", source="app_a.txt", page=0)
        c_b = rag.DocChunk(text="App B uses PostgreSQL database.", source="app_b.txt", page=1)
        rag._store.add_chunks("app_a.txt", [c_a])
        rag._store.add_chunks("app_b.txt", [c_b])
        self.assertEqual(rag._store.count(), 2)

        with patch("rag._invoke_ollama", return_value=("App B uses PostgreSQL database.", None)):
            res = rag.ask_question("Which database is used?", selected_doc="app_b.txt")
            self.assertEqual(res["mode"], "rag")
            self.assertEqual(res["sources"], ["app_b.txt (page 2)"])

    def test_req8_system_status_and_zero_cost_local_runtime(self):
        """Req 8: Verify system status reports EmbeddingGemma 2, Gemma 3n E2B, and local ₹0 inference."""
        status = rag.get_system_status()
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["embedding_model"], "embeddinggemma-2")
        self.assertIn("gemma3n", status["model_configured"].lower())
        self.assertTrue(status["ollama_online"])
        self.assertFalse(status["groq_fallback_active"])


if __name__ == "__main__":
    unittest.main()

