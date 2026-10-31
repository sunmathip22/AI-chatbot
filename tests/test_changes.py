import json
import os
import sys
import unittest
from unittest.mock import patch

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import rag


class TestRAGChanges(unittest.TestCase):

    def setUp(self):
        rag.clear_query_cache()
        rag._store.clear()
        rag.clear_conversation_history()

    def tearDown(self):
        rag.clear_query_cache()
        rag._store.clear()
        rag.clear_conversation_history()

    # ──────────────────────────────────────────────────────────────────────────
    # CHANGE 1: Instant Greeting & Small Talk (< 100 ms, No LLM call)
    # ──────────────────────────────────────────────────────────────────────────

    def test_change1_instant_greetings_and_smalltalk(self):
        """Verify greetings, thanks, bye, ok return fixed responses without LLM/store calls."""
        with patch("rag._invoke_ollama") as mock_ollama:
            # 1. English greeting
            res1 = rag.ask_question("Hello")
            self.assertEqual(res1["answer"], "Hello! I'm your chatbot assistant. How can I help you?")
            self.assertFalse(res1["escalate"])
            self.assertEqual(res1["mode"], "direct")

            # 2. Tanglish greeting
            res2 = rag.ask_question("vanakkam bro")
            self.assertIn("Vanakkam!", res2["answer"])
            self.assertFalse(res2["escalate"])

            # 3. Thanks
            res3 = rag.ask_question("Thanks a lot")
            self.assertEqual(res3["answer"], "You're welcome!")
            self.assertFalse(res3["escalate"])

            # 4. Bye
            res4 = rag.ask_question("bye bye")
            self.assertEqual(res4["answer"], "Bye, have a nice day!")

            # 5. Ok
            res5 = rag.ask_question("ok")
            self.assertEqual(res5["answer"], "Alright! Let me know if you need anything else.")

            # Model must NOT have been called for any of these
            mock_ollama.assert_not_called()

    def test_change1_instant_streaming_greeting(self):
        """Verify stream_ask_question yields immediate greeting tokens without calling model."""
        tokens = []
        done_data = None
        for chunk in rag.stream_ask_question("Hi"):
            if chunk.startswith("data: "):
                data = json.loads(chunk[6:].strip())
                if "token" in data:
                    tokens.append(data["token"])
                if data.get("done"):
                    done_data = data
        self.assertEqual("".join(tokens), "Hello! I'm your chatbot assistant. How can I help you?")
        self.assertFalse(done_data["escalate"])

    # ──────────────────────────────────────────────────────────────────────────
    # CHANGE 2: Threshold-based Routing & Business Questions
    # ──────────────────────────────────────────────────────────────────────────

    def test_change2_business_question_above_threshold_rag_answer(self):
        """Score above threshold -> RAG answer from retrieved chunks."""
        rag._store.add_chunks("herbal_shop.txt", [
            rag.DocChunk(
                text="Neem Soap (100 g) is Rs. 59. Murungai keerai powder is Rs. 89. Shop timing: 9:00 AM to 8:00 PM.",
                source="herbal_shop.txt",
                page=0
            )
        ])

        with patch("rag._invoke_ollama", return_value=("Neem Soap is Rs. 59 and the shop is open from 9:00 AM to 8:00 PM.", None)):
            res = rag.ask_question("What is the price of Neem Soap?")
            self.assertEqual(res["mode"], "rag")
            self.assertFalse(res["escalate"])
            self.assertIn("59", res["answer"])
            self.assertIn("herbal_shop.txt (page 1)", res["sources"])

    def test_change2_business_question_below_threshold_escalates(self):
        """Score below threshold + business question -> Escalate message + escalate=true."""
        rag._store.add_chunks("herbal_shop.txt", [
            rag.DocChunk(
                text="Shop timing: Monday to Saturday 9:00 AM to 8:00 PM.",
                source="herbal_shop.txt",
                page=0
            )
        ])

        # Ask about a business product that is absent from documents
        res = rag.ask_question("What is the cost of Aloe Vera Gel?")
        self.assertEqual(res["answer"], "Intha information ennoda documents la illa. Owner kitta confirm panni solren.")
        self.assertTrue(res["escalate"])
        self.assertEqual(res["sources"], [])

    def test_change2_general_question_below_threshold_direct_model_call(self):
        """Score below threshold + general question -> Direct LLM without document context."""
        rag._store.add_chunks("herbal_shop.txt", [
            rag.DocChunk(
                text="Shop timing: Monday to Saturday 9:00 AM to 8:00 PM.",
                source="herbal_shop.txt",
                page=0
            )
        ])

        with patch("rag._invoke_ollama", return_value=("Photosynthesis is the process plants use to make food using sunlight.", None)) as mock_llm:
            res = rag.ask_question("What is photosynthesis?")
            self.assertEqual(res["mode"], "direct")
            self.assertFalse(res["escalate"])
            self.assertEqual(res["sources"], [])
            self.assertIn("Photosynthesis", res["answer"])
            call_prompt = mock_llm.call_args[0][0]
            self.assertNotIn("Verified Context:", call_prompt)

    # ──────────────────────────────────────────────────────────────────────────
    # CHANGE 3: Speed, Options, Cache & Environment Settings
    # ──────────────────────────────────────────────────────────────────────────

    def test_change3_in_memory_cache(self):
        """Verify identical repeated queries return from cache in < 1 ms."""
        rag._store.add_chunks("shop.txt", [
            rag.DocChunk(text="Herbal Tea price is Rs. 120.", source="shop.txt", page=0)
        ])

        with patch("rag._invoke_ollama", return_value=("Herbal Tea is priced at Rs. 120.", None)) as mock_llm:
            res1 = rag.ask_question("What is the rate of Herbal Tea?")
            self.assertEqual(mock_llm.call_count, 1)

            # Second call (same query with varied whitespace/casing)
            res2 = rag.ask_question("  what is the rate of herbal tea?  ")
            # Should be returned from cache, so mock_llm is NOT called again
            self.assertEqual(mock_llm.call_count, 1)
            self.assertEqual(res1["answer"], res2["answer"])

    def test_change3_ollama_options_passed(self):
        """Verify Ollama options (num_ctx=2048, num_predict=120, temperature=0.2, keep_alive=30m)."""
        self.assertEqual(rag.get_num_ctx(), 2048)
        self.assertEqual(rag.get_num_predict(), 120)
        self.assertEqual(rag.get_temperature(), 0.2)
        self.assertEqual(rag.get_keep_alive(), "30m")
        self.assertEqual(rag.get_top_k(), 3)
        self.assertEqual(rag.get_chunk_size(), 350)


if __name__ == "__main__":
    unittest.main()
