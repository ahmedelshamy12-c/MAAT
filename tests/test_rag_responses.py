import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import RAG


def completion(answer):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )


class RagResponseTests(unittest.TestCase):
    def test_recent_history_keeps_the_latest_eight_valid_messages(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
            for index in range(10)
        ]

        recent = RAG._recent_history(history)

        self.assertEqual(len(recent), 8)
        self.assertEqual(recent[0]["content"], "message-2")
        self.assertEqual(recent[-1]["content"], "message-9")

    def test_law_excerpts_are_grouped_from_chroma_metadata(self):
        class FakeCollection:
            def count(self):
                return 4

            def get(self, **_kwargs):
                return {
                    "documents": [
                        "First civil excerpt.",
                        "Second civil excerpt.",
                        "First criminal excerpt.",
                        "Second criminal excerpt.",
                    ],
                    "metadatas": [
                        {"law_name": "Civil Code", "categories": "Civil"},
                        {"law_name": "Civil Code", "categories": "Civil"},
                        {"law_name": "Criminal Code", "categories": "Criminal"},
                        {"law_name": "Criminal Code", "categories": "Criminal"},
                    ],
                }

        with patch("RAG._get_collection", return_value=FakeCollection()):
            laws = RAG.get_law_excerpts(max_laws=2, chunks_per_law=2)

        self.assertEqual([law["law_name"] for law in laws], ["Civil Code", "Criminal Code"])
        self.assertEqual(laws[0]["excerpts"], ["First civil excerpt.", "Second civil excerpt."])

    def test_creator_question_returns_configured_identity_without_model_call(self):
        with patch.object(
            RAG.client.chat.completions,
            "create",
            side_effect=AssertionError("creator answer should be deterministic"),
        ):
            answer = RAG.generate_chat_response("Who built MAAT?")
            arabic_answer = RAG.generate_chat_response("من هو أحمد مدحت؟")
            nickname_answer = RAG.generate_chat_response("What is Ahmed Medhat's nickname?")
            alias_answer = RAG.generate_chat_response("Who is CABST?")

        self.assertEqual(
            answer,
            "MAAT was built by Ahmed Medhat, also known as CABST, a 14-year-old Egyptian programmer.",
        )
        self.assertEqual(
            arabic_answer,
            "تم تطوير MAAT بواسطة أحمد مدحت (CABST)، وهو مبرمج مصري يبلغ من العمر 14 عامًا.",
        )
        self.assertEqual(nickname_answer, answer)
        self.assertEqual(alias_answer, answer)

    def test_small_talk_uses_chat_model_without_retrieval(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi!"}]
        with (
            patch("RAG.retrieve", side_effect=AssertionError("small talk should not retrieve")),
            patch.object(RAG.client.chat.completions, "create", return_value=completion("I'm doing well!")) as create,
        ):
            answer = RAG.generate_chat_response("How are you?", history=history)

        self.assertEqual(answer, "I'm doing well!")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[-3:-1], history)
        self.assertIn("same language", messages[0]["content"])
        self.assertIsNone(RAG.embed_model)
        self.assertIsNone(RAG.collection)

    def test_legal_response_keeps_retrieved_sources_and_history(self):
        history = [{"role": "user", "content": "How does a contract form?"}]
        sources = [{"law_name": "Civil Code"}]
        with (
            patch("RAG.retrieve", return_value=(["A contract requires agreement."], sources)),
            patch.object(RAG.client.chat.completions, "create", return_value=completion("Answer")) as create,
        ):
            answer, returned_sources = RAG.generate_answer(
                "What about its requirements?",
                history=history,
                document_context="File: contract.txt\nPayment is due monthly.",
            )

        self.assertEqual(answer, "Answer")
        self.assertEqual(returned_sources, sources)
        messages = create.call_args.kwargs["messages"]
        self.assertIn("only authority", messages[0]["content"])
        self.assertIn("Arabic even when the current question is in another language", messages[0]["content"])
        self.assertIn("for an English question, write the answer in English", messages[0]["content"])
        self.assertEqual(messages[-2], history[0])
        self.assertIn("A contract requires agreement.", messages[-1]["content"])
        self.assertIn("Payment is due monthly.", messages[-1]["content"])

    def test_english_query_is_embedded_without_translation_before_retrieval(self):
        query = "What are the requirements for a valid contract?"
        encoder = Mock()
        encoded_query = Mock()
        encoded_query.tolist.return_value = [[0.1, 0.2]]
        encoder.encode.return_value = encoded_query
        collection = Mock()
        collection.query.return_value = {"documents": [["Arabic source"]], "metadatas": [[{"law_name": "Civil Code"}] ]}

        with (
            patch("RAG._get_embed_model", return_value=encoder) as get_encoder,
            patch("RAG._get_collection", return_value=collection) as get_collection,
            patch("RAG._get_bm25_index", return_value=(None, [], [])),
        ):
            documents, sources = RAG.retrieve(query, k=1)

        self.assertEqual(documents, ["Arabic source"])
        self.assertEqual(sources, [{"law_name": "Civil Code"}])
        get_encoder.return_value.encode.assert_called_once_with([query], normalize_embeddings=True)
        get_collection.return_value.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=15,
        )


if __name__ == "__main__":
    unittest.main()
