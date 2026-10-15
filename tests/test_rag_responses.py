import unittest
from types import SimpleNamespace
from unittest.mock import patch

import RAG


def completion(answer):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )


class RagResponseTests(unittest.TestCase):
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
            )

        self.assertEqual(answer, "Answer")
        self.assertEqual(returned_sources, sources)
        messages = create.call_args.kwargs["messages"]
        self.assertIn("only authority", messages[0]["content"])
        self.assertEqual(messages[-2], history[0])
        self.assertIn("A contract requires agreement.", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
