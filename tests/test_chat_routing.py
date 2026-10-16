import unittest

from utils.chat_routing import is_about_creator, is_small_talk


class SmallTalkRoutingTests(unittest.TestCase):
    def test_english_greetings_and_polite_chat_use_chat_mode(self):
        for message in (
            "hi",
            "Hello there!",
            "hey, how are you?",
            "how u doing",
            "yoooooooooo",
            "hiowdy",
            "thank you",
            "tell me a joke",
            "What's up?",
            "Do you like coffee?",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_small_talk(message))

    def test_arabic_greetings_and_polite_chat_use_chat_mode(self):
        for message in ("مرحباً", "السلام عليكم", "كيف حالك؟", "إزيك؟", "شكراً لك"):
            with self.subTest(message=message):
                self.assertTrue(is_small_talk(message))

    def test_legal_questions_stay_on_retrieval_path(self):
        for message in (
            "What are the requirements for a valid contract?",
            "ما شروط صحة العقد في القانون المصري؟",
            "Can you explain article 3?",
        ):
            with self.subTest(message=message):
                self.assertFalse(is_small_talk(message))

    def test_creator_questions_use_chat_mode(self):
        for message in (
            "Who built MAAT?",
            "Who is the developer of the MAAT system?",
            "Who is the MAAT creator?",
            "Who is Ahmed Medhat?",
            "What is Ahmed Medhat's nickname?",
            "Who is CABST?",
            "مين طور ماعت؟",
            "مين اللي عمل MAAT؟",
            "من هو أحمد مدحت؟",
            "ما لقب أحمد مدحت؟",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_about_creator(message))
                self.assertTrue(is_small_talk(message))


if __name__ == "__main__":
    unittest.main()
