import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from services import auth_service, user_service


class SessionPersistenceTests(unittest.TestCase):
    def test_session_token_restores_user_until_revoked(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions_file = Path(directory) / "sessions.json"
            user = {"id": "user-1", "username": "ahmed"}
            with (
                patch.object(auth_service, "SESSIONS_FILE", sessions_file),
                patch.object(auth_service, "get_user_by_id", return_value=user),
            ):
                token = auth_service.create_session("user-1")
                self.assertEqual(auth_service.get_session_user(token), user)
                self.assertTrue(auth_service.revoke_session(token))
                self.assertIsNone(auth_service.get_session_user(token))

    def test_most_recent_non_empty_conversation_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            conversations_file = Path(directory) / "conversations.json"
            with patch.object(user_service, "CONVERSATIONS_FILE", conversations_file):
                user_service.save_conversation("user-1", "older", [{"role": "user", "content": "Older"}])
                user_service.save_conversation("user-1", "newer", [{"role": "user", "content": "Newer"}])
                latest = user_service.get_most_recent_conversation("user-1")

        self.assertEqual(latest["id"], "newer")
        self.assertEqual(latest["messages"], [{"role": "user", "content": "Newer"}])

    def test_text_upload_is_available_as_private_chat_context(self):
        with tempfile.TemporaryDirectory() as directory:
            uploads_dir = Path(directory) / "uploads"
            uploads_file = Path(directory) / "uploads.json"
            upload = SimpleNamespace(name="agreement.txt", type="text/plain", getvalue=lambda: b"The tenant must pay rent on the first day of each month.")
            with (
                patch.object(user_service, "UPLOADS_DIR", uploads_dir),
                patch.object(user_service, "UPLOADS_FILE", uploads_file),
            ):
                record = user_service.save_upload("user-1", upload)
                text = user_service.upload_text(record)

        self.assertIn("tenant must pay rent", text)
        self.assertEqual(record["rag_status"], "Ready for use as private chat context.")
