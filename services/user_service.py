from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from utils.json_storage import DATA_DIR, read_json, update_json
from services.document_service import compact_document_text, extract_document_text


CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
PREFERENCES_FILE = DATA_DIR / "preferences.json"
UPLOADS_FILE = DATA_DIR / "uploads.json"
UPLOADS_DIR = DATA_DIR / "uploads"

CONVERSATIONS_DEFAULT = {"conversations": []}
PREFERENCES_DEFAULT = {"preferences": {}}
UPLOADS_DEFAULT = {"uploads": []}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_UPLOAD_TYPES = {"txt", "md", "pdf", "docx"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def title_from_messages(messages):
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            content = " ".join(message["content"].split())
            return content[:72] + ("..." if len(content) > 72 else "")
    return "Untitled conversation"


def list_conversations(user_id):
    data = read_json(CONVERSATIONS_FILE, CONVERSATIONS_DEFAULT)
    items = [item for item in data.get("conversations", []) if item.get("user_id") == user_id]
    return sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_conversation(user_id, conversation_id):
    for conversation in list_conversations(user_id):
        if conversation.get("id") == conversation_id:
            return conversation
    return None


def get_most_recent_conversation(user_id):
    return next(
        (
            conversation
            for conversation in list_conversations(user_id)
            if isinstance(conversation.get("messages"), list) and conversation["messages"]
        ),
        None,
    )


def save_conversation(user_id, conversation_id, messages):
    now = utc_now()

    def mutator(data):
        conversations = data.setdefault("conversations", [])
        for item in conversations:
            if item.get("id") == conversation_id and item.get("user_id") == user_id:
                item["title"] = title_from_messages(messages)
                item["messages"] = messages
                item["updated_at"] = now
                return item
        created = {
            "id": conversation_id or str(uuid4()),
            "user_id": user_id,
            "title": title_from_messages(messages),
            "messages": messages,
            "created_at": now,
            "updated_at": now,
        }
        conversations.append(created)
        return created

    return update_json(CONVERSATIONS_FILE, CONVERSATIONS_DEFAULT, mutator)


def delete_conversation(user_id, conversation_id):
    def mutator(data):
        before = len(data.get("conversations", []))
        data["conversations"] = [
            item for item in data.get("conversations", [])
            if not (item.get("id") == conversation_id and item.get("user_id") == user_id)
        ]
        return len(data["conversations"]) != before

    return update_json(CONVERSATIONS_FILE, CONVERSATIONS_DEFAULT, mutator)


def load_preferences(user_id):
    data = read_json(PREFERENCES_FILE, PREFERENCES_DEFAULT)
    return data.get("preferences", {}).get(user_id, {"language": "English", "show_sources": True, "retrieval_depth": 4})


def save_preferences(user_id, preferences):
    def mutator(data):
        data.setdefault("preferences", {})[user_id] = preferences
        return preferences

    return update_json(PREFERENCES_FILE, PREFERENCES_DEFAULT, mutator)


def list_uploads(user_id):
    data = read_json(UPLOADS_FILE, UPLOADS_DEFAULT)
    uploads = [item for item in data.get("uploads", []) if item.get("user_id") == user_id]
    return sorted(uploads, key=lambda item: item.get("uploaded_at", ""), reverse=True)


def get_upload(user_id, upload_id):
    return next((item for item in list_uploads(user_id) if item.get("id") == upload_id), None)


def upload_text(record):
    if not record:
        return ""
    path = DATA_DIR / record["stored_path"]
    if not path.is_file():
        raise FileNotFoundError(f"Stored document not found: {record['filename']}")
    return compact_document_text(extract_document_text(path, record["filename"]))


def save_upload(user_id, uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_UPLOAD_TYPES:
        raise ValueError("Supported files: PDF, DOCX, TXT, and MD.")
    payload = uploaded_file.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than the 10 MB prototype limit.")

    user_dir = UPLOADS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    upload_id = str(uuid4())
    safe_name = Path(uploaded_file.name).name
    stored_name = f"{upload_id}-{safe_name}"
    stored_path = user_dir / stored_name
    stored_path.write_bytes(payload)
    try:
        stored_reference = str(stored_path.relative_to(DATA_DIR))
    except ValueError:
        stored_reference = str(stored_path)
    record = {
        "id": upload_id,
        "user_id": user_id,
        "filename": safe_name,
        "content_type": uploaded_file.type,
        "size": len(payload),
        "stored_path": stored_reference,
        "uploaded_at": utc_now(),
        "rag_status": "Ready for use as private chat context.",
    }

    def mutator(data):
        data.setdefault("uploads", []).append(record)
        return record

    return update_json(UPLOADS_FILE, UPLOADS_DEFAULT, mutator)
