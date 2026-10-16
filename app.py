import base64
import importlib
import json
import time
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st

from services.auth_service import (
    authenticate,
    create_session,
    get_session_user,
    get_user_by_id,
    register_user,
    revoke_session,
)
from services.user_service import (
    delete_conversation,
    get_most_recent_conversation,
    get_upload,
    list_conversations,
    list_uploads,
    load_preferences,
    save_conversation,
    save_preferences,
    save_upload,
    upload_text,
)
from utils.chat_routing import is_small_talk


ROOT = Path(__file__).resolve().parent
HERO_IMAGE = ROOT / "assets" / "maat_intro_lockup_transparent.png"
BRAND_ICON_IMAGE = ROOT / "assets" / "4.jpeg"
THEME_FILE = ROOT / "styles" / "theme.css"
ENGLISH_TOPICS_FILE = ROOT / "data" / "legal_topics_english.json"
DEFAULT_CHAT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
LANGUAGES = ["English", "Arabic"]
DEPTH_OPTIONS = [3, 4, 6]
SESSION_QUERY_KEY = "maat_session"

COPY = {
    "English": {
        "dir": "ltr",
        "tagline": "Egyptian legal intelligence grounded in retrieved sources",
        "enter": "ENTER MAAT",
        "explore": "EXPLORE THE EXPERIENCE",
        "headline": "WHERE ANCIENT WISDOM MEETS MODERN JUSTICE.",
        "intro": "MAAT is an AI-powered assistant for exploring Egyptian law with source-aware answers and careful legal framing.",
        "login": "Login",
        "register": "Register",
        "logout": "Logout",
        "new": "New conversation",
        "dashboard": "Dashboard",
        "chat": "Legal AI Chat",
        "documents": "Documents",
        "topics": "Legal Topics",
        "profile": "Profile",
        "settings": "Settings",
        "about": "About MAAT",
        "contact": "Contact Us",
        "question_placeholder": "Ask an Egyptian legal question...",
        "not_advice": "Informational legal assistance only. Verify sources and consult a qualified lawyer before relying on an answer.",
        "sources": "Retrieved sources",
        "no_sources": "No sources were returned for this answer.",
        "thinking": "Searching legal texts and drafting the answer...",
        "chat_thinking": "Thinking of a reply...",
        "done": "Research completed in {seconds:.1f}s",
        "chat_done": "Reply ready",
        "copy": "Copy answer",
        "copied": "Copied!",
        "copy_failed": "Could not copy. Check browser clipboard permissions.",
        "suggestions": [
            "Who may initiate a criminal case?",
            "How does the Constitution describe Egypt's system?",
            "Which law replaced the former Civil Code?",
            "When is spousal maintenance due?",
        ],
    },
    "Arabic": {
        "dir": "rtl",
        "tagline": "ذكاء قانوني مصري قائم على المصادر المسترجعة",
        "enter": "دخول MAAT",
        "explore": "استكشاف التجربة",
        "headline": "حيث تلتقي الحكمة القديمة بالعدالة الحديثة.",
        "intro": "ماعت مساعد ذكي لاستكشاف القانون المصري بإجابات مرتبطة بالمصادر وبصياغة قانونية حذرة.",
        "login": "تسجيل الدخول",
        "register": "إنشاء حساب",
        "logout": "تسجيل الخروج",
        "new": "محادثة جديدة",
        "dashboard": "لوحة التحكم",
        "chat": "المحادثة القانونية",
        "documents": "المستندات",
        "topics": "الموضوعات القانونية",
        "profile": "الملف الشخصي",
        "settings": "الإعدادات",
        "about": "عن MAAT",
        "question_placeholder": "اكتب سؤالك القانوني المصري...",
        "not_advice": "هذه مساعدة معلوماتية فقط. راجع المصادر واستشر محاميا مختصا قبل الاعتماد المهني.",
        "sources": "المصادر المسترجعة",
        "no_sources": "لا توجد مصادر معروضة لهذه الإجابة.",
        "thinking": "يتم البحث في النصوص القانونية وصياغة الإجابة...",
        "chat_thinking": "يتم إعداد الرد...",
        "done": "اكتمل البحث خلال {seconds:.1f} ثانية",
        "chat_done": "تم إعداد الرد",
        "copy": "نسخ الإجابة",
        "copied": "تم النسخ!",
        "copy_failed": "تعذر النسخ. تحقق من أذونات الحافظة في المتصفح.",
        "suggestions": [
            "من يختص برفع الدعوى الجنائية؟",
            "كيف يصف الدستور نظام جمهورية مصر العربية؟",
            "ما القانون الذي ألغى القانون المدني القديم؟",
            "متى تستحق الزوجة النفقة؟",
        ],
    },
}

COPY["Arabic"]["contact"] = "Contact Us"

EXPERIENCE_COPY = {
    "English": {
        "title": "Explore MAAT",
        "intro": "A source-aware workspace for navigating Egyptian legal information with greater clarity.",
        "heading": "What MAAT offers",
        "back": "Back",
        "enter": "Enter MAAT",
        "features": [
            ("Source-grounded answers", "Research Egyptian legal questions with answers tied to retrieved legal sources.", "menu_book"),
            ("Arabic and English", "Ask and read in Arabic or English, while working from the Arabic legal corpus.", "translate"),
            ("Document review", "Upload a document and use MAAT to identify key obligations, deadlines, and risks.", "upload_file"),
            ("Legal topic browsing", "Browse representative passages from the legal collection before you ask a question.", "account_balance"),
            ("Research-ready citations", "See the legal sources behind research answers and distinguish evidence from uncertainty.", "format_quote"),
        ],
    },
    "Arabic": {
        "title": "استكشف MAAT",
        "intro": "مساحة عمل واعية بالمصادر لاستكشاف المعلومات القانونية المصرية بوضوح أكبر.",
        "heading": "ما الذي يقدمه MAAT؟",
        "back": "رجوع",
        "enter": "دخول MAAT",
        "features": [
            ("إجابات قائمة على المصادر", "ابحث في المسائل القانونية المصرية بإجابات مرتبطة بالمصادر القانونية المسترجعة.", "menu_book"),
            ("العربية والإنجليزية", "اسأل واقرأ بالعربية أو الإنجليزية مع الاستناد إلى المدونة القانونية العربية.", "translate"),
            ("مراجعة المستندات", "ارفع مستندًا ودع MAAT يساعدك في تحديد الالتزامات والمواعيد والمخاطر الرئيسية.", "upload_file"),
            ("تصفح الموضوعات القانونية", "اطّلع على مقاطع ممثلة من المجموعة القانونية قبل طرح سؤالك.", "account_balance"),
            ("إجابات جاهزة للبحث", "اطّلع على المصادر القانونية وراء الإجابات وفرّق بين الأدلة ونقاط عدم اليقين.", "format_quote"),
        ],
    },
}


def image_data_uri(path):
    if not path.exists():
        return ""
    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def inject_styles():
    css = THEME_FILE.read_text(encoding="utf-8") if THEME_FILE.exists() else ""
    hero_uri = image_data_uri(HERO_IMAGE)
    icon_uri = image_data_uri(BRAND_ICON_IMAGE)
    theme_marker = "maat-theme-light" if st.session_state.get("theme") == "light" else "maat-theme-dark"
    st.html(f"<style>{css}</style><style>:root {{--maat-hero-image: url('{hero_uri}'); --maat-brand-icon: url('{icon_uri}');}}</style><span class='{theme_marker}' aria-hidden='true'></span>")


def t(key):
    # Safe lookup: a missing/invalid language falls back to English instead of crashing.
    language = st.session_state.get("language")
    return COPY.get(language, COPY["English"])[key]


LANGUAGE_LABELS = {"English": "English", "Arabic": "العربية"}


def on_language_change(widget_key, user_id=None):
    value = st.session_state.get(widget_key)
    if value in COPY:
        st.session_state.language = value
        if user_id:
            persist_preferences(user_id)


def render_language_selector(widget_key, user_id=None):
    # Sync the widget with the saved language, then let the callback update it on click.
    st.session_state[widget_key] = st.session_state.language
    st.radio(
        "Language / اللغة",
        LANGUAGES,
        horizontal=True,
        key=widget_key,
        format_func=lambda value: LANGUAGE_LABELS[value],
        on_change=on_language_change,
        args=(widget_key, user_id),
    )


def on_depth_change():
    value = st.session_state.get("depth_widget")
    if value in DEPTH_OPTIONS:
        st.session_state.retrieval_depth = value


def init_state():
    st.session_state.setdefault("auth_view", "intro")
    st.session_state.setdefault("language", "English")
    st.session_state.setdefault("page", "Dashboard")
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("retrieval_depth", 4)
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("theme", "dark")

    # Self-heal bad values (e.g. None from a deselected segmented control).
    if st.session_state.language not in COPY:
        st.session_state.language = "English"
    if st.session_state.retrieval_depth not in DEPTH_OPTIONS:
        st.session_state.retrieval_depth = 4
    # MAAT now ships with one visual theme.
    st.session_state.theme = "dark"


def current_user():
    user = st.session_state.get("user")
    if not user:
        return None
    fresh = get_user_by_id(user["id"])
    if fresh:
        st.session_state.user = fresh
    return fresh


def restore_latest_conversation(user_id):
    latest = get_most_recent_conversation(user_id)
    if latest:
        st.session_state.conversation_id = latest["id"]
        st.session_state.messages = latest["messages"]
        st.session_state.page = "Legal AI Chat"


def initialize_user_session(user, resume_conversation=False):
    st.session_state.user = user
    prefs = load_preferences(user["id"])
    language = prefs.get("language", "English")
    st.session_state.language = language if language in COPY else "English"
    st.session_state.show_sources = prefs.get("show_sources", True)
    st.session_state.theme = "dark"
    depth = prefs.get("retrieval_depth", 4)
    st.session_state.retrieval_depth = depth if depth in DEPTH_OPTIONS else 4
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.page = "Dashboard"
    if resume_conversation:
        restore_latest_conversation(user["id"])


def login_user(user):
    initialize_user_session(user)
    token = create_session(user["id"])
    st.session_state.auth_token = token
    st.query_params[SESSION_QUERY_KEY] = token


def restore_persistent_login():
    if st.session_state.get("user"):
        return
    token = st.query_params.get(SESSION_QUERY_KEY)
    if isinstance(token, list):
        token = token[-1] if token else None
    user = get_session_user(token)
    if user:
        st.session_state.auth_token = token
        initialize_user_session(user, resume_conversation=True)
        return
    if token:
        del st.query_params[SESSION_QUERY_KEY]


def logout():
    token = st.session_state.get("auth_token") or st.query_params.get(SESSION_QUERY_KEY)
    if isinstance(token, list):
        token = token[-1] if token else None
    revoke_session(token)
    if SESSION_QUERY_KEY in st.query_params:
        del st.query_params[SESSION_QUERY_KEY]
    for key in ["user", "messages", "conversation_id", "pending_question", "auth_token"]:
        st.session_state.pop(key, None)
    init_state()
    st.session_state.auth_view = "login"
    st.rerun()


def render_intro():
    st.html(
        f"""
        <main class="intro-shell" dir="{t('dir')}">
            <section class="intro-hero">
                <img class="intro-logo" src="{image_data_uri(HERO_IMAGE)}" alt="Ma'at">
                <p class="kicker">MAAT | Egyptian Legal AI</p>
                <h1>{escape(t('headline'))}</h1>
                <p>{escape(t('intro'))}</p>
                <div class="intro-proof">
                    <span>Source-grounded RAG</span>
                    <span>Private JSON prototype</span>
                    <span>Arabic / English</span>
                </div>
            </section>
        </main>
        """
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button(t("enter"), type="primary", width="stretch"):
            st.session_state.auth_view = "login"
            st.rerun()
        if st.button(t("explore"), width="stretch"):
            st.session_state.auth_view = "explore"
            st.rerun()


def render_experience_features(copy):
    columns = st.columns(3)
    for index, (title, body, _icon) in enumerate(copy["features"], start=1):
        with columns[(index - 1) % len(columns)]:
            st.html(
                f'''
                <article class="feature-card" dir="{t("dir")}">
                    <span>{index:02d}</span>
                    <strong>{escape(title)}</strong>
                    <p>{escape(body)}</p>
                </article>
                '''
            )


def render_experience():
    copy = EXPERIENCE_COPY[st.session_state.language]
    st.html(
        f'''
        <section class="page-hero" dir="{t("dir")}">
            <p class="kicker">MAAT | Egyptian Legal AI</p>
            <h1>{escape(copy["title"])}</h1>
            <p>{escape(copy["intro"])}</p>
        </section>
        '''
    )
    st.html(f'<h3 class="section-title">{escape(copy["heading"])}</h3>')
    render_experience_features(copy)

    left, middle, right = st.columns([1, 1, 1])
    with left:
        if st.button(copy["back"], width="stretch"):
            st.session_state.auth_view = "intro"
            st.rerun()
    with middle:
        if st.button(copy["enter"], type="primary", width="stretch"):
            st.session_state.auth_view = "login"
            st.rerun()


def render_auth():
    render_language_selector("auth_language_widget")

    if st.session_state.auth_view == "intro":
        render_intro()
        return
    if st.session_state.auth_view == "explore":
        render_experience()
        return

    left, right = st.columns([1.1, 0.9], gap="large", vertical_alignment="center")
    with left:
        st.html(
            f"""
            <section class="auth-art" dir="{t('dir')}">
                <div class="auth-wordmark">MAAT</div>
                <p class="kicker">MAAT | Egyptian Legal AI</p>
                <h1>{escape(t('headline'))}</h1>
                <p>{escape(t('intro'))}</p>
            </section>
            """
        )
    with right:
        tabs = st.tabs([t("login"), t("register")])
        with tabs[0]:
            with st.form("login_form", border=False):
                identifier = st.text_input("Email or username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(t("login"), type="primary", width="stretch")
                if submitted:
                    ok, message, user = authenticate(identifier, password)
                    if ok:
                        login_user(user)
                        st.rerun()
                    st.error(message)
        with tabs[1]:
            with st.form("register_form", border=False):
                full_name = st.text_input("Full name")
                email = st.text_input("Email address")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button(t("register"), type="primary", width="stretch")
                if submitted:
                    ok, errors, user = register_user(full_name, email, username, password, confirm)
                    if ok:
                        st.success("Account created. You are now signed in.")
                        login_user(user)
                        st.rerun()
                    for error in errors:
                        st.error(error)


def persist_preferences(user_id):
    save_preferences(
        user_id,
        {
            "language": st.session_state.language,
            "show_sources": st.session_state.show_sources,
            "retrieval_depth": st.session_state.retrieval_depth,
            "theme": st.session_state.theme,
        },
    )


def render_sidebar(user):
    with st.sidebar:
        st.html(f'<div class="brand-lockup"><div class="brand-symbol" role="img" aria-label="Ma’at"></div><div><b>MAAT</b><span>{escape(t("tagline"))}</span></div></div>')
        if st.button(t("new"), icon=":material/add:", width="stretch"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid4())
            st.session_state.page = "Legal AI Chat"
            st.rerun()

        page_labels = [t("dashboard"), t("chat"), t("topics"), t("profile"), t("settings"), t("about"), t("contact")]
        page_values = ["Dashboard", "Legal AI Chat", "Legal Topics", "Profile", "Settings", "About MAAT", "Contact Us"]
        current_index = page_values.index(st.session_state.page) if st.session_state.page in page_values else 0
        selected_label = st.radio("Navigation", page_labels, index=current_index, label_visibility="collapsed")
        st.session_state.page = dict(zip(page_labels, page_values))[selected_label]

        st.divider()
        st.caption("Chat history")
        for item in list_conversations(user["id"])[:8]:
            chat_column, delete_column = st.columns([3, 1], gap="small")
            with chat_column:
                if st.button(item["title"], key=f"conv-{item['id']}", width="stretch"):
                    st.session_state.conversation_id = item["id"]
                    st.session_state.messages = item.get("messages", [])
                    st.session_state.page = "Legal AI Chat"
                    st.rerun()
            with delete_column:
                if st.button(
                    "Delete",
                    icon=":material/delete:",
                    key=f"delete-conv-{item['id']}",
                    help="Delete this conversation",
                    width="stretch",
                ):
                    delete_conversation(user["id"], item["id"])
                    if st.session_state.conversation_id == item["id"]:
                        st.session_state.messages = []
                        st.session_state.conversation_id = None
                    st.rerun()

        st.divider()
        render_language_selector("sidebar_language_widget", user["id"])
        if st.button(t("logout"), icon=":material/logout:", width="stretch"):
            logout()


def source_label(source):
    law_name = source.get("law_name", "Legal source")
    categories = source.get("categories")
    if isinstance(categories, list):
        categories = ", ".join(str(category) for category in categories)
    return law_name, categories or "Uncategorized"


def render_sources(sources):
    if not sources:
        st.caption(t("no_sources"))
        return
    st.html(f'<h3 class="section-title">{escape(t("sources"))}</h3>')
    columns = st.columns(min(3, len(sources)), gap="small")
    for index, source in enumerate(sources, start=1):
        law_name, categories = source_label(source)
        with columns[(index - 1) % len(columns)]:
            st.html(
                f"""
                <article class="source-card">
                    <span>Source {index:02d}</span>
                    <strong>{escape(str(law_name))}</strong>
                    <em>{escape(str(categories))}</em>
                </article>
                """
            )


@st.cache_data(ttl=300, show_spinner=False)
def load_topic_law_excerpts():
    rag = importlib.import_module("RAG")
    return rag.get_law_excerpts(max_laws=7, chunks_per_law=2)


@st.cache_data(show_spinner=False)
def load_english_topic_law_excerpts():
    if not ENGLISH_TOPICS_FILE.exists():
        return []
    translations = json.loads(ENGLISH_TOPICS_FILE.read_text(encoding="utf-8"))
    return list(translations.values())


def render_message(message, index):
    if message["role"] == "user":
        st.html(f'<div class="message user-message">{escape(message["content"])}</div>')
        for attachment in message.get("attachments", []):
            st.caption(f":material/attach_file: {attachment.get('filename', 'Attached document')}")
        return
    with st.container(key=f"assistant-message-{index}"):
        st.markdown(message["content"])
        # Casual replies deliberately carry no citations. Empty source lists are
        # also not rendered as a misleading "no sources" legal-answer block.
        if message.get("sources"):
            st.caption(t("not_advice"))
            if st.session_state.show_sources:
                render_sources(message["sources"])
        answer = escape(message["content"], quote=True)
        copy_label = escape(t("copy"), quote=True)
        copied_label = escape(t("copied"), quote=True)
        copy_failed_label = escape(t("copy_failed"), quote=True)
        st.html(
            f"""
            <div class="maat-copy-control">
                <button type="button" class="maat-copy-button"
                    data-answer="{answer}"
                    data-copied="{copied_label}"
                    data-copy-failed="{copy_failed_label}">{copy_label}</button>
                <span class="maat-copy-feedback" aria-live="polite"></span>
            </div>
            <script>
                (() => {{
                    const control = document.currentScript.previousElementSibling;
                    const button = control.querySelector(".maat-copy-button");
                    const feedback = control.querySelector(".maat-copy-feedback");
                    button.addEventListener("click", async () => {{
                        try {{
                            await navigator.clipboard.writeText(button.dataset.answer);
                            feedback.textContent = button.dataset.copied;
                        }} catch (error) {{
                            feedback.textContent = button.dataset.copyFailed;
                        }}
                    }});
                }})();
            </script>
            """,
            unsafe_allow_javascript=True,
        )


def document_context_for_messages(user_id, messages):
    sections = []
    seen_uploads = set()
    for message in messages:
        for attachment in message.get("attachments", []):
            upload_id = attachment.get("id")
            if upload_id in seen_uploads:
                continue
            seen_uploads.add(upload_id)
            record = get_upload(user_id, upload_id)
            if record:
                sections.append(f"File: {record['filename']}\n{upload_text(record)}")
    return "\n\n---\n\n".join(sections)


def ask_question(user, question, attachments=None):
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = str(uuid4())
    user_message = {"role": "user", "content": question}
    if attachments:
        user_message["attachments"] = attachments
    st.session_state.messages.append(user_message)
    # Persist the user turn before model work begins so a refresh or provider
    # failure cannot orphan the active conversation.
    save_conversation(user["id"], st.session_state.conversation_id, st.session_state.messages)
    st.html(f'<div class="message user-message">{escape(question)}</div>')

    small_talk = is_small_talk(question)
    with st.status(t("chat_thinking") if small_talk else t("thinking"), expanded=False) as status:
        started_at = time.perf_counter()
        rag = importlib.import_module("RAG")
        history = st.session_state.messages[:-1]
        if small_talk:
            answer = rag.generate_chat_response(question, history=history)
            sources = None
        else:
            document_context = document_context_for_messages(user["id"], st.session_state.messages)
            answer, sources = rag.generate_answer(
                question,
                k=st.session_state.retrieval_depth,
                history=history,
                document_context=document_context,
            )
        elapsed = time.perf_counter() - started_at
        status.update(
            label=t("chat_done") if sources is None else t("done").format(seconds=elapsed),
            state="complete",
        )

    assistant_message = {"role": "assistant", "content": answer}
    if sources:
        assistant_message["sources"] = sources
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message, len(st.session_state.messages) - 1)
    save_conversation(user["id"], st.session_state.conversation_id, st.session_state.messages)


def render_dashboard(user):
    conversations = list_conversations(user["id"])
    uploads = list_uploads(user["id"])
    st.html(
        f"""
        <section class="page-hero" dir="{t('dir')}">
            <p class="kicker">MAAT PRIVATE CHAMBER</p>
            <h1>{escape('Welcome, ' + user['full_name'])}</h1>
            <p>{escape(t('tagline'))}</p>
        </section>
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversations", len(conversations), icon=":material/chat:")
    c2.metric("Uploaded documents", len(uploads), icon=":material/upload_file:")
    c3.metric("Retrieval depth", f"{st.session_state.retrieval_depth} sources", icon=":material/database:")

    st.html('<h3 class="section-title">Suggested questions</h3>')
    cols = st.columns(2)
    for i, prompt in enumerate(t("suggestions")):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggestion-{i}", width="stretch"):
                st.session_state.pending_question = prompt
                st.session_state.page = "Legal AI Chat"
                st.rerun()

    st.html('<h3 class="section-title">Recent conversations</h3>')
    if not conversations:
        st.info("No conversations yet. Start with a suggested legal question.")
    for item in conversations[:4]:
        st.html(f'<div class="list-row"><strong>{escape(item["title"])}</strong><span>{escape(item.get("updated_at", ""))}</span></div>')


def render_chat(user):
    st.html(f'<section class="chat-header" dir="{t("dir")}"><h1>{escape(t("chat"))}</h1><p>{escape(t("not_advice"))}</p></section>')
    if not st.session_state.messages:
        st.html('<div class="empty-state">Choose a question or write your own. Answers are grounded in the existing Egyptian legal corpus when retrieval succeeds.</div>')
        cols = st.columns(2)
        for i, prompt in enumerate(t("suggestions")):
            with cols[i % 2]:
                if st.button(prompt, key=f"chat-suggestion-{i}", width="stretch"):
                    st.session_state.pending_question = prompt
                    st.rerun()

    for index, message in enumerate(st.session_state.messages):
        render_message(message, index)

    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
    attachments = []
    submitted = st.chat_input(
        t("question_placeholder"),
        key="legal_chat_input",
        accept_file="multiple",
        file_type=["pdf", "docx", "txt", "md"],
        max_upload_size=10,
    )
    if submitted:
        prompt = submitted.text
        try:
            for uploaded_file in submitted.files:
                record = save_upload(user["id"], uploaded_file)
                attachments.append({"id": record["id"], "filename": record["filename"]})
        except Exception as error:
            st.error(f"Could not attach document: {error}")
            return
        if attachments and not prompt.strip():
            prompt = "Please review the attached document(s)."
    if prompt:
        try:
            ask_question(user, prompt, attachments=attachments)
        except Exception as error:
            st.error("MAAT could not generate an answer. Check GROQ_API_KEY, the local embedding model, and the Chroma collection.", icon=":material/error:")
            with st.expander("Error details"):
                st.code(str(error))


def render_topics():
    st.html(f'<section class="page-hero compact"><h1>{escape(t("topics"))}</h1><p>Read representative passages from the legal sources in the collection.</p></section>')
    st.html('<h3 class="section-title">Browse source passages</h3>')
    st.caption("English excerpts are pre-translated from Arabic source text." if st.session_state.language == "English" else "Representative excerpts are loaded directly from the Egyptian-law Chroma collection.")
    try:
        if st.session_state.language == "English":
            laws = load_english_topic_law_excerpts()
        else:
            with st.spinner("Loading legal source passages..."):
                laws = load_topic_law_excerpts()
    except Exception as error:
        st.error("Legal source passages could not be loaded from Chroma.")
        with st.expander("Error details"):
            st.code(str(error))
        laws = []

    for law in laws:
        display_name = law["law_name"]
        source_detail = "Pre-translated from Arabic source text" if st.session_state.language == "English" else str(law["categories"])
        st.html(
            f'<article class="source-card"><span>Egyptian legal source</span><strong>{escape(display_name)}</strong><em>{escape(source_detail)}</em></article>'
        )
        for index, excerpt in enumerate(law["excerpts"], start=1):
            st.markdown(f"**Excerpt {index}**")
            st.write(excerpt)

    if not laws:
        st.info("No legal source passages are available in the collection yet.")


def render_profile(user):
    st.html(f'<section class="page-hero compact"><h1>{escape(t("profile"))}</h1><p>Your account details. Sensitive fields such as password hashes are never shown.</p></section>')
    st.text_input("Full name", user["full_name"], disabled=True)
    st.text_input("Email", user["email"], disabled=True)
    st.text_input("Username", user["username"], disabled=True)
    st.text_input("User ID", user["id"], disabled=True)


def render_settings(user):
    st.html(f'<section class="page-hero compact"><h1>{escape(t("settings"))}</h1><p>Adjust retrieval and display preferences.</p></section>')
    st.session_state["depth_widget"] = st.session_state.retrieval_depth
    st.radio(
        "Number of retrieved sources",
        DEPTH_OPTIONS,
        horizontal=True,
        key="depth_widget",
        on_change=on_depth_change,
    )
    st.session_state.show_sources = st.toggle("Show retrieved sources in answers", value=st.session_state.show_sources)
    if st.button("Save settings", type="primary"):
        persist_preferences(user["id"])
        st.success("Settings saved.")
    st.divider()
    if st.session_state.conversation_id and st.button("Delete current conversation"):
        delete_conversation(user["id"], st.session_state.conversation_id)
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.success("Conversation deleted.")


def render_about():
    copy = EXPERIENCE_COPY[st.session_state.language]
    st.html(
        f"""
        <section class="page-hero" dir="{t('dir')}">
            <p class="kicker">MAAT | Egyptian Legal AI</p>
            <h1>About MAAT</h1>
            <p>{escape(copy['intro'])} MAAT uses hybrid retrieval to ground each answer in the legal collection.</p>
        </section>
        """
    )
    st.html(f'<h3 class="section-title">{escape(copy["heading"])}</h3>')
    render_experience_features(copy)
    st.html('<h3 class="section-title">How MAAT works</h3>')
    st.markdown("MAAT uses hybrid retrieval-augmented generation (RAG) to connect legal questions with relevant Egyptian legal material before drafting a response.")
    st.html(
        """
        <section class="rag-flow">
            <article class="rag-step"><strong>1. Understand the question</strong><p>MAAT converts the question into a semantic embedding so it can look for passages with similar legal meaning, even when the wording differs.</p></article>
            <article class="rag-step"><strong>2. Run two retrieval methods</strong><p>Dense search queries the Chroma vector store for meaning; BM25 keyword search finds passages that share important legal terms with the question.</p></article>
            <article class="rag-step"><strong>3. Fuse the rankings</strong><p>Reciprocal-rank fusion combines both result lists, rewarding passages that rank strongly in either retrieval method.</p></article>
            <article class="rag-step"><strong>4. Generate with evidence</strong><p>The highest-ranked passages are given to the language model as context, helping it produce a source-aware answer and show the retrieved sources.</p></article>
        </section>
        """
    )
    st.info("MAAT supports legal research and learning. It does not provide legal advice, representation, or a guarantee that an answer is complete or current. Verify sources and consult a qualified lawyer for decisions with legal consequences.")
    model_labels = "".join(f'<span class="maat-model">{escape(model)}</span>' for model in DEFAULT_CHAT_MODELS)
    st.html(f'<div class="model-list">{model_labels}</div>')


def render_contact():
    st.html(
        """
        <section class="page-hero compact">
            <p class="kicker">MAAT | Egyptian Legal AI</p>
            <h1>Contact Us</h1>
            <p>Meet the team behind MAAT in Cairo, Egypt.</p>
        </section>
        <p class="contact-location">Cairo, Egypt</p>
        <section class="contact-team">
            <article class="feature-card contact-card">
                <span>Programmer</span>
                <strong>Ahmed Medhat ElShamy</strong>
                <p>Ahmed is a 14-year-old programmer with a passion for coding, artificial intelligence, and building useful technology.</p>
                <p><a href="mailto:Ahmeddelshamii@gmail.com">Ahmeddelshamii@gmail.com</a></p>
            </article>
            <article class="feature-card contact-card">
                <span>Instructor/Sensei</span>
                <strong>Abdalla Sherif Omar</strong>
                <p>Abdalla is an AI Engineer and he is the Sensei of Ahmed at MAAT Project.</p>
                <p><a href="mailto:Abdallaasheriff@gmail.com">Abdallaasheriff@gmail.com</a></p>
            </article>
        </section>
        """
    )


def render_app():
    restore_persistent_login()
    user = current_user()
    if not user:
        render_auth()
        return
    render_sidebar(user)
    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard(user)
    elif page == "Legal AI Chat":
        render_chat(user)
    elif page == "Legal Topics":
        render_topics()
    elif page == "Profile":
        render_profile(user)
    elif page == "Settings":
        render_settings(user)
    elif page == "Contact Us":
        render_contact()
    else:
        render_about()


st.set_page_config(page_title="MAAT | Egyptian Legal AI", page_icon=str(BRAND_ICON_IMAGE), layout="wide")
init_state()
inject_styles()
render_app()
