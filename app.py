import base64
import importlib
import time
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st

from services.auth_service import authenticate, get_user_by_id, register_user
from services.user_service import (
    delete_conversation,
    list_conversations,
    list_uploads,
    load_preferences,
    save_conversation,
    save_preferences,
    save_upload,
)
from utils.chat_routing import is_small_talk


ROOT = Path(__file__).resolve().parent
HERO_IMAGE = ROOT / "assets" / "maat_hero_tutankhamun.jpg"
THEME_FILE = ROOT / "styles" / "theme.css"
DEFAULT_CHAT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
LANGUAGES = ["English", "Arabic"]
DEPTH_OPTIONS = [3, 4, 6]
STUDIO_STYLES = ["Executive brief", "Risk checklist", "Source map"]

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
        "studio": "Research Studio",
        "profile": "Profile",
        "settings": "Settings",
        "about": "About MAAT",
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
            "Explain the Egyptian Civil Code in simple language.",
            "What are the legal requirements for a valid contract?",
            "Explain the difference between civil and criminal liability.",
            "Help me understand this legal document.",
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
        "studio": "استوديو البحث",
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
            "اشرح القانون المدني المصري بلغة بسيطة.",
            "ما الشروط القانونية لصحة العقد؟",
            "اشرح الفرق بين المسؤولية المدنية والجنائية.",
            "ساعدني على فهم هذا المستند القانوني.",
        ],
    },
}

TOPICS = [
    ("Contracts", "Validity, obligations, termination, and remedies.", "balance"),
    ("Civil Liability", "Fault, harm, causation, and compensation.", "gavel"),
    ("Criminal Procedure", "Investigation, trial guarantees, and rights.", "shield"),
    ("Personal Status", "Family, inheritance, marriage, and custody.", "family_restroom"),
    ("Commercial Law", "Companies, trade obligations, and transactions.", "corporate_fare"),
    ("Customs & Public Law", "Administrative, constitutional, and customs questions.", "account_balance"),
]

MAAT_LOGO_SVG = """
<svg width="54" height="54" viewBox="0 0 64 64" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
<circle cx="32" cy="32" r="29" stroke="currentColor" stroke-width="1.2" opacity=".5"/>
<path d="M14 35c7.8-11.2 27.5-11.2 36 0-8.5 11.2-28.2 11.2-36 0Z" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>
<circle cx="32" cy="35" r="6.6" stroke="currentColor" stroke-width="2"/>
<circle cx="32" cy="35" r="2.5" fill="currentColor"/>
<path d="M20 22h24M24 17h16M31.8 8v9M15 45c10 5.8 23.6 5.8 34 0M25 51l-3 5M39 51l3 5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
<path d="M32 13 19 34h26L32 13Z" stroke="currentColor" stroke-width="1" opacity=".34"/>
</svg>
"""

FEATURES = [
    ("Contract Lens", "Turn a contract question into a structured Egyptian-law review.", "Review this contract under Egyptian law. Summarize validity, obligations, risks, and missing facts."),
    ("Case Compass", "Frame civil versus criminal liability and likely legal issues.", "Compare the possible civil and criminal liability in this scenario under Egyptian law."),
    ("Citation Map", "Ask MAAT to separate supported points from uncertain points.", "Answer with a source-grounded structure: supported findings, uncertain points, and questions to verify."),
    ("Document Brief", "Create a short legal brief from uploaded or pasted document text.", "Help me understand this legal document. Identify parties, obligations, deadlines, and risks."),
]


def image_data_uri(path):
    if not path.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def inject_styles():
    css = THEME_FILE.read_text(encoding="utf-8") if THEME_FILE.exists() else ""
    hero_uri = image_data_uri(HERO_IMAGE)
    st.html(f"<style>{css}</style><style>:root {{--maat-hero-image: url('{hero_uri}');}}</style>")


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

    # Self-heal bad values (e.g. None from a deselected segmented control).
    if st.session_state.language not in COPY:
        st.session_state.language = "English"
    if st.session_state.retrieval_depth not in DEPTH_OPTIONS:
        st.session_state.retrieval_depth = 4


def current_user():
    user = st.session_state.get("user")
    if not user:
        return None
    fresh = get_user_by_id(user["id"])
    if fresh:
        st.session_state.user = fresh
    return fresh


def login_user(user):
    st.session_state.user = user
    prefs = load_preferences(user["id"])
    language = prefs.get("language", "English")
    st.session_state.language = language if language in COPY else "English"
    st.session_state.show_sources = prefs.get("show_sources", True)
    depth = prefs.get("retrieval_depth", 4)
    st.session_state.retrieval_depth = depth if depth in DEPTH_OPTIONS else 4
    st.session_state.page = "Dashboard"
    st.session_state.messages = []
    st.session_state.conversation_id = None


def logout():
    for key in ["user", "messages", "conversation_id", "pending_question"]:
        st.session_state.pop(key, None)
    init_state()
    st.session_state.auth_view = "login"
    st.rerun()


def render_intro():
    st.html(
        f"""
        <main class="intro-shell" dir="{t('dir')}">
            <section class="intro-hero">
                <div class="eye-mark"><span>{MAAT_LOGO_SVG}</span></div>
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
            st.session_state.auth_view = "register"
            st.rerun()


def render_auth():
    render_language_selector("auth_language_widget")

    if st.session_state.auth_view == "intro":
        render_intro()
        return

    left, right = st.columns([1.1, 0.9], gap="large", vertical_alignment="center")
    with left:
        st.html(
            f"""
            <section class="auth-art" dir="{t('dir')}">
                <div class="eye-mark">{MAAT_LOGO_SVG}</div>
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
        },
    )


def render_sidebar(user):
    with st.sidebar:
        st.html(f'<div class="brand-lockup"><div class="brand-symbol">{MAAT_LOGO_SVG}</div><div><b>MAAT</b><span>{escape(t("tagline"))}</span></div></div>')
        if st.button(t("new"), icon=":material/add:", width="stretch"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid4())
            st.session_state.page = "Legal AI Chat"
            st.rerun()

        page_labels = [t("dashboard"), t("chat"), t("documents"), t("topics"), t("studio"), t("profile"), t("settings"), t("about")]
        page_values = ["Dashboard", "Legal AI Chat", "Documents", "Legal Topics", "Research Studio", "Profile", "Settings", "About MAAT"]
        current_index = page_values.index(st.session_state.page) if st.session_state.page in page_values else 0
        selected_label = st.radio("Navigation", page_labels, index=current_index, label_visibility="collapsed")
        st.session_state.page = dict(zip(page_labels, page_values))[selected_label]

        st.divider()
        st.caption("Chat history")
        for item in list_conversations(user["id"])[:8]:
            if st.button(item["title"], key=f"conv-{item['id']}", width="stretch"):
                st.session_state.conversation_id = item["id"]
                st.session_state.messages = item.get("messages", [])
                st.session_state.page = "Legal AI Chat"
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


def render_feature_deck(prefix):
    st.html('<h3 class="section-title">Legal command suite</h3>')
    cols = st.columns(4)
    for index, (title, body, prompt) in enumerate(FEATURES):
        with cols[index % 4]:
            st.html(
                f"""
                <article class="feature-card">
                    <span>0{index + 1}</span>
                    <strong>{escape(title)}</strong>
                    <p>{escape(body)}</p>
                </article>
                """
            )
            if st.button(f"Open {title}", key=f"{prefix}-feature-{index}", width="stretch"):
                st.session_state.pending_question = prompt
                st.session_state.page = "Legal AI Chat"
                st.rerun()


def render_message(message, index):
    if message["role"] == "user":
        st.html(f'<div class="message user-message">{escape(message["content"])}</div>')
        return
    with st.container(key=f"assistant-message-{index}"):
        st.markdown(message["content"])
        if message.get("sources") is not None:
            st.caption(t("not_advice"))
            if st.session_state.show_sources:
                render_sources(message.get("sources", []))
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


def ask_question(user, question):
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = str(uuid4())
    st.session_state.messages.append({"role": "user", "content": question})
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
            answer, sources = rag.generate_answer(
                question,
                k=st.session_state.retrieval_depth,
                history=history,
            )
        elapsed = time.perf_counter() - started_at
        status.update(
            label=t("chat_done") if sources is None else t("done").format(seconds=elapsed),
            state="complete",
        )

    assistant_message = {"role": "assistant", "content": answer, "sources": sources}
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

    render_feature_deck("dash")

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
    typed = st.chat_input(t("question_placeholder"), key="legal_chat_input")
    if typed:
        prompt = typed
    if prompt:
        try:
            ask_question(user, prompt)
        except Exception as error:
            st.error("MAAT could not generate an answer. Check GROQ_API_KEY, the local embedding model, and the Chroma collection.", icon=":material/error:")
            with st.expander("Error details"):
                st.code(str(error))


def render_documents(user):
    st.html(f'<section class="page-hero compact"><h1>{escape(t("documents"))}</h1><p>Upload legal files for user-specific tracking. Prototype uploads are stored separately from the shared vector corpus.</p></section>')
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt", "md"])
    if uploaded and st.button("Save document", type="primary"):
        try:
            record = save_upload(user["id"], uploaded)
            st.success(f"Stored {record['filename']} for your account.")
        except Exception as error:
            st.error(str(error))

    uploads = list_uploads(user["id"])
    if not uploads:
        st.info("No uploaded documents yet.")
    for item in uploads:
        st.html(f'<div class="list-row"><strong>{escape(item["filename"])}</strong><span>{item["size"]:,} bytes | {escape(item["rag_status"])}</span></div>')


def render_topics():
    st.html(f'<section class="page-hero compact"><h1>{escape(t("topics"))}</h1><p>Fast entry points for common Egyptian legal research areas.</p></section>')
    cols = st.columns(3)
    for i, (title, body, icon) in enumerate(TOPICS):
        with cols[i % 3]:
            if st.button(f":material/{icon}: {title}", key=f"topic-{title}", width="stretch"):
                st.session_state.pending_question = f"Explain key Egyptian law principles for {title}."
                st.session_state.page = "Legal AI Chat"
                st.rerun()
            st.caption(body)


def render_research_studio(user):
    st.html(
        """
        <section class="page-hero compact">
            <p class="kicker">MAAT RESEARCH STUDIO</p>
            <h1>Build a sharper legal prompt</h1>
            <p>Turn raw facts into a structured legal research request, then send it into the existing Egyptian-law RAG assistant.</p>
        </section>
        """
    )
    render_feature_deck("studio")
    st.html('<h3 class="section-title">Draft a legal research brief</h3>')
    with st.form("research_studio_form", border=False):
        matter_type = st.selectbox("Matter type", ["Contract", "Civil liability", "Criminal procedure", "Personal status", "Commercial", "Administrative"])
        facts = st.text_area("Facts or document excerpt", height=180, placeholder="Paste the key facts, clause, or legal document excerpt here.")
        output_style = st.radio("Answer style", STUDIO_STYLES, horizontal=True)
        submitted = st.form_submit_button("Send to MAAT", type="primary", width="stretch")
        if submitted:
            if not facts.strip():
                st.error("Add facts or a document excerpt before sending.")
            else:
                st.session_state.pending_question = (
                    f"Analyze this {matter_type.lower()} matter under Egyptian law. "
                    f"Use the style: {output_style}. "
                    "Separate source-supported points from assumptions and list practical next questions.\n\n"
                    f"Facts:\n{facts.strip()}"
                )
                st.session_state.page = "Legal AI Chat"
                st.rerun()


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
    st.html(
        """
        <section class="page-hero compact">
            <h1>About MAAT</h1>
            <p>MAAT combines an Egyptian legal retrieval pipeline with a refined Streamlit interface. The prototype uses JSON persistence for accounts, preferences, conversations, and upload metadata; migrate to SQLite or PostgreSQL before public production use.</p>
        </section>
        """
    )
    st.info("The RAG workflow remains connected to the existing Chroma collection and Groq configuration. MAAT does not fabricate guaranteed legal advice.")
    for model in DEFAULT_CHAT_MODELS:
        st.badge(model, icon=":material/check_circle:", color="green")


def render_app():
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
    elif page == "Documents":
        render_documents(user)
    elif page == "Legal Topics":
        render_topics()
    elif page == "Research Studio":
        render_research_studio(user)
    elif page == "Profile":
        render_profile(user)
    elif page == "Settings":
        render_settings(user)
    else:
        render_about()


st.set_page_config(page_title="MAAT | Egyptian Legal AI", page_icon=":material/balance:", layout="wide")
init_state()
inject_styles()
render_app()