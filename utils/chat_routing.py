import re
import unicodedata


ENGLISH_SMALL_TALK = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are you doing",
    "how you doing",
    "how u doing",
    "how ya doing",
    "how are you today",
    "how is it going",
    "how s it going",
    "what is up",
    "what s up",
    "and you",
    "thanks",
    "thank you",
    "thank you so much",
    "bye",
    "goodbye",
    "see you",
    "who are you",
    "what is your name",
    "what can you do",
    "are you an ai",
    "are you a bot",
    "tell me a joke",
    "make me laugh",
    "what is your favorite color",
    "what is your favorite food",
    "do you like coffee",
}

ARABIC_SMALL_TALK = {
    "اهلا",
    "اهلين",
    "اهلا وسهلا",
    "مرحبا",
    "السلام عليكم",
    "وعليكم السلام",
    "كيف حالك",
    "كيفك",
    "ازيك",
    "عامل ايه",
    "عاملة ايه",
    "وانت",
    "شكرا",
    "شكرا لك",
    "مع السلامة",
    "صباح الخير",
    "مساء الخير",
    "من انت",
    "ما اسمك",
    "ماذا تستطيع ان تفعل",
}

ARABIC_CREATOR_QUESTIONS = {
    "من بنى مات",
    "من بنى ماعت",
    "من صنع مات",
    "من صنع ماعت",
    "من طور مات",
    "من طور ماعت",
    "من هو مطور مات",
    "من هو مطور ماعت",
    "من هو مؤسس مات",
    "من هو مؤسس ماعت",
    "مين بنى مات",
    "مين بنى ماعت",
    "مين عمل مات",
    "مين عمل ماعت",
    "مين طور مات",
    "مين طور ماعت",
    "مين هو احمد مدحت",
    "من هو احمد مدحت",
}


def _normalize_small_talk(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char) and char != "\u0640")
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return re.sub(r"[^\w\s]", " ", text).split()


def is_about_creator(text):
    normalized = " ".join(_normalize_small_talk(text or ""))
    if (
        normalized in ARABIC_CREATOR_QUESTIONS
        or "ahmed medhat" in normalized
        or "احمد مدحت" in normalized
        or re.search(r"\bcabst\b", normalized)
    ):
        return True

    return bool(
        re.search(
            r"\bwho\b.*\b(?:built|made|created|developed|programmed|founded)\b.*\b(?:maat|you|this|assistant|system)\b",
            normalized,
        )
        or re.search(
            r"\bwho\b.*\b(?:creator|developer|founder|builder|programmer)\b.*\b(?:maat|you|this|assistant|system)\b",
            normalized,
        )
        or re.search(
            r"\b(?:creator|developer|founder|builder|programmer)\b.*\bof\b.*\bmaat\b",
            normalized,
        )
        or re.search(
            r"\bmaat\b.*\b(?:creator|developer|founder|builder|programmer)\b",
            normalized,
        )
        or re.search(
            r"\b(?:مين|من)\b.*\b(?:بنى|بنا|عمل|صنع|طور|برمج|انشا|اسس|مطور|مبرمج|مؤسس)\b.*\b(?:maat|مات|ماعت)\b",
            normalized,
        )
    )


def is_small_talk(text):
    normalized = " ".join(_normalize_small_talk(text or ""))
    if is_about_creator(text) or normalized in ENGLISH_SMALL_TALK or normalized in ARABIC_SMALL_TALK:
        return True

    return bool(
        re.fullmatch(
            r"(?:hi|hello|hey)(?: there)? how are you(?: doing)?(?: today)?",
            normalized,
        )
        or re.fullmatch(
            r"(?:h+i+|h+e+y+|h+e+l+o+|yo+|howdy+|hiowdy+)(?: there)?",
            normalized,
        )
        or re.fullmatch(r"(?:اهلا|مرحبا) كيف حالك", normalized)
    )
