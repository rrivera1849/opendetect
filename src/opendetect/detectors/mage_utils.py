"""Utilities specifically for the MAGE detector."""

from __future__ import annotations

import re
from itertools import chain

from cleantext import clean


class MosesPunctNormalizer:
    EXTRA_WHITESPACE = [
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]

    NORMALIZE_UNICODE = [
        ("„", r'"'), ("“", r'"'), ("”", r'"'),
        ("–", r"-"), ("—", r" - "), (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"), ("‚", r"'"), ("’", r"'"),
        (r"''", r'"'), ("´´", r'"'), ("…", r"..."),
    ]

    FRENCH_QUOTES = [
        ("\u00A0«\u00A0", r'"'), ("«\u00A0", r'"'), ("«", r'"'),
        ("\u00A0»\u00A0", r'"'), ("\u00A0»", r'"'), ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [
        ("\u00A0%", r"%"), ("nº\u00A0", "nº "), ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"), ("\u00A0cm", r" cm"), ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"), ("\u00A0;", r";"), (",\u00A0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]
    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),
    ]
    DE_ES_CZ_CS_FR = [("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>")]
    OTHER = [("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>")]

    def __init__(self, lang="en", penn=True, norm_quote_commas=True, norm_numbers=True):
        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]
        if penn:
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)
        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)
        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)

        self.substitutions = list(chain(*self.substitutions))

    def normalize(self, text: str) -> str:
        for regexp, substitution in self.substitutions:
            text = re.sub(regexp, substitution, str(text))
        return text.strip()


def _tokenization_norm(text: str) -> str:
    text = text.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(
        ' !', '!').replace(' ;', ';').replace(' \'', '\'').replace(' ’ ', '\'').replace(
        ' :', ':').replace('<newline>', '\n').replace('`` ', '"').replace(
        ' \'\'', '"').replace('\'\'', '"').replace('.. ', '... ').replace(
        ' )', ')').replace('( ', '(').replace(' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(' i\'', ' I\'').replace('\\\'', '\'').replace('\n ', '\n').strip()
    return text


def _clean_text(text: str) -> str:
    plm_special_tokens = r'(\<pad\>)|(\<s\>)|(\<\/s\>)|(\<unk\>)|(\<\|endoftext\|\>)'
    text = re.sub(plm_special_tokens, "", text)

    moses_norm = MosesPunctNormalizer()
    text = moses_norm.normalize(text)
    text = _tokenization_norm(text)
    
    text = clean(
        text,
        fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=True,
        no_urls=True, no_emails=True, no_phone_numbers=True,
        no_numbers=False, no_digits=False, no_currency_symbols=False,
        no_punct=False, replace_with_punct="", replace_with_url="",
        replace_with_email="", replace_with_phone_number="",
        replace_with_number="<NUMBER>", replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>", lang="en"
    )
    
    punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
    text = re.sub(punct_pattern, '', text)
    spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
    text = re.sub(spe_pattern, '', text)
    text = " ".join(text.split())
    return text


def _rm_line_break(text: str) -> str:
    text = text.replace("\n", "\\n")
    text = re.sub(r'(?:\\n)*\\n', r'\\n', text)
    text = re.sub(r'^.{0,3}\\n', '', text)
    text = text.replace("\\n", " ")
    return text


def preprocess_mage(text: str) -> str:
    """MAGE specific text preprocessing pipeline."""
    text = _rm_line_break(text)
    text = _clean_text(text)
    return text
