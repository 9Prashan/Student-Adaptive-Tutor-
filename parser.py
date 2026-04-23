"""
parser.py
---------
1. parse_bot_response()    — splits SOLUTION_CHECKER / PROGRESS_TRACKER / MENTOR
2. parse_content()         — converts raw DSL into clean Markdown + LaTeX
3. render_mentor_message() — final cleanup + safe auto-LaTeX pass
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BOT RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_bot_response(raw: str) -> dict:
    result = {
        "solution_checker": None,
        "progress_tracker": None,
        "mentor":           raw.strip(),
        "is_conclude":      "@conclude" in raw,
        "raw":              raw,
    }

    cleaned = raw.strip().lstrip("[").rstrip("]")

    sc_match = re.search(
        r"SOLUTION_CHECKER\s*:\s*(.*?)(?=PROGRESS_TRACKER|MENTOR|$)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    pt_match = re.search(
        r"PROGRESS_TRACKER\s*:\s*(.*?)(?=MENTOR|$)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    mentor_match = re.search(
        r"MENTOR\s*:\s*(.*)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )

    def _clean_field(val: str):
        val = val.strip().rstrip(",").strip()
        val = re.sub(r"^<(.*)>$", r"\1", val.strip())
        val = val.strip()
        return None if val.lower() in ("none", "") else val

    if sc_match:
        result["solution_checker"] = _clean_field(sc_match.group(1))
    if pt_match:
        result["progress_tracker"] = _clean_field(pt_match.group(1))
    if mentor_match:
        mentor_text = mentor_match.group(1).strip().rstrip("]").strip()
        result["mentor"] = mentor_text

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONTENT / PROBLEM-SOLUTION DSL PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_content(raw: str) -> str:
    if "string(" not in raw and "latex(" not in raw:
        return _clean_latex_escapes(raw)

    lines = []
    for token_type, token_value in _tokenize(raw):
        if token_type == "string":
            line = _process_inline_latex(token_value)
            if line.strip():
                lines.append(line.strip())
        elif token_type == "latex":
            lines.append(f"$${_clean_latex_escapes(token_value)}$$")
        elif token_type == "evaluation_expression":
            lines.append(_parse_evaluation_expression(token_value))
        elif token_type == "objective_answer_types":
            lines.append(_parse_objective_answer_types(token_value))
        elif token_type == "text":
            text = token_value.strip().strip(",[]")
            if text:
                lines.append(_process_inline_latex(text))

    result = "\n\n".join(l for l in lines if l.strip())
    return result if result else _clean_latex_escapes(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RENDER MENTOR MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_mentor_message(mentor_text: str) -> str:
    text = mentor_text

    # 1. Strip @conclude tag
    text = re.sub(r"@conclude\s*:?\s*<?[^>\n]*>?", "", text).strip()

    # 2. Strip lone leading < or trailing >
    text = re.sub(r"^<\s*", "", text)
    text = re.sub(r"\s*>$", "", text)

    # 3. Process inline latex(...) DSL if model used it
    text = _process_inline_latex(text)

    # 4. Clean over-escaped backslashes
    text = _clean_latex_escapes(text)

    # 5. Fix broken bold: remove ** that ended up INSIDE a $...$ block
    text = _fix_bold_inside_latex(text)

    # 6. Auto-wrap bare math — SAFE mode (only unambiguous standalone equations)
    text = _safe_auto_latexify(text)

    # 7. Bold Step N: labels that aren't already bold
    text = re.sub(r"(?m)^(?<!\*\*)(Step\s+\d+\s*:)", r"**\1**", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAFE AUTO-LATEXIFY
#     Only wraps expressions we are 100% sure are math.
#     Does NOT touch anything already inside $...$ or $$...$$
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auto_latexify(text: str) -> str:
    """
    Process the text line by line.
    For each line, split on existing LaTeX spans and only touch the plain parts.
    """
    lines = text.split("\n")
    out = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            out.append(line)
            continue
        out.append(_latexify_plain_segments(line))

    return "\n".join(out)


# Patterns applied ONLY to plain-text segments (not already in LaTeX)
# Each entry: (compiled_regex, replacement_string_or_callable)
# ORDER MATTERS — more specific patterns first
_SAFE_PATTERNS = [
    # ── Percentages: 10% 6.5%  ──────────────────────────────────────────────
    (re.compile(r"(?<!\$)(?<![\\])(\b\d+(?:\.\d+)?)\s*%(?!\$)"),
     r"$\1\\%$"),

    # ── Currency amounts: ₹2450 ₹50 ────────────────────────────────────────
    (re.compile(r"(?<!\$)(₹\s*\d+(?:[,\d]*)?(?:\.\d+)?)(?!\$)"),
     lambda m: f"${m.group(1).replace(' ','')}$"),

    # ── Pure standalone equations with = sign (whole expression on one match)
    #    e.g.  g + b = 65   50g + 30b = 2450   b = 65 - g   20g = 500
    #    Must NOT already be inside $...$
    (re.compile(
        r"(?<!\$)(?<![=<>!])"
        r"(\b(?:\d*[A-Za-z]\w*|\d+(?:\.\d+)?)"   # left-hand start
        r"(?:\s*[\+\-]\s*(?:\d*[A-Za-z]\w*|\d+(?:\.\d+)?))*"  # more terms
        r"\s*=\s*"                                  # equals
        r"(?:\d*[A-Za-z]\w*|\d+(?:\.\d+)?)"        # right-hand start
        r"(?:\s*[\+\-\*\/]\s*(?:\d*[A-Za-z]\w*|\d+(?:\.\d+)?))*)"  # more rhs terms
        r"(?!\$)(?![=])"
    ), lambda m: f"${m.group(1).strip()}$"),

    # ── Fractions: 500/20  1/x  ─────────────────────────────────────────────
    (re.compile(r"(?<!\$)(?<!\w)(\d+\s*/\s*\d+)(?!\$)(?!\w)"),
     lambda m: f"${m.group(1).strip()}$"),
]


def _latexify_plain_segments(line: str) -> str:
    """
    Split a line into [plain, $latex$, plain, $$latex$$, ...] segments.
    Apply _SAFE_PATTERNS only to the plain segments.
    """
    # Regex to find existing LaTeX spans ($$...$$ before $...$)
    latex_span = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)")
    segments = latex_span.split(line)

    result = []
    for seg in segments:
        if seg.startswith("$"):
            # Already LaTeX — never touch
            result.append(seg)
        else:
            for pattern, repl in _SAFE_PATTERNS:
                if callable(repl):
                    seg = pattern.sub(repl, seg)
                else:
                    seg = pattern.sub(repl, seg)
            result.append(seg)
    return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIX BOLD MARKERS THAT LEAKED INSIDE LATEX
# ─────────────────────────────────────────────────────────────────────────────

def _fix_bold_inside_latex(text: str) -> str:
    """
    When the model writes  $**g + b** = 65$  the ** ends up inside the math.
    Move them outside: **$g + b = 65$**
    Also remove stray ** that sit right against a $ with no matching pair inside.
    """
    # Remove ** that are directly inside $...$
    def clean_math(m):
        inner = m.group(1).replace("**", "")
        return f"${inner}$"

    text = re.sub(r"\$([^$]*?\*\*[^$]*?)\$", clean_math, text)

    # Fix dangling ** next to $ — e.g.  **$g = 25$**  is fine, leave it
    # But  $g = 25**$  or  $**g = 25$  → remove the **
    text = re.sub(r"\$\*\*([^$]+)\$", r"$\1$", text)
    text = re.sub(r"\$([^$]+)\*\*\$", r"$\1$", text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DSL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str):
    i, buf = 0, []
    known = ["string","latex","evaluation_expression",
             "objective_answer_types","type","continuous"]
    while i < len(text):
        matched = next((t for t in known if text[i:].startswith(t+"(")), None)
        if matched:
            if buf:
                yield ("text", "".join(buf)); buf = []
            start = i + len(matched) + 1
            depth, j = 1, start
            while j < len(text) and depth > 0:
                if text[j] == "(": depth += 1
                elif text[j] == ")": depth -= 1
                j += 1
            inner = text[start:j-1]
            if matched in ("string","continuous"): yield ("string", inner)
            elif matched == "latex":               yield ("latex", inner)
            elif matched == "evaluation_expression": yield ("evaluation_expression", inner)
            elif matched == "objective_answer_types": yield ("objective_answer_types", inner)
            i = j
        else:
            buf.append(text[i]); i += 1
    if buf: yield ("text", "".join(buf))


def _process_inline_latex(text: str) -> str:
    def repl(m): return f"${_clean_latex_escapes(m.group(1))}$"
    return _replace_token(text, "latex", repl)


def _replace_token(text: str, token: str, replacer) -> str:
    out, i, search = [], 0, token + "("
    while i < len(text):
        idx = text.find(search, i)
        if idx == -1: out.append(text[i:]); break
        out.append(text[i:idx])
        start = idx + len(search)
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "(": depth += 1
            elif text[j] == ")": depth -= 1
            j += 1
        inner = text[start:j-1]
        class _FM:
            def group(self, n): return inner
        out.append(replacer(_FM())); i = j
    return "".join(out)


def _clean_latex_escapes(text: str) -> str:
    text = text.replace("\\\\\\\\", "\\")
    text = text.replace("\\\\", "\\")
    return text


def _parse_evaluation_expression(inner: str) -> str:
    lhs_m = re.search(r"lhs\((\[.*?\])\)", inner, re.DOTALL)
    rhs_m = re.search(r"rhs\((\[.*?\])\)", inner, re.DOTALL)
    lhs = " ".join(_process_inline_latex(p) for p in
                   (_extract_string_values(lhs_m.group(1)) if lhs_m else []) if p.strip())
    rhs = " ".join(_process_inline_latex(p) for p in
                   (_extract_string_values(rhs_m.group(1)) if rhs_m else []) if p.strip())
    return f"{lhs} $=$ {rhs}" if lhs and rhs else f"{lhs}{rhs}"


def _parse_objective_answer_types(inner: str) -> str:
    lines = ["**Choose the correct answer:**\n"]
    values = _extract_string_values(inner)
    for i in range(0, len(values), 2):
        label = values[i].strip() if i < len(values) else ""
        desc  = values[i+1].strip() if i+1 < len(values) else ""
        if label:
            lines.append(f"- **({label})** {_process_inline_latex(desc)}")
    return "\n".join(lines)


def _extract_string_values(text: str) -> list:
    return [val for _, val in _tokenize(text)
            if re.sub(r"latex\([^)]*\)", "", val).strip()]