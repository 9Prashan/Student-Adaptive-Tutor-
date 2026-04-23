"""
app.py  —  EduBot Streamlit UI
Run with:  streamlit run app.py
"""

import base64
import streamlit as st
from EduBot import Bot
from parser import parse_bot_response, parse_content, render_mentor_message

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduBot — AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px;
}

/* ── Main area ── */
.main .block-container { padding-top: 1.5rem; max-width: 900px; }

/* ── Chat bubbles ── */
.bot-bubble-wrap  { display: flex; justify-content: flex-start; margin-bottom: 0.5rem; }
.user-bubble-wrap { display: flex; justify-content: flex-end;   margin-bottom: 0.5rem; }

.bot-bubble {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.2rem;
    max-width: 82%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    color: #1e293b !important;
}
.bot-bubble * { color: #1e293b !important; }

.user-bubble {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: #ffffff !important;
    border-radius: 18px 4px 18px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 70%;
    box-shadow: 0 1px 4px rgba(99,102,241,0.3);
}
.user-bubble * { color: #ffffff !important; }

.bubble-avatar {
    font-size: 1.4rem;
    margin-right: 0.5rem;
    align-self: flex-start;
    margin-top: 0.2rem;
    flex-shrink: 0;
}

/* ── Info panels — FIX: explicit dark text ── */
.info-panel {
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.83rem;
    color: #1e293b !important;
}
.info-panel * { color: #1e293b !important; }
.panel-checker { background: #fef08a; border-left: 4px solid #ca8a04; }
.panel-tracker { background: #bae6fd; border-left: 4px solid #0284c7; }
.panel-label {
    font-weight: 700;
    font-size: 0.68rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    color: #374151 !important;
    opacity: 1 !important;
}

/* ── Conclude banner ── */
.conclude-banner {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
    font-weight: 600;
    color: #15803d !important;
    margin-top: 0.5rem;
}

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-active { background: #dcfce7; color: #15803d !important; }
.badge-done   { background: #e0f2fe; color: #0369a1 !important; }
.badge-idle   { background: #f1f5f9; color: #64748b !important; }

/* ── Image preview ── */
.img-preview {
    border-radius: 10px;
    border: 2px dashed #6366f1;
    padding: 0.4rem;
    margin-top: 0.4rem;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

/* ── Chat input ── */
.stChatInput > div { border-radius: 12px !important; border: 2px solid #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "bot":                None,
        "messages":           [],
        "chat_display":       [],
        "session_active":     False,
        "session_done":       False,
        "language":           "English",
        "problem_formatted":  "",
        "solution_formatted": "",
        "uploaded_image_b64": None,   # base64 string of uploaded image
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_bot() -> Bot:
    if st.session_state.bot is None:
        st.session_state.bot = Bot()
    return st.session_state.bot


def image_to_b64(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.read()).decode("utf-8")


def render_bot_bubble(parsed: dict):
    sc          = parsed.get("solution_checker")
    pt          = parsed.get("progress_tracker")
    mentor_raw  = parsed.get("mentor", "")
    mentor_text = render_mentor_message(mentor_raw)
    is_conclude = parsed.get("is_conclude", False)

    # ── panels (Solution Checker + Progress Tracker) ──
    panels_html = ""
    if sc:
        panels_html += (
            f'<div class="info-panel panel-checker">'
            f'<div class="panel-label">✅ Solution Checker</div>'
            f'<span style="color:#1e293b">{sc}</span></div>'
        )
    if pt:
        panels_html += (
            f'<div class="info-panel panel-tracker">'
            f'<div class="panel-label">📊 Progress Tracker</div>'
            f'<span style="color:#1e293b">{pt}</span></div>'
        )

    if panels_html:
        st.markdown(
            f'<div class="bot-bubble-wrap">'
            f'<span class="bubble-avatar">🎓</span>'
            f'<div class="bot-bubble">{panels_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── mentor message (rendered via st.markdown for LaTeX support) ──
    _, col = st.columns([0.04, 0.96])
    with col:
        with st.container(border=True):
            st.markdown(mentor_text)
            if is_conclude:
                st.markdown(
                    '<div class="conclude-banner">🎉 Great work! You\'ve mastered this problem!</div>',
                    unsafe_allow_html=True,
                )


def render_user_bubble(text: str):
    st.markdown(
        f'<div class="user-bubble-wrap">'
        f'<div class="user-bubble">{text}</div>'
        f'<span class="bubble-avatar" style="margin-left:0.5rem;margin-right:0">🧑‍🎓</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_content_card(label: str, content: str, color: str, border: str):
    parsed = parse_content(content)
    st.markdown(
        f'<div style="background:{color};border-left:4px solid {border};'
        f'border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.4rem;">'
        f'<span style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#374151;opacity:0.7">{label}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(parsed)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduBot")
    st.markdown("*AI-powered adaptive tutor*")
    st.divider()

    # ── Input mode toggle ──
    input_mode = st.radio(
        "Input mode",
        ["✏️ Type Problem", "🖼️ Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    problem_input  = ""
    solution_input = ""

    if input_mode == "✏️ Type Problem":
        st.markdown("### 📝 Problem")
        problem_input = st.text_area(
            "problem", height=140,
            placeholder="Paste the problem here…",
            label_visibility="collapsed",
        )
        st.markdown("### 💡 Solution")
        solution_input = st.text_area(
            "solution", height=180,
            placeholder="Paste the solution here…",
            label_visibility="collapsed",
        )

    else:  # Image upload mode
        st.markdown("### 🖼️ Upload Question Image")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded question", use_container_width=True)
            uploaded_file.seek(0)
            st.session_state.uploaded_image_b64 = image_to_b64(uploaded_file)
            st.session_state.uploaded_image_type = uploaded_file.type
            problem_input  = "[Image uploaded — bot will read the question from the image]"
            solution_input = "[Bot will guide student based on the image question]"
        else:
            st.session_state.uploaded_image_b64  = None
            st.caption("Supported: PNG, JPG, JPEG, WEBP")

    st.markdown("### 🌐 Language")
    language = st.selectbox(
        "language",
        ["English", "Hindi", "Spanish", "French", "German", "Arabic", "Bengali", "Tamil"],
        label_visibility="collapsed",
    )
    st.session_state.language = language

    st.divider()

    start_col, reset_col = st.columns(2)
    with start_col:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with reset_col:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn:
        keys_to_clear = [
            "messages", "chat_display", "session_active", "session_done",
            "problem_formatted", "solution_formatted", "uploaded_image_b64",
        ]
        for key in keys_to_clear:
            if key in ("messages", "chat_display"):
                st.session_state[key] = []
            elif key in ("session_active", "session_done"):
                st.session_state[key] = False
            else:
                st.session_state[key] = "" if key != "uploaded_image_b64" else None
        st.rerun()

    st.divider()
    if st.session_state.session_done:
        st.markdown('<span class="status-badge badge-done">✔ Session Complete</span>', unsafe_allow_html=True)
    elif st.session_state.session_active:
        st.markdown('<span class="status-badge badge-active">● Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-idle">○ Idle</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🎓  EduTrainer — Adaptive Learning Assistant")

# ── Start session ──
if start_btn:
    has_image   = st.session_state.uploaded_image_b64 is not None
    has_text    = problem_input.strip() and solution_input.strip()
    image_mode  = (input_mode == "🖼️ Upload Image")

    if image_mode and not has_image:
        st.error("Please upload an image first.")
    elif not image_mode and not has_text:
        st.error("Please enter both a problem and a solution.")
    else:
        bot = get_bot()
        with st.spinner("📚 Setting up your session…"):

            if image_mode and has_image:
                # Pass image directly to the model via a special message
                prob_fmt  = "Question extracted from uploaded image."
                soln_fmt  = "Guide the student through solving the problem shown in the image."
                st.session_state.problem_formatted  = prob_fmt
                st.session_state.solution_formatted = soln_fmt

                # Build image-aware initial messages
                from prompts import bot_sys_prompt
                msgs = [
                    {"role": "system", "content": bot_sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": st.session_state.uploaded_image_type,
                                    "data": st.session_state.uploaded_image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "The image above contains a math/science problem. "
                                    "Act as the EduBot tutor team (Solution Checker, Progress Tracker, Mentor) "
                                    "and guide the student to solve it step by step. "
                                    "Start with a motivational message and the first guiding question. "
                                    "Use the format:\n"
                                    "[\nSOLUTION_CHECKER: <none>,\n"
                                    "PROGRESS_TRACKER: <first milestone>,\n"
                                    "MENTOR: <message>\n]"
                                ),
                            },
                        ],
                    },
                ]
            else:
                prob_fmt = bot.format_content_sync(problem_input)
                soln_fmt = bot.format_content_sync(solution_input)
                st.session_state.problem_formatted  = prob_fmt
                st.session_state.solution_formatted = soln_fmt
                msgs = bot.build_initial_messages(prob_fmt, soln_fmt)

            raw_response = bot.get_bot_response_sync(msgs)

            display_response = raw_response
            if language != "English":
                display_response = bot.translate_sync(raw_response, language)

            msgs.append({"role": "assistant", "content": raw_response})
            st.session_state.messages       = msgs
            st.session_state.session_active = True
            st.session_state.session_done   = False
            st.session_state.chat_display   = [{
                "role":    "bot",
                "content": display_response,
                "parsed":  parse_bot_response(display_response),
            }]
            if "@conclude" in raw_response:
                st.session_state.session_done   = True
                st.session_state.session_active = False
        st.rerun()

# ── Problem / solution preview ──
if st.session_state.problem_formatted:
    with st.expander("📌 Problem & Solution Reference", expanded=False):
        if st.session_state.uploaded_image_b64:
            b64 = st.session_state.uploaded_image_b64
            mime = getattr(st.session_state, "uploaded_image_type", "image/png")
            st.markdown(
                f'<img src="data:{mime};base64,{b64}" '
                f'style="max-width:100%;border-radius:10px;border:2px dashed #6366f1">',
                unsafe_allow_html=True,
            )
        else:
            render_content_card("Problem",            st.session_state.problem_formatted,  "#eff6ff", "#3b82f6")
            render_content_card("Reference Solution", st.session_state.solution_formatted, "#f0fdf4", "#22c55e")

st.divider()

# ── Chat history ──
if st.session_state.chat_display:
    for entry in st.session_state.chat_display:
        if entry["role"] == "bot":
            render_bot_bubble(entry["parsed"])
        else:
            render_user_bubble(entry["content"])
else:
    if not st.session_state.session_active:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem;color:#94a3b8;">
                <div style="font-size:3rem;">🎓</div>
                <h3 style="color:#64748b;">Ready to learn?</h3>
                <p>Type your problem in the sidebar <strong>or upload an image</strong>,<br>
                then click <strong>▶ Start</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Chat input ──
if st.session_state.session_active and not st.session_state.session_done:
    user_input = st.chat_input("Type your answer or question here…")

    if user_input:
        bot  = get_bot()
        lang = st.session_state.language

        input_for_model = user_input
        if lang != "English":
            with st.spinner("Translating…"):
                input_for_model = bot.translate_sync(user_input, "English")

        st.session_state.chat_display.append({"role": "user", "content": user_input, "parsed": None})
        st.session_state.messages.append({"role": "user", "content": "STUDENT_RESPONSE:\n" + input_for_model})

        with st.spinner("🤔 Thinking…"):
            raw_response = bot.get_bot_response_sync(st.session_state.messages)

        display_response = raw_response
        if lang != "English":
            with st.spinner("Translating response…"):
                display_response = bot.translate_sync(raw_response, lang)

        st.session_state.messages.append({"role": "assistant", "content": raw_response})
        parsed = parse_bot_response(display_response)
        st.session_state.chat_display.append({"role": "bot", "content": display_response, "parsed": parsed})

        if "@conclude" in raw_response:
            st.session_state.session_done   = True
            st.session_state.session_active = False

        st.rerun()

elif st.session_state.session_done:
    st.success("🎉 Session complete! Click ↺ Reset to start a new problem.")