import hashlib
import json
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_engine import RAGEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


def get_secret(name):
    """
    Load a secret from:
    1. Streamlit Cloud Secrets
    2. Local .env file
    """

    # Streamlit Cloud
    try:
        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    # Local .env
    value = os.getenv(name)

    return value


OLLAMA_API_KEY = get_secret("OLLAMA_API_KEY")


# ============================================================
# CHECK OLLAMA API KEY
# ============================================================

if not OLLAMA_API_KEY:

    st.error(
        "OLLAMA_API_KEY was not found.\n\n"
        "For local development, add it to your .env file.\n"
        "For Streamlit Cloud, add it under App Settings → Secrets."
    )

    st.stop()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666666;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            padding: 1rem 1.2rem;
            border-radius: 0.7rem;
            border: 1px solid #e5e7eb;
            margin-top: 1rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OLLAMA CLOUD CONFIGURATION
# ============================================================

OLLAMA_MODEL = "gpt-oss:20b-cloud"

OLLAMA_BASE_URL = "https://ollama.com"


# ============================================================
# NORMAL LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
        },
    )


llm = load_llm()


# ============================================================
# JSON LLM
# ============================================================

@st.cache_resource
def load_quiz_llm():

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        format="json",
        base_url=OLLAMA_BASE_URL,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
        },
    )


quiz_llm = load_quiz_llm()


# ============================================================
# QUESTION PROMPT
# ============================================================

question_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant.

Answer the user's question using ONLY the study material
provided below.

Study material:
{context}

Question:
{question}

Rules:

- Answer clearly and simply.
- Use the study material as your source.
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not available in the study material,
  say:

"I couldn't find the answer in the uploaded study material."
"""
)


# ============================================================
# EXPLAIN SIMPLY PROMPT
# ============================================================

explain_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant.

Use ONLY the study material provided below.

Study material:
{context}

Topic:
{question}

Instructions:

- Explain the topic in 4-6 short points.
- Use only facts explicitly stated in the study material.
- Preserve the terminology used in the study material.
- Use examples only if they appear in the study material.
- Do not create new examples.
- Do not add analogies.
- Do not add outside knowledge.
- Do not add general programming knowledge.
- Do not add unsupported conclusions.
- Do not mention unrelated topics.
"""
)


# ============================================================
# SUMMARY PROMPT
# ============================================================

summary_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant.

Use the study material provided below to summarize
the requested topic.

Study material:
{context}

Topic:
{question}

Instructions:

- Give a concise summary.
- Include the main points supported by the study material.
- Use simple language.
- Do not add unrelated information.
- Do not use outside knowledge.
"""
)


# ============================================================
# MCQ JSON PROMPT
# ============================================================

mcq_json_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant creating multiple-choice questions.

Requested topic:
{question}

Study material:
{context}

Create exactly {number_of_questions} questions.

STRICT RULES:

- Use ONLY the study material.
- Questions must be about the requested topic.
- Do not use outside knowledge.
- Each question must have exactly four options.
- Options must be A, B, C, and D.
- There must be exactly one correct answer.
- The correct answer must be supported by the study material.
- The explanation must support the selected answer.
- Do not create unrelated questions.
- Never invent facts.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "answer": "B",
            "explanation": "Short explanation"
        }}
    ]
}}
"""
)


# ============================================================
# QUIZ JSON PROMPT
# ============================================================

quiz_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant creating an interactive quiz.

Requested topic:
{question}

Study material:
{context}

Create EXACTLY {number_of_questions} DIFFERENT
multiple-choice questions.

STRICT RULES:

- Use ONLY the study material.
- Questions must be about the requested topic.
- Do not use outside knowledge.
- Do not invent facts.
- Each question must have exactly four options.
- Options must be A, B, C, and D.
- There must be exactly one correct answer.
- The answer must be A, B, C, or D.
- The explanation must support the correct answer.
- Every question must test a different fact, concept, definition,
  property, example, or detail from the study material.
- Do not repeat or rephrase the same question.
- Do not create unrelated questions.

IMPORTANT:

You MUST return exactly the requested number of questions
whenever the study material contains enough information.

If you previously generated questions, the new questions
must be different from them.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "answer": "B",
            "explanation": "Short explanation"
        }}
    ]
}}
"""
)


# ============================================================
# QUIZ QUESTION VALIDATION
# ============================================================

def validate_question(item):
    """
    Validate a single generated quiz question.

    Returns the cleaned question if valid.
    Otherwise returns None.
    """

    if not isinstance(item, dict):
        return None

    question_text = item.get("question")

    options = item.get("options")

    correct_answer = item.get("answer")

    explanation = item.get("explanation")

    required_options = {
        "A",
        "B",
        "C",
        "D",
    }

    # --------------------------------------------------------
    # Question text
    # --------------------------------------------------------

    if not isinstance(question_text, str):
        return None

    question_text = question_text.strip()

    if not question_text:
        return None

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    if not isinstance(options, dict):
        return None

    if set(options.keys()) != required_options:
        return None

    for letter in required_options:

        option = options.get(letter)

        if not isinstance(option, str):
            return None

        if not option.strip():
            return None

    # --------------------------------------------------------
    # Correct answer
    # --------------------------------------------------------

    if not isinstance(correct_answer, str):
        return None

    correct_answer = (
        correct_answer
        .strip()
        .upper()
    )

    if correct_answer not in required_options:
        return None

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    if not isinstance(explanation, str):
        return None

    explanation = explanation.strip()

    if not explanation:
        return None

    # --------------------------------------------------------
    # Clean question
    # --------------------------------------------------------

    item["question"] = question_text

    item["answer"] = correct_answer

    item["explanation"] = explanation

    item["options"] = {
        "A": options["A"].strip(),
        "B": options["B"].strip(),
        "C": options["C"].strip(),
        "D": options["D"].strip(),
    }

    return item


# ============================================================
# NORMALIZE QUESTION FOR DUPLICATE CHECKING
# ============================================================

def normalize_question(text):
    """
    Normalize question text so duplicate questions
    can be detected.
    """

    return " ".join(
        text
        .strip()
        .lower()
        .split()
    )


# ============================================================
# GENERATE QUIZ QUESTIONS WITH RETRY
# ============================================================

def generate_questions_with_retry(
    llm,
    prompt_template,
    context,
    question,
    number_of_questions,
    max_attempts=5,
):
    """
    Generate the requested number of valid questions.

    The function:

    1. Requests questions from the LLM.
    2. Validates every question.
    3. Removes duplicates.
    4. Requests additional questions if needed.
    5. Continues until the requested count is reached
       or the maximum number of attempts is exhausted.
    """

    valid_questions = []

    existing_question_keys = set()

    for attempt in range(max_attempts):

        remaining = (
            number_of_questions
            - len(valid_questions)
        )

        if remaining <= 0:
            break

        # ----------------------------------------------------
        # Ask for extra questions when previous attempts
        # did not produce enough valid questions.
        # ----------------------------------------------------

        request_count = remaining

        # On retry, request extra candidates.
        if attempt > 0:
            request_count = min(
                remaining + 2,
                10
            )

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt_message = prompt_template.invoke(
            {
                "context": context,
                "question": question,
                "number_of_questions": request_count,
            }
        )

        # ----------------------------------------------------
        # Call LLM
        # ----------------------------------------------------

        try:

            response = llm.invoke(
                prompt_message
            )

        except Exception as error:

            if attempt == max_attempts - 1:
                raise error

            continue

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        raw_content = response.content

        if isinstance(
            raw_content,
            list
        ):

            raw_content = "".join(
                str(part)
                for part in raw_content
            )

        if not isinstance(
            raw_content,
            str
        ):

            raw_content = str(
                raw_content
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                raw_content
            )

        except json.JSONDecodeError:

            continue

        if not isinstance(
            data,
            dict
        ):

            continue

        questions = data.get(
            "questions",
            []
        )

        if not isinstance(
            questions,
            list
        ):

            continue

        # ----------------------------------------------------
        # Validate generated questions
        # ----------------------------------------------------

        for item in questions:

            cleaned_question = validate_question(
                item
            )

            if cleaned_question is None:
                continue

            normalized = normalize_question(
                cleaned_question["question"]
            )

            # ------------------------------------------------
            # Prevent duplicate questions
            # ------------------------------------------------

            if normalized in existing_question_keys:
                continue

            existing_question_keys.add(
                normalized
            )

            valid_questions.append(
                cleaned_question
            )

            # ------------------------------------------------
            # Stop once requested count is reached
            # ------------------------------------------------

            if len(valid_questions) >= number_of_questions:
                break

        # ----------------------------------------------------
        # Check if complete
        # ----------------------------------------------------

        if len(valid_questions) >= number_of_questions:
            break

    # --------------------------------------------------------
    # Return only requested number
    # --------------------------------------------------------

    return valid_questions[
        :number_of_questions
    ]


# ============================================================
# SESSION STATE
# ============================================================

if "engine" not in st.session_state:
    st.session_state.engine = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False

if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📚 AI Study Assistant"
    )

    st.caption(
        "Upload your study material and choose a learning mode."
    )

    st.divider()

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Study Material"
    )

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        help="Upload a PDF containing your study material.",
    )

    if uploaded_file is not None:

        st.success(
            uploaded_file.name
        )

    st.divider()

    # --------------------------------------------------------
    # STUDY MODE
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Study Mode"
    )

    mode = st.selectbox(
        "Choose a mode",
        [
            "Ask a question",
            "Explain simply",
            "Summarize",
            "Generate MCQs",
            "Take a Quiz",
        ],
    )

    # --------------------------------------------------------
    # NUMBER OF QUESTIONS
    # --------------------------------------------------------

    number_of_questions = 5

    if mode in {
        "Generate MCQs",
        "Take a Quiz",
    }:

        number_of_questions = st.slider(
            "Number of questions",
            min_value=1,
            max_value=10,
            value=5,
        )

    st.divider()

    st.caption(
        "Powered by LangChain • Chroma • Ollama Cloud"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 AI Study Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Learn from your study material using AI-powered retrieval."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# REQUIRE PDF
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a study PDF from the sidebar to get started."
    )

    st.stop()


# ============================================================
# CREATE PDF SIGNATURE
# ============================================================

pdf_bytes = uploaded_file.getvalue()

pdf_signature = hashlib.sha256(
    pdf_bytes
).hexdigest()


# ============================================================
# SAVE UPLOADED PDF
# ============================================================

upload_dir = "uploads"

os.makedirs(
    upload_dir,
    exist_ok=True,
)

pdf_path = os.path.join(
    upload_dir,
    uploaded_file.name,
)


if (
    st.session_state.pdf_signature
    != pdf_signature
):

    with open(
        pdf_path,
        "wb",
    ) as file:

        file.write(
            pdf_bytes
        )


# ============================================================
# CREATE / LOAD RAG ENGINE
# ============================================================

if (
    st.session_state.engine is None
    or st.session_state.pdf_signature
    != pdf_signature
):

    with st.spinner(
        "Loading and indexing your study material..."
    ):

        try:

            engine = RAGEngine(
                pdf_path
            )

            index_result = engine.index_pdf()

            st.session_state.engine = engine

            st.session_state.pdf_signature = (
                pdf_signature
            )

            st.session_state.pdf_name = (
                uploaded_file.name
            )

            # Reset quiz

            st.session_state.quiz_data = None

            st.session_state.quiz_index = 0

            st.session_state.quiz_score = 0

            st.session_state.quiz_finished = False

            st.session_state.quiz_topic = None

        except Exception as error:

            st.error(
                f"Could not process the PDF: {error}"
            )

            st.stop()

else:

    engine = st.session_state.engine

    index_result = {
        "status": "existing",
        "chunks": engine.collection.count(),
    }


# ============================================================
# PDF STATUS
# ============================================================

col1, col2 = st.columns(
    [2, 1]
)


with col1:

    st.markdown(
        "### 📄 Current Study Material"
    )

    st.write(
        uploaded_file.name
    )


with col2:

    if index_result["status"] == "indexed":

        st.success(
            f"Indexed: {index_result['chunks']} chunks"
        )

    else:

        st.info(
            f"Indexed: {index_result['chunks']} chunks"
        )


st.divider()


# ============================================================
# QUESTION / TOPIC
# ============================================================

st.markdown(
    "### ✏️ What would you like to study?"
)

question = st.text_input(
    "Enter your question or topic",
    placeholder="Example: What is a variable?",
    label_visibility="collapsed",
)


# ============================================================
# GENERATE BUTTON
# ============================================================

button_text = (
    "Start Quiz"
    if mode == "Take a Quiz"
    else "Generate"
)

generate_clicked = st.button(
    button_text,
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN GENERATION LOGIC
# ============================================================

if generate_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question or topic."
        )

    else:

        # ----------------------------------------------------
        # RETRIEVE MATERIAL
        # ----------------------------------------------------

        with st.spinner(
            "Searching the study material..."
        ):

            try:

                retrieval = engine.retrieve(
                    question,
                    n_results=3,
                )

            except Exception as error:

                st.error(
                    f"Could not search the study material: {error}"
                )

                st.stop()

        # ----------------------------------------------------
        # CHECK RETRIEVAL
        # ----------------------------------------------------

        if not retrieval["found"]:

            st.warning(
                "I couldn't find the relevant information "
                "in the uploaded study material."
            )

        else:

            context = "\n\n".join(
                retrieval["chunks"]
            )

            # =================================================
            # TAKE QUIZ
            # =================================================

            if mode == "Take a Quiz":

                with st.spinner(
                    "Creating your quiz..."
                ):

                    try:

                        valid_questions = (
                            generate_questions_with_retry(
                                llm=quiz_llm,
                                prompt_template=quiz_prompt,
                                context=context,
                                question=question,
                                number_of_questions=number_of_questions,
                                max_attempts=5,
                            )
                        )

                    except Exception as error:

                        st.error(
                            f"Could not create quiz: {error}"
                        )

                        st.stop()

                # ------------------------------------------------
                # CHECK QUESTION COUNT
                # ------------------------------------------------

                if not valid_questions:

                    st.error(
                        "No valid quiz questions were generated."
                    )

                    st.stop()

                # ------------------------------------------------
                # IMPORTANT:
                # If fewer questions were generated, do NOT
                # silently start a smaller quiz.
                # ------------------------------------------------

                if len(valid_questions) < number_of_questions:

                    st.error(
                        f"Could generate only "
                        f"{len(valid_questions)} valid questions "
                        f"out of the requested "
                        f"{number_of_questions}."
                    )

                    st.info(
                        "The uploaded study material may not contain "
                        "enough distinct information for the requested "
                        "number of questions. Try a broader topic or "
                        "a smaller number of questions."
                    )

                    st.stop()

                # ------------------------------------------------
                # STORE QUIZ
                # ------------------------------------------------

                st.session_state.quiz_data = (
                    valid_questions
                )

                st.session_state.quiz_index = 0

                st.session_state.quiz_score = 0

                st.session_state.quiz_finished = False

                st.session_state.quiz_topic = question

                st.rerun()

            # =================================================
            # GENERATE MCQs
            # =================================================

            elif mode == "Generate MCQs":

                mcq_prompt_message = (
                    mcq_json_prompt.invoke(
                        {
                            "context": context,
                            "question": question,
                            "number_of_questions":
                                number_of_questions,
                        }
                    )
                )

                with st.spinner(
                    "Generating your MCQs..."
                ):

                    try:

                        response = quiz_llm.invoke(
                            mcq_prompt_message
                        )

                        raw_content = response.content

                        if isinstance(
                            raw_content,
                            list,
                        ):

                            raw_content = "".join(
                                str(part)
                                for part in raw_content
                            )

                        mcq_data = json.loads(
                            raw_content
                        )

                    except json.JSONDecodeError:

                        st.error(
                            "The MCQ response was not valid JSON. "
                            "Please try again."
                        )

                        st.stop()

                    except Exception as error:

                        st.error(
                            f"Could not generate MCQs: {error}"
                        )

                        st.stop()

                questions = mcq_data.get(
                    "questions",
                    [],
                )

                # ------------------------------------------------
                # VALIDATE MCQs
                # ------------------------------------------------

                valid_questions = []

                seen_questions = set()

                for item in questions:

                    cleaned_question = validate_question(
                        item
                    )

                    if cleaned_question is None:
                        continue

                    normalized = normalize_question(
                        cleaned_question["question"]
                    )

                    if normalized in seen_questions:
                        continue

                    seen_questions.add(
                        normalized
                    )

                    valid_questions.append(
                        cleaned_question
                    )

                    if (
                        len(valid_questions)
                        >= number_of_questions
                    ):
                        break

                if not valid_questions:

                    st.error(
                        "The generated MCQs did not contain "
                        "valid questions."
                    )

                    st.stop()

                # ------------------------------------------------
                # DISPLAY MCQs
                # ------------------------------------------------

                st.markdown(
                    "### 📝 Multiple Choice Questions"
                )

                for index, item in enumerate(
                    valid_questions,
                    start=1,
                ):

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"#### Question {index}"
                        )

                        st.write(
                            item["question"]
                        )

                        st.markdown(
                            f"**A.** {item['options']['A']}"
                        )

                        st.markdown(
                            f"**B.** {item['options']['B']}"
                        )

                        st.markdown(
                            f"**C.** {item['options']['C']}"
                        )

                        st.markdown(
                            f"**D.** {item['options']['D']}"
                        )

                        with st.expander(
                            "Show answer"
                        ):

                            st.success(
                                f"Correct Answer: "
                                f"{item['answer']}"
                            )

                            st.write(
                                "Explanation:",
                                item["explanation"],
                            )

            # =================================================
            # NORMAL MODES
            # =================================================

            else:

                if mode == "Ask a question":

                    prompt = (
                        question_prompt.invoke(
                            {
                                "context": context,
                                "question": question,
                            }
                        )
                    )

                elif mode == "Explain simply":

                    prompt = (
                        explain_prompt.invoke(
                            {
                                "context": context,
                                "question": question,
                            }
                        )
                    )

                else:

                    prompt = (
                        summary_prompt.invoke(
                            {
                                "context": context,
                                "question": question,
                            }
                        )
                    )

                with st.spinner(
                    "Generating answer..."
                ):

                    try:

                        response = llm.invoke(
                            prompt
                        )

                        answer = response.content

                    except Exception as error:

                        error_text = str(error)

                        if "401" in error_text:

                            st.error(
                                "Ollama authentication failed. "
                                "Please check your OLLAMA_API_KEY."
                            )

                        elif "403" in error_text:

                            st.error(
                                "Ollama denied access to the model. "
                                "Please check your Ollama Cloud account "
                                "and API key."
                            )

                        else:

                            st.error(
                                f"Could not generate the answer: "
                                f"{error}"
                            )

                        st.stop()

                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.markdown(
                    "### 🤖 AI Study Assistant"
                )

                with st.container(
                    border=True
                ):

                    st.write(
                        answer
                    )

            # =================================================
            # SOURCE CONTEXT
            # =================================================

            with st.expander(
                "📖 Show retrieved study material"
            ):

                for index, chunk in enumerate(
                    retrieval["chunks"],
                    start=1,
                ):

                    st.markdown(
                        f"**Source chunk {index}**"
                    )

                    st.write(
                        chunk
                    )

                    st.divider()


# ============================================================
# ACTIVE QUIZ
# ============================================================

if (
    mode == "Take a Quiz"
    and st.session_state.quiz_data is not None
    and not st.session_state.quiz_finished
):

    questions = (
        st.session_state.quiz_data
    )

    current_index = (
        st.session_state.quiz_index
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if current_index >= len(questions):

        st.session_state.quiz_finished = True

        st.rerun()

    current_question = (
        questions[current_index]
    )

    st.divider()

    # ========================================================
    # PROGRESS
    # ========================================================

    st.progress(
        (current_index + 1)
        / len(questions)
    )

    st.markdown(
        f"### 📝 Question {current_index + 1} "
        f"of {len(questions)}"
    )

    st.write(
        current_question["question"]
    )

    options = (
        current_question["options"]
    )

    answer = st.radio(
        "Choose your answer:",
        [
            f"A. {options['A']}",
            f"B. {options['B']}",
            f"C. {options['C']}",
            f"D. {options['D']}",
        ],
        key=f"quiz_answer_{current_index}",
    )

    if st.button(
        "Submit Answer",
        type="primary",
        key=f"submit_quiz_{current_index}",
        use_container_width=True,
    ):

        student_answer = (
            answer[0]
            .strip()
            .upper()
        )

        correct_answer = (
            current_question["answer"]
            .strip()
            .upper()
        )

        if student_answer == correct_answer:

            st.session_state.quiz_score += 1

            st.success(
                "✅ Correct!"
            )

        else:

            st.error(
                "❌ Wrong!"
            )

            st.info(
                f"Correct answer: {correct_answer}"
            )

            st.write(
                "Explanation:",
                current_question["explanation"],
            )

        # ----------------------------------------------------
        # MOVE TO NEXT QUESTION
        # ----------------------------------------------------

        if (
            current_index + 1
            >= len(questions)
        ):

            st.session_state.quiz_finished = True

        else:

            st.session_state.quiz_index += 1

        st.rerun()


# ============================================================
# QUIZ RESULT
# ============================================================

if (
    mode == "Take a Quiz"
    and st.session_state.quiz_data is not None
    and st.session_state.quiz_finished
):

    questions = (
        st.session_state.quiz_data
    )

    total = len(questions)

    score = (
        st.session_state.quiz_score
    )

    if total > 0:

        percentage = (
            score / total
        ) * 100

    else:

        percentage = 0

    st.divider()

    st.markdown(
        "### 🎯 Quiz Result"
    )

    result_col1, result_col2 = st.columns(
        2
    )

    with result_col1:

        st.metric(
            "Score",
            f"{score}/{total}",
        )

    with result_col2:

        st.metric(
            "Percentage",
            f"{percentage:.1f}%",
        )

    if percentage >= 80:

        st.success(
            "Excellent work! 🎉"
        )

    elif percentage >= 60:

        st.info(
            "Good job! Keep practicing. 👍"
        )

    else:

        st.warning(
            "Keep practicing and review the material. 📚"
        )

    # ========================================================
    # NEW QUIZ
    # ========================================================

    if st.button(
        "Start New Quiz",
        use_container_width=True,
    ):

        st.session_state.quiz_data = None

        st.session_state.quiz_index = 0

        st.session_state.quiz_score = 0

        st.session_state.quiz_finished = False

        st.session_state.quiz_topic = None

        st.rerun()