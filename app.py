import hashlib
import json
import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_engine import RAGEngine


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# Custom styling
# ==================================================

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

        .section-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
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


# ==================================================
# Normal LLM
# ==================================================

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2"
    )


llm = load_llm()


# ==================================================
# JSON LLM for MCQs and Quiz
# ==================================================

@st.cache_resource
def load_quiz_llm():
    return ChatOllama(
        model="llama3.2",
        format="json"
    )


quiz_llm = load_quiz_llm()


# ==================================================
# Q&A prompt
# ==================================================

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
- Do not invent information that is not supported
  by the study material.
- If the answer is not available in the study material,
  say:
  "I couldn't find the answer in the uploaded study material."
"""
)


# ==================================================
# Explain Simply prompt
# ==================================================

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
- Preserve the terminology and wording used in the
  study material as much as possible.
- Use examples only if they appear in the study material.
- Do not create new examples.
- Do not add analogies or outside knowledge.
- Do not add general programming knowledge.
- Do not add conclusions that are not explicitly
  supported by the study material.
- Do not mention related topics unless they are directly
  relevant to the requested topic.
"""
)


# ==================================================
# Summary prompt
# ==================================================

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
"""
)


# ==================================================
# MCQ JSON prompt
# ==================================================

mcq_json_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant creating multiple-choice questions.

Requested topic:
{question}

Study material:
{context}

Create exactly {number_of_questions} questions.

STRICT RULES:
- Use only the study material.
- Questions must be about the requested topic.
- Do not use outside knowledge.
- Each question must have exactly four options.
- Options must be A, B, C, and D.
- There must be exactly one correct answer.
- The correct answer must be supported by the study material.
- The explanation must support the selected answer.
- Do not create unrelated questions.
- If the study material does not contain enough information,
  create fewer questions rather than inventing information.

Return ONLY valid JSON using exactly this structure:

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


# ==================================================
# Quiz JSON prompt
# ==================================================

quiz_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Study Assistant creating an interactive quiz.

Requested topic:
{question}

Study material:
{context}

Create up to {number_of_questions} multiple-choice questions.

STRICT RULES:
- Use ONLY the study material.
- Questions must be about the requested topic.
- Do not use outside knowledge.
- Each question must have exactly four options:
  A, B, C, and D.
- There must be exactly one correct answer.
- The answer must be A, B, C, or D.
- The explanation must support the correct answer.
- Do not create unrelated questions.
- If there is not enough information, create fewer questions.
- Never invent facts.

Return ONLY valid JSON:

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


# ==================================================
# Session state
# ==================================================

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


# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.markdown(
        "## 📚 AI Study Assistant"
    )

    st.caption(
        "Upload your study material and choose a learning mode."
    )

    st.divider()

    # ----------------------------------------------
    # PDF upload
    # ----------------------------------------------

    st.markdown(
        "### 📄 Study Material"
    )

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(
            uploaded_file.name
        )

    st.divider()

    # ----------------------------------------------
    # Study mode
    # ----------------------------------------------

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
        ]
    )

    # ----------------------------------------------
    # Number of questions
    # ----------------------------------------------

    number_of_questions = 5

    if mode in {
        "Generate MCQs",
        "Take a Quiz"
    }:

        number_of_questions = st.slider(
            "Number of questions",
            min_value=1,
            max_value=10,
            value=5
        )

    st.divider()

    st.caption(
        "Powered by LangChain • Chroma • Ollama • Llama 3.2"
    )


# ==================================================
# Main title
# ==================================================

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


# ==================================================
# Require PDF
# ==================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a study PDF from the sidebar to get started."
    )

    st.stop()


# ==================================================
# Create PDF signature
# ==================================================

pdf_bytes = uploaded_file.getvalue()

pdf_signature = hashlib.sha256(
    pdf_bytes
).hexdigest()


# ==================================================
# Save uploaded PDF
# ==================================================

upload_dir = "uploads"

os.makedirs(
    upload_dir,
    exist_ok=True
)

pdf_path = os.path.join(
    upload_dir,
    uploaded_file.name
)


# Save when the PDF content changes
if (
    st.session_state.pdf_signature
    != pdf_signature
):

    with open(
        pdf_path,
        "wb"
    ) as file:

        file.write(
            pdf_bytes
        )


# ==================================================
# Create / load RAG engine
# ==================================================

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

            # Reset quiz when PDF changes
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
        "chunks": engine.collection.count()
    }


# ==================================================
# PDF status
# ==================================================

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


# ==================================================
# Question / topic
# ==================================================

st.markdown(
    "### ✏️ What would you like to study?"
)

question = st.text_input(
    "Enter your question or topic",
    placeholder=(
        "Example: What is a variable?"
    ),
    label_visibility="collapsed"
)


# ==================================================
# Generate button
# ==================================================

button_text = "Start Quiz" if mode == "Take a Quiz" else "Generate"

generate_clicked = st.button(
    button_text,
    type="primary",
    use_container_width=True
)


# ==================================================
# Main generation logic
# ==================================================

if generate_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question or topic."
        )

    else:

        # ------------------------------------------
        # Retrieve relevant material
        # ------------------------------------------

        with st.spinner(
            "Searching the study material..."
        ):

            if mode in {
                "Generate MCQs",
                "Take a Quiz"
            }:

                retrieval = engine.retrieve(
                    question,
                    n_results=1
                )

            else:

                retrieval = engine.retrieve(
                    question,
                    n_results=3
                )


        # ------------------------------------------
        # Check retrieval
        # ------------------------------------------

        if not retrieval["found"]:

            st.warning(
                "I couldn't find the relevant information "
                "in the uploaded study material."
            )

        else:

            context = "\n\n".join(
                retrieval["chunks"]
            )


            # ==================================================
            # TAKE A QUIZ
            # ==================================================

            if mode == "Take a Quiz":

                quiz_prompt_message = (
                    quiz_prompt.invoke(
                        {
                            "context": context,
                            "question": question,
                            "number_of_questions":
                                number_of_questions
                        }
                    )
                )

                with st.spinner(
                    "Creating your quiz..."
                ):

                    try:

                        response = quiz_llm.invoke(
                            quiz_prompt_message
                        )

                        quiz_data = json.loads(
                            response.content
                        )

                    except json.JSONDecodeError:

                        st.error(
                            "The quiz could not be created correctly. "
                            "Please try again."
                        )

                        st.stop()

                    except Exception as error:

                        st.error(
                            f"Could not create quiz: {error}"
                        )

                        st.stop()


                questions = quiz_data.get(
                    "questions",
                    []
                )


                if not questions:

                    st.error(
                        "No quiz questions were generated."
                    )

                    st.stop()


                # Store quiz
                st.session_state.quiz_data = questions
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_finished = False
                st.session_state.quiz_topic = question

                st.rerun()


            # ==================================================
            # GENERATE MCQs
            # ==================================================

            elif mode == "Generate MCQs":

                mcq_prompt_message = (
                    mcq_json_prompt.invoke(
                        {
                            "context": context,
                            "question": question,
                            "number_of_questions":
                                number_of_questions
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

                        mcq_data = json.loads(
                            response.content
                        )

                    except json.JSONDecodeError:

                        st.error(
                            "The MCQs could not be formatted "
                            "correctly. Please try again."
                        )

                        st.stop()

                    except Exception as error:

                        st.error(
                            f"Could not generate MCQs: {error}"
                        )

                        st.stop()


                questions = mcq_data.get(
                    "questions",
                    []
                )


                if not questions:

                    st.error(
                        "No MCQs were generated."
                    )

                    st.stop()


                # ------------------------------------------
                # Validate MCQs
                # ------------------------------------------

                valid_questions = []

                for item in questions:

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    question_text = item.get(
                        "question"
                    )

                    options = item.get(
                        "options"
                    )

                    correct_answer = item.get(
                        "answer"
                    )

                    explanation = item.get(
                        "explanation"
                    )

                    if not question_text:
                        continue

                    if not isinstance(
                        options,
                        dict
                    ):
                        continue

                    required_options = {
                        "A",
                        "B",
                        "C",
                        "D"
                    }

                    if not required_options.issubset(
                        options.keys()
                    ):
                        continue

                    if not all(
                        options.get(letter)
                        for letter in required_options
                    ):
                        continue

                    if correct_answer not in required_options:
                        continue

                    if not explanation:
                        continue

                    valid_questions.append(
                        item
                    )


                if not valid_questions:

                    st.error(
                        "The generated MCQs did not contain "
                        "valid questions."
                    )

                    st.stop()


                # ------------------------------------------
                # Display MCQs
                # ------------------------------------------

                st.markdown(
                    "### 📝 Multiple Choice Questions"
                )

                for index, item in enumerate(
                    valid_questions,
                    start=1
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
                                f"Explanation: "
                                f"{item['explanation']}"
                            )


            # ==================================================
            # NORMAL MODES
            # ==================================================

            else:

                if mode == "Ask a question":

                    prompt = (
                        question_prompt.invoke(
                            {
                                "context": context,
                                "question": question
                            }
                        )
                    )

                elif mode == "Explain simply":

                    prompt = (
                        explain_prompt.invoke(
                            {
                                "context": context,
                                "question": question
                            }
                        )
                    )

                else:

                    prompt = (
                        summary_prompt.invoke(
                            {
                                "context": context,
                                "question": question
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

                        st.error(
                            f"Could not generate the answer: {error}"
                        )

                        st.stop()


                st.markdown(
                    "### 🤖 AI Study Assistant"
                )

                with st.container(
                    border=True
                ):

                    st.write(
                        answer
                    )


            # ==================================================
            # Source context
            # ==================================================

            with st.expander(
                "📖 Show retrieved study material"
            ):

                for index, chunk in enumerate(
                    retrieval["chunks"],
                    start=1
                ):

                    st.markdown(
                        f"**Source chunk {index}**"
                    )

                    st.write(
                        chunk
                    )

                    st.divider()


# ==================================================
# ACTIVE QUIZ
# ==================================================

if (
    mode == "Take a Quiz"
    and st.session_state.quiz_data is not None
    and not st.session_state.quiz_finished
):

    questions = st.session_state.quiz_data

    current_index = (
        st.session_state.quiz_index
    )

    current_question = (
        questions[current_index]
    )

    st.divider()

    st.markdown(
        f"### 📝 Question {current_index + 1} "
        f"of {len(questions)}"
    )

    st.write(
        current_question["question"]
    )

    options = current_question["options"]

    answer = st.radio(
        "Choose your answer:",
        [
            f"A. {options['A']}",
            f"B. {options['B']}",
            f"C. {options['C']}",
            f"D. {options['D']}"
        ],
        key=f"quiz_answer_{current_index}"
    )

    if st.button(
        "Submit Answer",
        type="primary",
        key=f"submit_quiz_{current_index}",
        use_container_width=True
    ):

        student_answer = answer[0]

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
                f"Correct answer: "
                f"{correct_answer}"
            )

            st.write(
                "Explanation:",
                current_question["explanation"]
            )

        if (
            current_index + 1
            >= len(questions)
        ):

            st.session_state.quiz_finished = True

        else:

            st.session_state.quiz_index += 1

        st.rerun()


# ==================================================
# QUIZ RESULT
# ==================================================

if (
    mode == "Take a Quiz"
    and st.session_state.quiz_data is not None
    and st.session_state.quiz_finished
):

    questions = st.session_state.quiz_data

    total = len(questions)

    score = st.session_state.quiz_score

    percentage = (
        score / total
    ) * 100

    st.divider()

    st.markdown(
        "### 🎯 Quiz Result"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Score",
            f"{score}/{total}"
        )

    with result_col2:

        st.metric(
            "Percentage",
            f"{percentage:.1f}%"
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


    if st.button(
        "Start New Quiz",
        use_container_width=True
    ):

        st.session_state.quiz_data = None
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_finished = False
        st.session_state.quiz_topic = None

        st.rerun()