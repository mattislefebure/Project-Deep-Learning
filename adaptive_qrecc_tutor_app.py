import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Small knowledge base
# -----------------------------

KNOWLEDGE_BASE = [
    {
        "title": "Neural Networks",
        "text": "A neural network is a machine learning model inspired by the human brain. It learns patterns from data using layers of connected nodes called neurons."
    },
    {
        "title": "Gradient Descent",
        "text": "Gradient descent is an optimization algorithm used to minimize a loss function by updating model parameters in the direction of the negative gradient."
    },
    {
        "title": "Question Rewriting",
        "text": "Question rewriting transforms an unclear conversational question into a complete standalone question using the dialogue history."
    },
    {
        "title": "Conversational Context",
        "text": "Conversational context refers to previous dialogue turns that help understand follow-up questions, references, and ambiguous expressions."
    },
    {
        "title": "Retrieval-Augmented Generation",
        "text": "Retrieval-Augmented Generation combines information retrieval with text generation to produce more grounded and relevant responses."
    },
    {
        "title": "Adaptive Tutoring",
        "text": "Adaptive tutoring adjusts explanations based on the learner's level, confusion, and previous interactions."
    }
]

CONFUSION_KEYWORDS = [
    "i don't understand",
    "i do not understand",
    "i still don't understand",
    "confused",
    "explain again",
    "what do you mean",
    "can you simplify",
    "simpler",
    "not clear",
    "i'm lost",
    "je ne comprends pas",
    "explique encore",
    "pas clair"
]

PRONOUNS = [
    "he", "she", "it", "they", "this", "that",
    "these", "those", "his", "her", "their"
]

# -----------------------------
# 2. TF-IDF Retriever
# -----------------------------

kb_texts = [item["text"] for item in KNOWLEDGE_BASE]
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kb_texts)


def retrieve_context(query, top_k=2):
    """
    Retrieves the most relevant knowledge base documents using TF-IDF similarity.
    This avoids heavy dependencies like sentence-transformers.
    """
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, kb_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_docs = []
    for idx in top_indices:
        retrieved_docs.append({
            "title": KNOWLEDGE_BASE[idx]["title"],
            "text": KNOWLEDGE_BASE[idx]["text"],
            "score": float(similarities[idx])
        })

    return retrieved_docs

# -----------------------------
# 3. Memory Manager
# -----------------------------


def get_memory_summary(history, max_turns=4):
    """
    Keeps only the last few turns as short-term memory.
    """
    if not history:
        return "No memory yet."

    recent_history = history[-max_turns:]
    summary = []

    for turn in recent_history:
        summary.append(f"Student: {turn['user']}")
        summary.append(f"Tutor: {turn['assistant']}")

    return "\n".join(summary)

# -----------------------------
# 4. Confusion Detection
# -----------------------------


def detect_confusion(user_question, history):
    q = user_question.lower()

    keyword_detected = any(keyword in q for keyword in CONFUSION_KEYWORDS)

    repeated_question = False
    if history:
        previous_questions = [turn["user"].lower() for turn in history[-3:]]
        repeated_question = any(q == previous for previous in previous_questions)

    short_uncertain_question = len(q.split()) <= 4 and "?" in q

    confusion_score = 0
    if keyword_detected:
        confusion_score += 0.6
    if repeated_question:
        confusion_score += 0.3
    if short_uncertain_question:
        confusion_score += 0.2

    confusion_score = min(confusion_score, 1.0)
    is_confused = confusion_score >= 0.5

    return is_confused, confusion_score

# -----------------------------
# 5. Question Rewriting
# -----------------------------


def rewrite_question(user_question, history):
    """
    Simple rule-based question rewriting.
    In the final report, explain that this can be replaced by T5-small trained on QReCC.
    """
    if not history:
        return user_question

    question_words = user_question.lower().split()
    has_pronoun = any(word.strip("?,.!;") in PRONOUNS for word in question_words)

    if not has_pronoun:
        return user_question

    last_user_question = history[-1]["user"]
    last_assistant_answer = history[-1]["assistant"]

    rewritten = (
        f"Previous student question: {last_user_question}. "
        f"Previous tutor answer: {last_assistant_answer}. "
        f"Current question: {user_question}"
    )

    return rewritten

# -----------------------------
# 6. Adaptive Response Generator
# -----------------------------


def generate_response(rewritten_question, retrieved_docs, is_confused):
    context = retrieved_docs[0]["text"] if retrieved_docs else "I do not have enough information."
    title = retrieved_docs[0]["title"] if retrieved_docs else "Unknown topic"

    if is_confused:
        response = f"""
I understand. Let me explain it more simply.

Topic: {title}

Simple explanation:
{context}

In easier words:
The important idea is to understand what problem this concept solves.
Do not try to memorize everything first. Start with the basic intuition.

Example follow-up:
You can ask: "Can you give me an example?" or "Why is it useful?"
"""
    else:
        response = f"""
Topic: {title}

Answer:
{context}

Interpreted question:
{rewritten_question}

You can ask a follow-up question, and I will keep the context.
"""

    return response.strip()

# -----------------------------
# 7. Custom Evaluation Metric
# -----------------------------


def compute_alqs(confusion_score, retrieval_score, history_length):
    """
    Adaptive Learning Quality Score.
    This is a custom research metric for the project.
    """
    coherence_score = min(history_length / 5, 1.0)
    clarity_score = 1.0 - confusion_score
    adaptation_score = confusion_score
    grounding_score = retrieval_score

    alqs = (
        0.25 * coherence_score +
        0.25 * clarity_score +
        0.25 * adaptation_score +
        0.25 * grounding_score
    )

    return round(alqs, 3)

# -----------------------------
# 8. Terminal App
# -----------------------------


def print_debug_info(rewritten_question, confusion_score, retrieval_score, alqs):
    print("\n--- Research Debug View ---")
    print(f"Rewritten question: {rewritten_question}")
    print(f"Confusion score: {confusion_score}")
    print(f"Retrieval score: {round(retrieval_score, 3)}")
    print(f"Adaptive Learning Quality Score: {alqs}")
    print("---------------------------\n")


def main():
    history = []

    print("============================================")
    print(" Adaptive Conversational Tutor using QReCC")
    print("============================================")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.")
    print("Type 'memory' to display conversation memory.")
    print("\nExample questions:")
    print("- What is a neural network?")
    print("- Why is it useful?")
    print("- I don't understand, explain again")
    print("============================================\n")

    while True:
        user_question = input("Student: ").strip()

        if user_question.lower() == "exit":
            print("Tutor: Goodbye!")
            break

        if user_question.lower() == "memory":
            print("\nConversation Memory:")
            print(get_memory_summary(history))
            print()
            continue

        if not user_question:
            print("Tutor: Please ask a question.\n")
            continue

        rewritten_question = rewrite_question(user_question, history)
        is_confused, confusion_score = detect_confusion(user_question, history)
        retrieved_docs = retrieve_context(rewritten_question)
        response = generate_response(rewritten_question, retrieved_docs, is_confused)

        retrieval_score = retrieved_docs[0]["score"] if retrieved_docs else 0
        alqs = compute_alqs(confusion_score, retrieval_score, len(history))

        print("\nTutor:")
        print(response)

        print_debug_info(
            rewritten_question,
            confusion_score,
            retrieval_score,
            alqs
        )

        history.append({
            "user": user_question,
            "assistant": response,
            "rewritten_question": rewritten_question,
            "confusion_score": confusion_score,
            "retrieval_score": retrieval_score,
            "alqs": alqs
        })


if __name__ == "__main__":
    main()
