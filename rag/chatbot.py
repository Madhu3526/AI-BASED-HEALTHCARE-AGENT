"""
RAG Medical Chatbot
Conversational interface that answers doctor queries about diseases,
treatments, symptoms, and diagnosis findings using the knowledge base.

Supports two modes:
  1. General query — retrieves relevant knowledge chunks and generates
     a structured answer.
  2. Report-grounded query — doctor asks follow-up questions about the
     current patient's diagnosis report (context-aware).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rag.knowledge_base import MedicalKnowledgeBase, MEDICAL_KNOWLEDGE


# ── Chat message ───────────────────────────────────────────────────────────────
@dataclass
class ChatMessage:
    role:    str   # "user" | "assistant"
    content: str


# ── Intent classifier ──────────────────────────────────────────────────────────
_INTENT_PATTERNS: Dict[str, List[str]] = {
    "treatment":   ["treat", "therapy", "management", "medication", "drug", "medicine",
                    "antibiotic", "surgery", "how to manage"],
    "symptoms":    ["symptom", "sign", "present", "complain", "feel", "suffer",
                    "manifestation", "what does it feel"],
    "diagnosis":   ["diagnose", "diagnosis", "detect", "identify", "confirm",
                    "test", "investigation", "workup", "how to find"],
    "risk":        ["risk", "cause", "factor", "predispose", "who gets",
                    "susceptible", "prone"],
    "overview":    ["what is", "explain", "describe", "overview", "about",
                    "definition", "tell me about"],
    "follow_up":   ["follow", "next step", "after", "monitor", "referral",
                    "specialist", "check up"],
    "probability": ["probability", "chance", "confidence", "percent", "likely",
                    "high", "low", "score"],
    "compare":     ["difference", "compare", "versus", "vs", "distinguish",
                    "differentiate"],
    "general":     [],  # fallback
}

def _classify_intent(query: str) -> str:
    q = query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return "general"


def _extract_disease_mentions(query: str) -> List[str]:
    """Return disease names mentioned in the query (case-insensitive)."""
    q = query.lower().replace("_", " ")
    found = []
    for name in MEDICAL_KNOWLEDGE:
        if name.lower().replace("_", " ") in q:
            found.append(name)
    return found


# ── Response builder ───────────────────────────────────────────────────────────
def _format_list(items: List[str], prefix: str = "•") -> str:
    return "\n".join(f"  {prefix} {item}" for item in items)


def _build_answer(
    intent: str,
    diseases: List[str],
    retrieved_docs: List[Dict],
    report_context: Optional[Dict] = None,
) -> str:
    """
    Compose a structured natural-language answer from retrieved knowledge.

    Args:
        intent:         classified user intent
        diseases:       disease names mentioned or detected in query
        retrieved_docs: top-k documents from FAISS / keyword search
        report_context: current patient's diagnosis probabilities (optional)
    """
    if not diseases and not retrieved_docs:
        return (
            "I couldn't find specific information for that query. "
            "Please ask about a specific disease (e.g. 'What are the symptoms of Pneumonia?') "
            "or a clinical topic (e.g. 'treatment for pleural effusion')."
        )

    lines = []

    # ── If specific diseases are mentioned, give structured answer ─────────
    for disease in diseases[:2]:   # limit to 2 diseases per response
        info = MEDICAL_KNOWLEDGE.get(disease)
        if not info:
            continue

        lines.append(f"**{disease}**")

        if intent == "overview":
            lines.append(info.get("overview", "No overview available."))

        elif intent == "symptoms":
            lines.append("Symptoms:")
            lines.append(_format_list(info.get("symptoms", ["Not available"])))

        elif intent == "treatment":
            lines.append("Treatment:")
            lines.append(_format_list(info.get("treatment", ["Consult specialist"])))
            follow = info.get("follow_up", "")
            if follow:
                lines.append(f"\nFollow-up: {follow}")

        elif intent == "risk":
            lines.append("Risk factors:")
            lines.append(_format_list(info.get("risk_factors", ["Not available"])))

        elif intent == "follow_up":
            lines.append(f"Follow-up recommendation: {info.get('follow_up', 'Consult treating physician.')}")

        elif intent == "diagnosis":
            lines.append(info.get("overview", ""))
            lines.append("\nKey investigations:")
            lines.append(_format_list([
                "Chest X-ray (current study)",
                "CT chest for further characterisation",
                "Blood tests: CBC, CRP, cultures",
                "Pulmonary function tests if indicated",
            ]))

        else:
            # General: give a brief overview + key treatment points
            lines.append(info.get("overview", ""))
            treatments = info.get("treatment", [])
            if treatments:
                lines.append("\nKey management steps:")
                lines.append(_format_list(treatments[:3]))

        lines.append("")

    # ── If report context is provided, add patient-specific note ──────────
    if report_context and diseases:
        for disease in diseases[:1]:
            prob = report_context.get(disease)
            if prob is not None:
                lines.append(
                    f"*For this patient: {disease} detected with "
                    f"{prob*100:.1f}% confidence by the AI model.*"
                )

    # ── Fallback: use retrieved doc snippets ──────────────────────────────
    if not diseases and retrieved_docs:
        for doc in retrieved_docs[:2]:
            lines.append(f"**{doc['disease']} — {doc['aspect'].replace('_', ' ').title()}**")
            lines.append(doc["content"])
            lines.append("")

    if not lines:
        return "No relevant information found. Please rephrase your question."

    return "\n".join(lines).strip()


# ── RAG Chatbot ────────────────────────────────────────────────────────────────
class MedicalChatbot:
    """
    Conversational RAG chatbot for clinical queries.

    Maintains per-session chat history.
    Optionally grounded in the current patient's diagnosis report.

    Usage::

        kb      = MedicalKnowledgeBase()
        chatbot = MedicalChatbot(kb)

        # General query
        response = chatbot.chat("What are the symptoms of Pneumonia?")

        # Grounded in a diagnosis report
        chatbot.set_report_context({"Pneumonia": 0.82, "Consolidation": 0.61})
        response = chatbot.chat("How should I treat the top finding?")

        # Get full conversation
        for msg in chatbot.history:
            print(f"[{msg.role}] {msg.content}")
    """

    SYSTEM_GREETING = (
        "Hello, I'm your Medical AI Assistant. I can help you with:\n"
        "  • Disease overviews, symptoms, and risk factors\n"
        "  • Treatment guidelines and management plans\n"
        "  • Follow-up and referral recommendations\n"
        "  • Questions about the current patient's diagnosis\n\n"
        "Ask me anything about the findings in this X-ray report."
    )

    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        max_history: int = 20,
    ):
        self.kb          = knowledge_base
        self.max_history = max_history
        self.history: List[ChatMessage] = []
        self._report_context: Optional[Dict[str, float]] = None
        self._report_diseases: List[str] = []

    def set_report_context(
        self,
        probabilities: Dict[str, float],
        threshold: float = 0.40,
    ) -> None:
        """
        Bind this chatbot session to a specific patient's diagnosis report.
        Enables context-aware answers about detected conditions.

        Args:
            probabilities: {disease: probability} from DiagnosisReport.raw_probabilities
            threshold:     minimum probability to consider a disease "detected"
        """
        self._report_context  = probabilities
        self._report_diseases = [
            d for d, p in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            if p >= threshold
        ]

    def clear_context(self) -> None:
        """Remove report context (switch to general mode)."""
        self._report_context  = None
        self._report_diseases = []

    def reset(self) -> None:
        """Clear conversation history and context."""
        self.history = []
        self.clear_context()

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.

        Adds both messages to history.
        """
        user_message = user_message.strip()
        if not user_message:
            return "Please enter a question."

        self.history.append(ChatMessage(role="user", content=user_message))

        response = self._generate_response(user_message)

        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

        self.history.append(ChatMessage(role="assistant", content=response))
        return response

    def _generate_response(self, query: str) -> str:
        """Core RAG response generation."""
        query_lower = query.lower()

        # ── Special commands ───────────────────────────────────────────────
        if any(w in query_lower for w in ("hello", "hi ", "hey", "start")):
            return self.SYSTEM_GREETING

        if any(w in query_lower for w in ("clear", "reset", "restart", "new chat")):
            self.history = []
            return "Conversation cleared. How can I help you?"

        if any(w in query_lower for w in ("what did you find", "what was detected",
                                           "current findings", "show findings")):
            return self._summarise_findings()

        # ── Intent + disease extraction ────────────────────────────────────
        intent   = _classify_intent(query)
        diseases = _extract_disease_mentions(query)

        # If no disease mentioned but there's a report context, infer from it
        if not diseases and self._report_diseases:
            if any(w in query_lower for w in ("top", "primary", "main", "first",
                                               "the finding", "the disease", "it")):
                diseases = self._report_diseases[:1]
            elif any(w in query_lower for w in ("all", "each", "every", "findings")):
                diseases = self._report_diseases[:3]

        # ── Retrieve from knowledge base ───────────────────────────────────
        retrieved = self.kb.query(query, top_k=4)

        # ── Build answer ───────────────────────────────────────────────────
        answer = _build_answer(intent, diseases, retrieved, self._report_context)

        # Append confidence info if relevant
        if self._report_context and diseases:
            prob_lines = []
            for d in diseases:
                p = self._report_context.get(d)
                if p is not None:
                    prob_lines.append(f"{d}: {p*100:.1f}%")
            if prob_lines and intent not in ("overview",):
                answer += "\n\n*Patient probabilities: " + " | ".join(prob_lines) + "*"

        return answer

    def _summarise_findings(self) -> str:
        """Summarise the current report's detected diseases."""
        if not self._report_context:
            return (
                "No diagnosis report is loaded. Please upload an X-ray image "
                "and run the analysis first."
            )
        if not self._report_diseases:
            return "No significant pathology detected above the confidence threshold in this scan."

        lines = ["**Current findings for this patient:**\n"]
        for disease in self._report_diseases:
            prob = self._report_context.get(disease, 0)
            info = MEDICAL_KNOWLEDGE.get(disease, {})
            follow = info.get("follow_up", "")
            lines.append(f"• **{disease}** — {prob*100:.1f}% confidence")
            if follow:
                lines.append(f"  Follow-up: {follow}")
        lines.append(
            "\nAsk me about any of these conditions for more detail — "
            "e.g. 'How is Pneumonia treated?'"
        )
        return "\n".join(lines)

    def get_suggested_questions(self) -> List[str]:
        """
        Return suggested follow-up questions based on the current report context.
        Shown as quick-action buttons in the dashboard.
        """
        if not self._report_diseases:
            return [
                "What is Pneumonia?",
                "How is Effusion treated?",
                "What are symptoms of Emphysema?",
                "Explain Atelectasis",
            ]

        suggestions = []
        for disease in self._report_diseases[:2]:
            suggestions.extend([
                f"What are the symptoms of {disease}?",
                f"How is {disease} treated?",
                f"What is the follow-up for {disease}?",
                f"What causes {disease}?",
            ])
        return suggestions[:6]
