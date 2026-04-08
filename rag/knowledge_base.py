"""
Medical Knowledge Base — Retrieval-Augmented Generation (RAG)
Uses FAISS vector store + sentence-transformers to retrieve
relevant disease information given a query or disease name.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Disease knowledge corpus ───────────────────────────────────────────────────
MEDICAL_KNOWLEDGE: Dict[str, Dict] = {
    "Atelectasis": {
        "overview": (
            "Atelectasis is a complete or partial collapse of a lung or a section of the lung. "
            "It occurs when the small air sacs (alveoli) within the lung become deflated or "
            "filled with fluid."
        ),
        "symptoms": [
            "Difficulty breathing", "Rapid shallow breathing", "Wheezing",
            "Coughing", "Cyanosis (bluish skin discoloration)",
        ],
        "risk_factors": [
            "General anesthesia", "Lung diseases (asthma, COPD)",
            "Prolonged bed rest", "Respiratory infections",
        ],
        "treatment": [
            "Chest physiotherapy and deep breathing exercises",
            "Bronchoscopy to remove blockages",
            "Positive expiratory pressure devices",
            "Incentive spirometry",
            "Treatment of underlying cause",
        ],
        "follow_up": "Pulmonologist consultation recommended within 1-2 weeks.",
    },
    "Cardiomegaly": {
        "overview": (
            "Cardiomegaly (enlarged heart) is not a disease itself but a sign of an underlying "
            "condition. It may be caused by high blood pressure, coronary artery disease, "
            "infections, or abnormal heart rhythms."
        ),
        "symptoms": [
            "Shortness of breath", "Dizziness", "Abnormal heart rhythm (arrhythmia)",
            "Swelling (edema)", "Fatigue",
        ],
        "risk_factors": [
            "High blood pressure", "Family history of heart disease",
            "Coronary artery disease", "Diabetes", "Obesity",
        ],
        "treatment": [
            "Medications: ACE inhibitors, beta-blockers, diuretics",
            "Lifestyle changes: low sodium diet, exercise",
            "Treatment of underlying cause",
            "In severe cases: implantable devices or heart transplant",
        ],
        "follow_up": "Urgent cardiology referral. Echocardiogram and ECG recommended.",
    },
    "Effusion": {
        "overview": (
            "Pleural effusion is an abnormal, excessive collection of fluid in the pleural space "
            "surrounding the lungs. Can be transudative or exudative depending on the cause."
        ),
        "symptoms": [
            "Chest pain (pleuritic)", "Dry non-productive cough",
            "Dyspnea (shortness of breath)", "Orthopnea",
        ],
        "risk_factors": [
            "Heart failure", "Pneumonia", "Malignancy",
            "Pulmonary embolism", "Liver cirrhosis",
        ],
        "treatment": [
            "Thoracentesis (drainage of fluid)",
            "Pleurodesis for recurrent effusions",
            "Treatment of underlying cause",
            "Diuretics for heart failure-related effusions",
        ],
        "follow_up": "Chest CT scan and thoracentesis with fluid analysis. Oncology or pulmonology referral.",
    },
    "Infiltration": {
        "overview": (
            "Pulmonary infiltrates are substances denser than air (fluid, cells, protein) that "
            "fill the lung's alveolar space. Appear as white opacities on chest X-ray. "
            "Often indicate infection, inflammation, or oedema."
        ),
        "symptoms": [
            "Productive cough", "Fever", "Dyspnea",
            "Pleuritic chest pain", "Haemoptysis",
        ],
        "risk_factors": [
            "Bacterial or viral infection", "Immunocompromised state",
            "Aspiration", "Drug toxicity", "Autoimmune disease",
        ],
        "treatment": [
            "Antibiotics for bacterial infection",
            "Antifungal or antiviral agents as appropriate",
            "Corticosteroids for inflammatory infiltrates",
            "Supportive oxygen therapy",
        ],
        "follow_up": "Sputum culture, complete blood count, and repeat chest X-ray in 4-6 weeks.",
    },
    "Mass": {
        "overview": (
            "A pulmonary mass is a lung lesion > 3 cm in diameter. Pulmonary masses require "
            "thorough investigation to rule out malignancy. They can be benign (hamartoma, "
            "abscess) or malignant (primary lung cancer, metastasis)."
        ),
        "symptoms": [
            "Persistent cough", "Haemoptysis", "Unexplained weight loss",
            "Chest pain", "Hoarseness", "Clubbing of fingers",
        ],
        "risk_factors": [
            "Smoking (strongest risk factor)", "Asbestos or radon exposure",
            "Family history of lung cancer", "Previous lung disease",
        ],
        "treatment": [
            "CT-guided biopsy for tissue diagnosis",
            "PET scan for staging",
            "Surgical resection (lobectomy) for operable cases",
            "Chemotherapy, radiotherapy, or targeted therapy",
        ],
        "follow_up": "URGENT: Oncology and thoracic surgery referral. CT chest with contrast immediately.",
    },
    "Nodule": {
        "overview": (
            "A pulmonary nodule is a small rounded growth in the lung ≤ 3 cm. "
            "Most are benign (granulomas from past infections), but some represent "
            "early-stage lung cancer. Management depends on size, shape, and patient risk."
        ),
        "symptoms": [
            "Usually asymptomatic (incidental finding)",
            "Rarely: cough, chest discomfort",
        ],
        "risk_factors": [
            "Age > 40", "Smoking history",
            "Occupational exposures (silica, asbestos)",
            "History of prior malignancy",
        ],
        "treatment": [
            "Low-risk nodules: surveillance CT at 3-6 months",
            "High-risk nodules: PET-CT, bronchoscopy, or biopsy",
            "Resection if malignancy confirmed",
        ],
        "follow_up": "Follow Fleischner Society guidelines for nodule surveillance. Repeat CT in 3 months.",
    },
    "Pneumonia": {
        "overview": (
            "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
            "The air sacs may fill with fluid or pus, causing cough, fever, chills, and "
            "difficulty breathing. Common causative agents: Streptococcus pneumoniae, "
            "Haemophilus influenzae, Mycoplasma, viruses."
        ),
        "symptoms": [
            "High fever and chills", "Productive cough with purulent sputum",
            "Chest pain on breathing", "Dyspnea", "Fatigue",
            "Confusion (especially in elderly)",
        ],
        "risk_factors": [
            "Age extremes (< 2 or > 65)", "Smoking",
            "Immunosuppression", "COPD or asthma",
            "Recent viral upper respiratory infection",
        ],
        "treatment": [
            "Antibiotics (amoxicillin or macrolide for community-acquired)",
            "Hospitalisation if severe (PSI/CURB-65 score)",
            "Supplemental oxygen",
            "IV fluids and supportive care",
            "Antiviral agents for viral pneumonia",
        ],
        "follow_up": "Clinical reassessment in 48-72 hours. Repeat chest X-ray at 6 weeks.",
    },
    "Pneumothorax": {
        "overview": (
            "Pneumothorax is the presence of air in the pleural space, causing lung collapse. "
            "Can be spontaneous (primary: no underlying disease; secondary: underlying lung "
            "disease) or traumatic. A tension pneumothorax is a life-threatening emergency."
        ),
        "symptoms": [
            "Sudden onset sharp pleuritic chest pain",
            "Acute dyspnea", "Decreased breath sounds on affected side",
            "Tachycardia", "Hypotension (tension pneumothorax)",
        ],
        "risk_factors": [
            "Tall thin male (primary spontaneous)",
            "COPD, cystic fibrosis, Marfan syndrome",
            "Mechanical ventilation", "Chest trauma",
        ],
        "treatment": [
            "Small pneumothorax: observation with supplemental O2",
            "Large/symptomatic: needle aspiration or chest tube insertion",
            "Tension pneumothorax: IMMEDIATE needle decompression (emergency)",
            "Recurrent: pleurodesis or VATS",
        ],
        "follow_up": "URGENT evaluation. If tension pneumothorax suspected — immediate intervention.",
    },
    "Consolidation": {
        "overview": (
            "Pulmonary consolidation occurs when the alveolar air is replaced by fluid, "
            "pus, blood, or cells. Produces a solid appearance on chest X-ray. "
            "Most commonly caused by pneumonia."
        ),
        "symptoms": [
            "Cough with purulent or bloody sputum",
            "High fever", "Pleuritic chest pain",
            "Bronchial breath sounds on auscultation",
        ],
        "risk_factors": [
            "Bacterial infection", "Aspiration",
            "Lung cancer (post-obstructive)", "Pulmonary oedema",
        ],
        "treatment": [
            "Broad-spectrum antibiotics pending cultures",
            "Supportive oxygen therapy",
            "Bronchoscopy if obstructive lesion suspected",
        ],
        "follow_up": "Blood cultures, sputum culture, urinary antigen tests. Reassess in 48 hours.",
    },
    "Edema": {
        "overview": (
            "Pulmonary oedema is an abnormal accumulation of fluid in the lung parenchyma, "
            "impairing gas exchange. Usually caused by left-sided heart failure (cardiogenic) "
            "or lung injury (non-cardiogenic/ARDS)."
        ),
        "symptoms": [
            "Severe dyspnea (especially at night — PND)",
            "Pink frothy sputum", "Orthopnea",
            "Crackles (rales) at lung bases",
            "Tachycardia and hypertension",
        ],
        "risk_factors": [
            "Congestive heart failure", "Myocardial infarction",
            "Severe hypertension", "ARDS", "Sepsis",
        ],
        "treatment": [
            "URGENT: Sit patient upright, high-flow oxygen",
            "IV furosemide (loop diuretic)",
            "Nitrates for cardiogenic oedema",
            "Non-invasive ventilation (CPAP/BiPAP)",
            "Treat underlying cause",
        ],
        "follow_up": "URGENT cardiology review. Echocardiogram. BNP/NT-proBNP levels.",
    },
    "Emphysema": {
        "overview": (
            "Emphysema is a form of COPD involving permanent enlargement of air spaces "
            "distal to terminal bronchioles with destruction of alveolar walls. "
            "Results in air trapping and reduced surface area for gas exchange."
        ),
        "symptoms": [
            "Progressive dyspnea on exertion",
            "Barrel chest", "Diminished breath sounds",
            "Pursed-lip breathing", "Chronic cough",
            "Weight loss in advanced disease",
        ],
        "risk_factors": [
            "Smoking (primary cause)", "Alpha-1 antitrypsin deficiency",
            "Air pollution", "Occupational dust exposure",
        ],
        "treatment": [
            "Smoking cessation (most important)",
            "Bronchodilators: SABA, LABA, LAMA",
            "Pulmonary rehabilitation",
            "Long-term oxygen therapy if hypoxaemic",
            "Lung volume reduction surgery in select cases",
        ],
        "follow_up": "Pulmonology referral. Spirometry (FEV1/FVC ratio). Smoking cessation programme.",
    },
    "Fibrosis": {
        "overview": (
            "Pulmonary fibrosis is scarring and thickening of lung tissue, reducing lung "
            "compliance and gas exchange. Idiopathic pulmonary fibrosis (IPF) is the most "
            "common form with unknown cause and poor prognosis."
        ),
        "symptoms": [
            "Progressive exertional dyspnea",
            "Dry hacking cough",
            "Finger clubbing",
            "Fine inspiratory crackles (Velcro crackles)",
            "Fatigue and weight loss",
        ],
        "risk_factors": [
            "Age > 60", "Male sex", "Smoking",
            "Occupational exposures (asbestos, silica)",
            "Certain medications (amiodarone, methotrexate)",
            "Connective tissue diseases",
        ],
        "treatment": [
            "Anti-fibrotic agents: pirfenidone or nintedanib",
            "Supplemental oxygen",
            "Pulmonary rehabilitation",
            "Lung transplant evaluation for eligible patients",
            "Management of comorbidities",
        ],
        "follow_up": "Pulmonology referral. HRCT chest. PFTs (spirometry, DLCO). 6-minute walk test.",
    },
    "Pleural_Thickening": {
        "overview": (
            "Pleural thickening is fibrosis and scarring of the pleural membrane. "
            "Often results from previous pleural disease (empyema, haemothorax) or "
            "asbestos exposure. Can restrict lung expansion and cause dyspnea."
        ),
        "symptoms": [
            "Dyspnea (especially with exertion)",
            "Chest pain or tightness",
            "Reduced chest expansion",
            "Dullness on percussion",
        ],
        "risk_factors": [
            "Asbestos exposure (mesothelioma risk)", "Previous empyema or haemothorax",
            "Tuberculosis", "Prior pleurodesis",
        ],
        "treatment": [
            "Management of underlying cause",
            "Decortication surgery for significant restriction",
            "Pulmonary rehabilitation",
            "Surveillance for mesothelioma if asbestos-related",
        ],
        "follow_up": "Occupational history review. Asbestosis/mesothelioma screening if applicable.",
    },
    "Hernia": {
        "overview": (
            "Hiatal or diaphragmatic hernia visible on chest X-ray represents herniation "
            "of abdominal contents (stomach, bowel) through the diaphragm into the chest. "
            "Can cause respiratory and gastrointestinal symptoms."
        ),
        "symptoms": [
            "Heartburn and acid reflux", "Dysphagia (difficulty swallowing)",
            "Chest pain", "Shortness of breath",
            "Nausea and vomiting",
        ],
        "risk_factors": [
            "Age > 50", "Obesity", "Pregnancy",
            "Heavy lifting", "Chronic coughing or vomiting",
        ],
        "treatment": [
            "Lifestyle modifications: small meals, avoid lying down after eating",
            "Proton pump inhibitors for GERD symptoms",
            "Surgical repair (laparoscopic Nissen fundoplication) for symptomatic cases",
            "Emergency surgery for incarcerated hernia",
        ],
        "follow_up": "Gastroenterology or general surgery referral. Upper GI endoscopy or barium swallow.",
    },
}


# ── MedicalKnowledgeBase ────────────────────────────────────────────────────────
class MedicalKnowledgeBase:
    """
    FAISS-backed retrieval-augmented knowledge base for chest diseases.

    At initialisation:
      1. Encodes all knowledge documents with sentence-transformers.
      2. Indexes them in a FAISS flat L2 index.

    At query time:
      - Embeds the query.
      - Retrieves top-k most similar documents.
      - Returns structured disease information.

    Falls back to keyword matching if sentence-transformers is not installed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ):
        self.disease_names = list(MEDICAL_KNOWLEDGE.keys())
        self._encoder       = None
        self._index         = None
        self._docs: List[str]  = []
        self._doc_meta: List[Dict] = []

        self._build_corpus()

        if use_faiss:
            self._init_faiss(model_name)

    # ── Build corpus ───────────────────────────────────────────────────────────
    def _build_corpus(self) -> None:
        """Convert knowledge dict into retrievable text documents."""
        for disease, info in MEDICAL_KNOWLEDGE.items():
            # One document per knowledge aspect
            for aspect in ("overview", "symptoms", "risk_factors", "treatment"):
                value = info.get(aspect, "")
                if isinstance(value, list):
                    value = "; ".join(value)

                doc = f"{disease} — {aspect.replace('_', ' ').title()}: {value}"
                self._docs.append(doc)
                self._doc_meta.append({
                    "disease": disease,
                    "aspect":  aspect,
                    "content": value,
                })

    def _init_faiss(self, model_name: str) -> None:
        """Embed documents and build FAISS index."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            print(f"[RAG] Loading encoder: {model_name}")
            self._encoder = SentenceTransformer(model_name)

            embeddings = self._encoder.encode(
                self._docs, show_progress_bar=True, convert_to_numpy=True
            )
            embeddings = embeddings.astype(np.float32)

            dim   = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            self._index = index
            print(f"[RAG] FAISS index built: {index.ntotal} documents, dim={dim}")

        except ImportError as e:
            print(f"[RAG] WARNING: {e} — falling back to keyword search.")

    # ── Query interface ────────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a free-text query.

        Returns:
            list of {"disease", "aspect", "content", "score"} dicts
        """
        if self._index is not None and self._encoder is not None:
            return self._faiss_query(query_text, top_k)
        return self._keyword_query(query_text, top_k)

    def _faiss_query(self, query_text: str, top_k: int) -> List[Dict]:
        q_emb   = self._encoder.encode([query_text], convert_to_numpy=True).astype(np.float32)
        dists, indices = self._index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(dists[0], indices[0]):
            if idx < len(self._doc_meta):
                meta   = self._doc_meta[idx].copy()
                meta["score"] = float(1 / (1 + dist))   # similarity ∈ (0, 1]
                results.append(meta)
        return results

    def _keyword_query(self, query_text: str, top_k: int) -> List[Dict]:
        query_lower = query_text.lower()
        scored: List[Tuple[float, Dict]] = []
        for doc, meta in zip(self._docs, self._doc_meta):
            score = sum(1 for word in query_lower.split() if word in doc.lower())
            if score > 0:
                m         = meta.copy()
                m["score"] = score
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    # ── Structured disease lookup ──────────────────────────────────────────────
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        Return the full structured info dict for a disease.
        Case-insensitive, partial match supported.
        """
        for name, info in MEDICAL_KNOWLEDGE.items():
            if disease_name.lower().replace(" ", "_") in name.lower():
                return {"disease": name, **info}
        return None

    def get_treatment_plan(self, disease_name: str) -> List[str]:
        """Return treatment recommendations for a specific disease."""
        info = self.get_disease_info(disease_name)
        if info:
            return info.get("treatment", [])
        return ["No treatment information found. Consult specialist."]

    def get_follow_up(self, disease_name: str) -> str:
        """Return follow-up recommendation for a specific disease."""
        info = MEDICAL_KNOWLEDGE.get(disease_name, {})
        return info.get("follow_up", "Follow up with treating physician.")

    def get_all_diseases(self) -> List[str]:
        return self.disease_names

    def retrieve_for_diagnoses(
        self,
        predicted_diseases: List[Tuple[str, float]],
        top_k_per_disease: int = 2,
    ) -> Dict[str, Dict]:
        """
        Given a list of (disease_name, probability) pairs, retrieve
        structured information for each disease.

        Returns:
            dict mapping disease_name → full info with probability
        """
        result = {}
        for disease, prob in predicted_diseases:
            info = self.get_disease_info(disease) or {}
            info["probability"] = round(prob, 4)
            info["follow_up"]   = self.get_follow_up(disease)
            result[disease]     = info
        return result
