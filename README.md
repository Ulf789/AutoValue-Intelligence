# AutoValue-Intelligence 🚗

An AI-powered automotive market intelligence and valuation system that combines quantitative machine learning forecasting with qualitative LLM reasoning to eliminate information asymmetry in the premium second-hand car market.

---

## Key Features
- **Data-Driven Valuation Engine:** Trained a Random Forest Regression pipeline via `Scikit-learn` on custom synthetic datasets featuring non-linear depreciation curves and Gaussian noise.
- **Context-Driven Prompting:** Integrated `LangChain (LCEL)` to construct a structured prompt pipeline, dynamically injecting machine learning price predictions to mitigate LLM hallucinations.
- **Domain-Specific Technical Risk Assessment:** Delivers automated mechanical risk analysis tailored to high-performance European assets (e.g., S63 rod bearings, magnetic ride dampers).
- **Interactive UI Dashboard:** A sleek, user-friendly web interface deployed using `Streamlit` for instantaneous deal metric rendering and tactical negotiation strategy generation.

---

## Tech Stack
- **Core Language:** Python
- **AI & LLM Orchestration:** LangChain Expression Language (LCEL), OpenAI API
- **Machine Learning & Data:** Scikit-learn, Pandas, NumPy
- **Interface & Deployment:** Streamlit

---

## System Architecture
1. **Quantitative Layer:** Takes vehicle specifications (Model, Year, Mileage) and runs inference through the trained Random Forest model to output an AI Baseline Price.
2. **Qualitative Reasoning Layer:** Feeds the pricing delta (Asking Price vs. Baseline) along with domain knowledge into a tailored prompt, processed by GPT-4o.
3. **Presentation Layer:** Renders standard deviation metrics, transaction health tags, and structured negotiation reports on the Streamlit frontend.
