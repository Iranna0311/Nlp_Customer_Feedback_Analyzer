import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

# Import project utilities
from src.utils.data_preprocessing import DataPreprocessor
from src.utils.feature_engineering import FeedbackFeatureEngineer
from src.utils.model_evaluation import evaluate_classification_model
from src.utils.statistical_analysis import analyze_text_lengths, compute_mcnemar_test

try:
    from src.models.logistic_regression import LogisticRegressionModel
except Exception:
    import joblib
    from sklearn.linear_model import LogisticRegression

    class LogisticRegressionModel:
        def __init__(self, C: float = 1.0, max_iter: int = 1000):
            self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

        def train(self, X_train, y_train):
            print("Training Logistic Regression Model...")
            self.model.fit(X_train, y_train)

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

        def save_model(self, filepath: str):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Logistic Regression model saved to: {filepath}")


try:
    from src.models.random_forest import RandomForestModel
except Exception:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    class RandomForestModel:
        def __init__(self, n_estimators: int = 100, max_depth: int = None):
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        def train(self, X_train, y_train):
            print("Training Random Forest Model...")
            self.model.fit(X_train, y_train)

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

        def save_model(self, filepath: str):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Random Forest model saved to: {filepath}")


try:
    from src.models.svm import SVMClassifierModel
except Exception:
    from sklearn.svm import SVC

    class SVMClassifierModel:
        def __init__(self, kernel='linear', C=1.0, random_state=42):
            self.model = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)

        def train(self, X_train, y_train):
            self.model.fit(X_train, y_train)
            return self

        def predict(self, X_test):
            return self.model.predict(X_test)

        def predict_proba(self, X_test):
            return self.model.predict_proba(X_test)

        def score(self, X_test, y_test):
            return self.model.score(X_test, y_test)


try:
    from src.models.neural_networks import NeuralNetworkModel
except Exception:
    from sklearn.neural_network import MLPClassifier

    class NeuralNetworkModel:
        def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=300,
        ):
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                random_state=random_state,
                max_iter=max_iter,
            )

        def train(self, X_train, y_train):
            print("Training Neural Network Model...")
            self.model.fit(X_train, y_train)
            return self

        def predict(self, X_test):
            return self.model.predict(X_test)

        def predict_proba(self, X_test):
            return self.model.predict_proba(X_test)

        def score(self, X_test, y_test):
            return self.model.score(X_test, y_test)

        def save_model(self, filepath: str):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Neural network model saved to: {filepath}")

# Initialize FastAPI App
app = FastAPI(
    title="NLP Customer Feedback Analyzer API", 
    description="A production-ready API for analyzing customer review sentiment.",
    version="1.0.0"
)

ANALYZER_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NLP Customer Feedback Analyzer</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #0f172a;
            --card: #111827;
            --muted: #94a3b8;
            --text: #e5e7eb;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --danger: #fb7185;
            --border: rgba(148, 163, 184, 0.2);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #1e293b 0%, var(--bg) 55%);
            color: var(--text);
            display: grid;
            place-items: center;
            padding: 24px;
        }
        .wrap {
            width: min(960px, 100%);
            display: grid;
            gap: 20px;
        }
        .hero, .panel {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(14px);
        }
        .hero {
            padding: 28px;
        }
        .eyebrow {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 12px;
            margin-bottom: 10px;
        }
        h1 {
            margin: 0 0 10px;
            font-size: clamp(28px, 5vw, 48px);
            line-height: 1.05;
        }
        .sub {
            margin: 0;
            color: var(--muted);
            max-width: 68ch;
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: 1.3fr 0.9fr;
            gap: 20px;
        }
        .panel {
            padding: 22px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
        }
        textarea {
            width: 100%;
            min-height: 180px;
            resize: vertical;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: rgba(2, 6, 23, 0.8);
            color: var(--text);
            padding: 16px;
            font: inherit;
            line-height: 1.6;
            outline: none;
        }
        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }
        .actions {
            display: flex;
            gap: 12px;
            margin-top: 14px;
            flex-wrap: wrap;
        }
        button {
            border: none;
            border-radius: 999px;
            padding: 12px 18px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.15s ease;
        }
        button:hover { transform: translateY(-1px); }
        .primary { background: linear-gradient(135deg, var(--accent), #2563eb); color: white; }
        .secondary { background: rgba(148, 163, 184, 0.14); color: var(--text); border: 1px solid var(--border); }
        .samples {
            display: grid;
            gap: 10px;
            margin-top: 18px;
        }
        .sample {
            text-align: left;
            width: 100%;
            background: rgba(148, 163, 184, 0.1);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .result {
            display: grid;
            gap: 14px;
            align-content: start;
        }
        .badge {
            display: inline-flex;
            width: fit-content;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 700;
        }
        .positive { background: rgba(34, 197, 94, 0.15); color: #86efac; }
        .negative { background: rgba(251, 113, 133, 0.15); color: #fda4af; }
        .meta {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .small {
            font-size: 13px;
            color: var(--muted);
            margin-top: 10px;
        }
        .error {
            color: #fecaca;
            background: rgba(127, 29, 29, 0.4);
            border: 1px solid rgba(248, 113, 113, 0.35);
            padding: 12px 14px;
            border-radius: 12px;
            display: none;
        }
        .loading {
            opacity: 0.7;
            pointer-events: none;
        }
        @media (max-width: 860px) {
            .grid { grid-template-columns: 1fr; }
            .hero, .panel { padding: 18px; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <div class="eyebrow">Customer feedback analyzer</div>
            <h1>Analyze real feedback instantly.</h1>
            <p class="sub">Paste a review, comment, or support message and the model will classify it as positive or negative with a confidence score.</p>
        </section>

        <section class="grid">
            <div class="panel">
                <label for="feedback">Customer feedback</label>
                <textarea id="feedback" placeholder="Example: The delivery was quick and the product quality exceeded my expectations."></textarea>
                <div class="actions">
                    <button class="primary" id="analyzeBtn">Analyze feedback</button>
                    <button class="secondary" id="clearBtn" type="button">Clear</button>
                </div>
                <div class="samples">
                    <button class="sample secondary" data-sample="I love the new interface and the support team was amazing.">Try a positive sample</button>
                    <button class="sample secondary" data-sample="The app crashes every time I try to upload a file and it's very frustrating.">Try a negative sample</button>
                </div>
                <div class="small">This page calls the live `/predict` endpoint used by the API.</div>
            </div>

            <div class="panel result">
                <div id="errorBox" class="error"></div>
                <div id="statusBadge" class="badge positive">Waiting for input</div>
                <div>
                    <div class="small">Confidence</div>
                    <div id="confidence" style="font-size: 34px; font-weight: 800; margin-top: 4px;">—</div>
                </div>
                <div>
                    <div class="small">Cleaned text</div>
                    <div id="cleaned" class="meta">No analysis yet.</div>
                </div>
                <div>
                    <div class="small">Raw feedback</div>
                    <div id="raw" class="meta">No analysis yet.</div>
                </div>
            </div>
        </section>
    </div>

    <script>
        const textarea = document.getElementById('feedback');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const errorBox = document.getElementById('errorBox');
        const statusBadge = document.getElementById('statusBadge');
        const confidence = document.getElementById('confidence');
        const cleaned = document.getElementById('cleaned');
        const raw = document.getElementById('raw');

        function setLoading(isLoading) {
            document.body.classList.toggle('loading', isLoading);
            analyzeBtn.textContent = isLoading ? 'Analyzing...' : 'Analyze feedback';
        }

        function showError(message) {
            errorBox.textContent = message;
            errorBox.style.display = 'block';
        }

        function clearError() {
            errorBox.textContent = '';
            errorBox.style.display = 'none';
        }

        function renderResult(data) {
            const positive = data.sentiment === 'Positive';
            statusBadge.textContent = `${data.sentiment} sentiment`;
            statusBadge.className = `badge ${positive ? 'positive' : 'negative'}`;
            confidence.textContent = `${Math.round((data.confidence_score || 0) * 100)}%`;
            cleaned.textContent = data.cleaned_text || '';
            raw.textContent = data.raw_feedback || '';
        }

        async function analyze() {
            const value = textarea.value.trim();
            clearError();

            if (!value) {
                showError('Please enter some customer feedback first.');
                return;
            }

            setLoading(true);
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: value })
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction request failed.');
                }

                renderResult(data);
            } catch (error) {
                showError(error.message || 'Something went wrong while analyzing the feedback.');
            } finally {
                setLoading(false);
            }
        }

        analyzeBtn.addEventListener('click', analyze);
        clearBtn.addEventListener('click', () => {
            textarea.value = '';
            clearError();
            statusBadge.textContent = 'Waiting for input';
            statusBadge.className = 'badge positive';
            confidence.textContent = '—';
            cleaned.textContent = 'No analysis yet.';
            raw.textContent = 'No analysis yet.';
        });

        document.querySelectorAll('[data-sample]').forEach((button) => {
            button.addEventListener('click', () => {
                textarea.value = button.getAttribute('data-sample');
                clearError();
            });
        });

        textarea.addEventListener('keydown', (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                analyze();
            }
        });
    </script>
</body>
</html>"""

# Define the structure for incoming API requests
class FeedbackRequest(BaseModel):
    text: str

# Global placeholders for loaded production model artifacts
PREPROCESSOR = DataPreprocessor()
VECTORIZER = None
MODEL = None

@app.on_event("startup")
def load_artifacts():
    """Loads trained artifacts into memory automatically when server fires up."""
    global VECTORIZER, MODEL
    vectorizer_path = "data/models/tfidf_vectorizer.pkl"
    model_path = "data/models/logistic_regression.pkl"
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        print("\n[INFO] Loading production models into server memory...")
        # Load Vectorizer
        VECTORIZER = joblib.load(vectorizer_path)
        # Load Raw Sklearn Logistic Regression Model Object
        MODEL = joblib.load(model_path)
        print("[INFO] All artifacts loaded successfully. API is ready for predictions!\n")
    else:
        print("\n[WARNING] Model artifacts not found. Please run the training pipeline first using python -m src.main --train\n")


@app.get("/", response_class=HTMLResponse)
def analyzer_page():
    return HTMLResponse(content=ANALYZER_HTML)

@app.post("/predict")
def predict_sentiment(request: FeedbackRequest):
    """API Endpoint to analyze text sentiment in real-time."""
    if VECTORIZER is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not trained or loaded yet.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input feedback text cannot be empty.")
    
    # 1. Clean the incoming raw text string
    cleaned_text = PREPROCESSOR.clean_text(request.text)
    
    # 2. Vectorize the cleaned text string
    vectorized_text = VECTORIZER.transform([cleaned_text])
    
    # 3. Generate Prediction
    prediction = int(MODEL.predict(vectorized_text)[0])
    probabilities = MODEL.predict_proba(vectorized_text)[0]
    confidence = float(probabilities[prediction])
    
    sentiment_label = "Positive" if prediction == 1 else "Negative"
    
    return {
        "raw_feedback": request.text,
        "cleaned_text": cleaned_text,
        "sentiment": sentiment_label,
        "confidence_score": round(confidence, 4)
    }

# --- Original Training Pipeline Logic ---
def run_training_pipeline():
    print("==============================================")
    print("Starting NLP Customer Feedback Analyzer Pipeline")
    print("==============================================")

    raw_data_path = "data/raw/mock_feedback.csv"
    vectorizer_path = "data/models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Mock data file not found at {raw_data_path}. Please run 'python generate_dataset.py' first.")
        return

    df = pd.read_csv(raw_data_path)
    preprocessor = DataPreprocessor()
    processed_df = df.copy()
    processed_df['cleaned_text'] = processed_df['feedback'].astype(str).apply(preprocessor.clean_text)
    processed_df['label'] = processed_df['sentiment'].astype(str).str.lower().map({'positive': 1, 'negative': 0})
    processed_df = processed_df.dropna(subset=['cleaned_text', 'label'])
    processed_df['label'] = processed_df['label'].astype(int)
    
    analyze_text_lengths(processed_df, text_column='cleaned_text', label_column='label')

    engineer = FeedbackFeatureEngineer(max_features=1000)
    X = engineer.fit_transform(processed_df, text_column='cleaned_text')
    y = processed_df['label'].values
    engineer.save_vectorizer(vectorizer_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr_wrapper = LogisticRegressionModel()
    lr_wrapper.train(X_train, y_train)
    lr_preds = lr_wrapper.predict(X_test)
    
    print("\n--- Logistic Regression Evaluation ---")
    evaluate_classification_model(y_test, lr_preds)
    lr_wrapper.save_model("data/models/logistic_regression.pkl")
    
    rf_wrapper = RandomForestModel(n_estimators=10)
    rf_wrapper.train(X_train, y_train)
    rf_preds = rf_wrapper.predict(X_test)
    
    print("\n--- Random Forest Evaluation ---")
    evaluate_classification_model(y_test, rf_preds)
    rf_wrapper.save_model("data/models/random_forest.pkl")

    nn_wrapper = NeuralNetworkModel(hidden_layer_sizes=(50,), max_iter=500)
    nn_wrapper.train(X_train, y_train)
    nn_preds = nn_wrapper.predict(X_test)

    print("\n--- Neural Network Evaluation ---")
    evaluate_classification_model(y_test, nn_preds)
    nn_wrapper.save_model("data/models/neural_network.pkl")

    print("\n--- Running Significance Testing ---")
    compute_mcnemar_test(y_test, lr_preds, rf_preds)

if __name__ == "__main__":
    import sys
    # If explicitly running training flag, run training loop, otherwise guide user to start application server
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        run_training_pipeline()
    else:
        print("To retrain models, run: python -m src.main --train")
        print("To start the API web server, run: uvicorn src.main:app --reload")