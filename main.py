 from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# 1. Load dataset
# ------------------------------
DATA_PATH = os.environ.get("NURTURA_DATA_PATH", "data/sample_dataset.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['search_text'] = (df['symptom_keywords'].fillna('') + ' ' + df['advice_text'].fillna(''))
    return df

df = load_data()
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
vectors = vectorizer.fit_transform(df['search_text'])

# ------------------------------
# 2. Helper: Query function
# ------------------------------
def retrieve_advice(query, trimester=None, top_k=3):
    if not query or query.strip() == "":
        return []

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, vectors)[0]
    top_idxs = sims.argsort()[::-1][:top_k]
    results = []

    for idx in top_idxs:
        score = float(sims[idx])
        row = df.iloc[idx].to_dict()
        row['score'] = score
        results.append(row)

    # Filter by trimester
    if trimester and trimester.lower() != 'any':
        results = [r for r in results if str(r.get('trimester','')).lower() in (trimester.lower(), 'any', '')]

    return results

# ------------------------------
# 3. Flask app
# ------------------------------
app = Flask(__name__)

@app.route('/')
def root():
    return jsonify({"app": "Nurtura AI Backend (Flask)", "status": "ok"})

@app.route('/symptom', methods=['POST'])
def symptom():
    data = request.get_json()
    text = data.get("text", "")
    trimester = data.get("trimester", None)

    results = retrieve_advice(text, trimester)
    if not results:
        return jsonify({
            "reply": "Thanks — I couldn't find an exact match. Please describe your symptom more clearly or contact your doctor.",
            "matches": []
        })

    top = results[0]
    reply = {
        "advice_text": top.get('advice_text'),
        "advice_category": top.get('advice_category'),
        "urgency": top.get('urgency'),
        "match_score": top.get('score')
    }
    return jsonify({"reply": reply, "matches": results})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    text = data.get("text", "")
    trimester = data.get("trimester", None)
    results = retrieve_advice(text, trimester)

    if not results:
        return jsonify({"message": "I couldn’t find an exact match. Please consult your doctor."})

    top = results[0]
    message = f"I found advice: {top.get('advice_text')} (Urgency: {top.get('urgency')})."
    return jsonify({"message": message, "matches": results})

@app.route('/add-entry', methods=['POST'])
def add_entry():
    data = request.get_json()
    new_row = {
        "symptom_keywords": data.get("symptom_keywords", ""),
        "trimester": data.get("trimester", "any"),
        "advice_category": data.get("advice_category", "selfcare"),
        "advice_text": data.get("advice_text", ""),
        "urgency": data.get("urgency", "low")
    }

    df_existing = pd.read_csv(DATA_PATH)
    new_id = int(df_existing['id'].max()) + 1 if 'id' in df_existing.columns else 1
    new_row['id'] = new_id
    df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    df_existing.to_csv(DATA_PATH, index=False)

    global df, vectorizer, vectors
    df = load_data()
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    vectors = vectorizer.fit_transform(df['search_text'])

    return jsonify({"status": "ok", "new_id": new_id})

@app.route('/dataset/download', methods=['GET'])
def download_dataset():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "Dataset not found"}), 404
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = f.read()
    return jsonify({"csv": data})

# ------------------------------
# 4. Run locally
# ------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
