import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_TITLE = "Resume Screening System"
PROJECT_TYPE = "NLP Classification"
DATA_FILE = "text_dataset.csv"
TARGET_COLUMN = "category"
LEVEL = "Intermediate"
RANDOM_STATE = 42

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print(f"Starting {PROJECT_TITLE} Pipeline")
    print("=" * 60)

    # 1. Load dataset
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    text_col = "text" if "text" in df.columns else df.select_dtypes(include=["object"]).columns[0]
    print(f"Text Column: '{text_col}' | Target Column: '{TARGET_COLUMN}'")

    chart_paths = []

    # 2. Text length distribution
    df["text_length"] = df[text_col].astype(str).str.len()
    plt.figure(figsize=(7, 4))
    df["text_length"].hist(bins=25)
    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    p1 = CHART_DIR / "01_text_length_distribution.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths.append(str(p1))
    print(f"Saved: {p1}")

    # 3. Label distribution
    plt.figure(figsize=(7, 4))
    df[TARGET_COLUMN].astype(str).value_counts().plot(kind="bar")
    plt.title(f"Label Distribution: {TARGET_COLUMN}")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p2 = CHART_DIR / "02_label_distribution.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths.append(str(p2))
    print(f"Saved: {p2}")

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str),
        df[TARGET_COLUMN].astype(str),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN].astype(str),
    )

    # 5. Model training
    print("\nTraining TF-IDF + Logistic Regression model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=2500, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    # 6. Evaluation
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "Recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "F1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

    print("\nModel Evaluation Metrics:")
    print("-" * 35)
    for k, v in metrics.items():
        print(f"  {k:20}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    # 7. Confusion Matrix
    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    p3 = CHART_DIR / "03_confusion_matrix.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    chart_paths.append(str(p3))
    print(f"Saved: {p3}")

    # 8. Feature Importance
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    terms = np.array(vectorizer.get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = np.mean(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
        top_idx = np.argsort(weights)[-15:]
        top_terms = pd.DataFrame({"term": terms[top_idx], "importance": weights[top_idx]}).sort_values("importance", ascending=False)
        plt.figure(figsize=(8, 5))
        plt.barh(top_terms["term"][::-1], top_terms["importance"][::-1])
        plt.title("Most Influential TF-IDF Terms")
        plt.tight_layout()
        p4 = CHART_DIR / "04_top_text_features.png"
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close()
        chart_paths.append(str(p4))
        print(f"Saved: {p4}")

    # 9. Save summary
    summary_payload = {
        "project_title": PROJECT_TITLE,
        "level": LEVEL,
        "project_type": PROJECT_TYPE,
        "task_kind": "nlp_classification",
        "dataset_file": DATA_FILE,
        "dataset_shape": list(df.shape),
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
        "charts": chart_paths,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    with open("project_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print("\nUpdated project_summary.json successfully.")

    # 10. Save trained model
    import joblib
    model_path = Path("resume_screening_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}")

    # 11. Sample Inferences Demo
    print("\n" + "=" * 60)
    print("Sample Resume Predictions Demo:")
    print("=" * 60)
    sample_resumes = [
        "Experienced in Python, Pandas, machine learning, statistics, data models, and Scikit-learn.",
        "Cloud architect proficient in AWS, Azure, Docker, Kubernetes, cloud deployment, and DevOps.",
        "Cybersecurity analyst skilled in network security, firewall configuration, threat analysis, and vulnerability management.",
        "Frontend developer specialized in HTML, CSS, JavaScript, React, Node, and backend API integration.",
        "Business Analyst with expertise in Excel, SQL, Power BI dashboards, business requirements, and reporting."
    ]

    for i, text in enumerate(sample_resumes, 1):
        prediction = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        confidence = max(probs) * 100
        print(f"\n[Resume {i}]: \"{text[:75]}...\"")
        print(f" -> Predicted Category: {prediction} (Confidence: {confidence:.2f}%)")

    print("\n" + "=" * 60)
    print("Execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
