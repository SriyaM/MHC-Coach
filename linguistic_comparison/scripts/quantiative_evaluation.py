import pandas as pd
import spacy
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models and tools
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Define action verbs relevant to physical activity or planning
action_verbs = {
    "start", "plan", "try", "schedule", "commit", "walk", "run", "exercise",
    "move", "set", "do", "begin", "continue", "maintain", "build"
}

# Load your CSV file (upload first or change path)
df = pd.read_csv("messages.csv")  # Replace with your actual filename

# Define feature computation function
def compute_features(text):
    doc = nlp(str(text))
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    words = [token.text for token in doc if token.is_alpha]

    char_len = len(text)
    word_len = len(words)
    token_len = len(doc)

    # Sentiment
    sentiment_score = analyzer.polarity_scores(text)["compound"]

    # Action verb count
    action_count = sum(
        1 for token in doc
        if token.pos_ == "VERB" and token.dep_ not in {"aux", "cop"}
    )

    temporal_keywords = {
        "today", "tomorrow", "tonight", "morning", "evening", "daily",
        "each day", "every day", "this week", "next week", "in 10 minutes", "routine"
    }

    temporal_presence = int(
        any(ent.label_ in {"DATE", "TIME"} for ent in doc.ents) or
        any(kw in text.lower() for kw in temporal_keywords)
    )

    # Type-token ratio
    ttr = len(set(tokens)) / len(tokens) if tokens else 0

    # Punctuation features
    exclam_count = text.count("!")

    # Readability
    readability = textstat.flesch_reading_ease(text)

    return [
        char_len, word_len, token_len, sentiment_score, action_count,
        temporal_presence, ttr, exclam_count, readability
    ]

# Apply feature computation to each column (each technique)
results = df.copy()

for col in df.columns:
    print(f"Processing column: {col}")
    features = df[col].apply(compute_features)
    feature_df = pd.DataFrame(features.tolist(), columns=[
        f"{col}_char_len",
        f"{col}_word_len",
        f"{col}_token_len",
        f"{col}_sentiment",
        f"{col}_action_verb_count",
        f"{col}_temporal_ref",
        f"{col}_ttr",
        f"{col}_exclam_count",
        f"{col}_readability"
    ])
    results = pd.concat([results, feature_df], axis=1)

# Save results to CSV
results.to_csv("messages_with_features.csv", index=False)
print("Features extracted and saved to 'messages_with_features.csv'")

# Create summary stats for each generation method
summary_data = []

for col in df.columns:
    summary = {
        "Method": col,
        "Avg Word Len": results[f"{col}_word_len"].mean(),
        "Avg Sentiment": results[f"{col}_sentiment"].mean(),
        "Avg Action Verb Count": results[f"{col}_action_verb_count"].mean(),
        "% Temporal Ref": results[f"{col}_temporal_ref"].mean() * 100,
        "Avg TTR": results[f"{col}_ttr"].mean(),
        "Avg Exclam Count": results[f"{col}_exclam_count"].mean(),
        "Avg Readability": results[f"{col}_readability"].mean(),
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data).round(2)

summary_df.to_csv("summary_features_by_method.csv", index=False)
print("Summary statistics saved to 'summary_features_by_method.csv'")

