# retrain_model.py
import os
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ====== CONFIG ======
CSV_DATA_PATH = "classification_dataset.csv"
MODEL_JSON_PATH = os.environ.get("MODEL_JSON", "sentiment.json")
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", "sentiment.weights.h5")
TOKENIZER_PATH = os.environ.get("TOKENIZER", "tokenizer.pkl")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 100))
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100

# DB Connection from environment variables
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")

# Label mapping
LABELS = {"benign": 0, "suspicious": 1, "critical": 2}

# ====== FETCH FROM DB ======
def fetch_db_data():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        query = """
            SELECT text, classification
            FROM classified_messages
            WHERE classification IN ('benign', 'suspicious', 'critical')
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Retrieved {len(df)} messages from DB")
        return df
    except Exception as e:
        print("[ERROR] Failed to fetch from DB:", e)
        return pd.DataFrame(columns=["text", "classification"])

# ====== LOAD CSV DATA ======
def load_csv_data():
    if os.path.exists(CSV_DATA_PATH):
        df = pd.read_csv(CSV_DATA_PATH)
        if "label" in df.columns:
            df = df.rename(columns={"label": "classification"})
        print(f"Loaded {len(df)} samples from CSV")
        return df
    else:
        print("[WARN] CSV dataset not found, starting fresh.")
        return pd.DataFrame(columns=["text", "classification"])

# ====== MERGE DATASETS ======
def merge_datasets(db_df, csv_df):
    merged = pd.concat([csv_df, db_df], ignore_index=True)
    merged.drop_duplicates(subset=["text"], inplace=True)
    merged.dropna(subset=["text", "classification"], inplace=True)
    print(f"Final merged dataset size: {len(merged)}")
    return merged

# ====== PREPROCESS ======
def preprocess_data(df):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"].astype(str))
    sequences = tokenizer.texts_to_sequences(df["text"].astype(str))
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")
    labels = df["classification"].map(LABELS).values
    y = to_categorical(labels, num_classes=3)
    return padded, y, tokenizer

# ====== BUILD MODEL ======
def build_model(input_length):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def mark_db_examples_as_trained(ids):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE classified_messages SET trained = TRUE WHERE id = ANY(%s);",
            (ids,)
        )
    conn.commit()
    conn.close()

# ====== MAIN TRAINING ======
def main():
    # 1. Fetch data
    db_data = fetch_db_data()
    if not db_data.empty and "id" in db_data.columns:
        mark_db_examples_as_trained(list(db_data["id"]))
    csv_data = load_csv_data()
    merged_data = merge_datasets(db_data, csv_data)

    if merged_data.empty:
        print("[ERROR] No data available for training.")
        return

    # 2. Preprocess
    X, y, tokenizer = preprocess_data(merged_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Build and train model
    model = build_model(input_length=MAX_SEQ_LEN)
    print("[INFO] Starting training...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    metrics = {"val_accuracy": float(val_acc)}
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f)
    # 4. Save model & tokenizer
    model_json = model.to_json()
    with open(MODEL_JSON_PATH, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(MODEL_WEIGHTS_PATH)
    with open(TOKENIZER_PATH, "wb") as tok_file:
        pickle.dump(tokenizer, tok_file)

    print("[INFO] Model & tokenizer saved successfully.")

if __name__ == "__main__":
    main()