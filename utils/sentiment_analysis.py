import pandas as pd
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from preprocessing import (
    Config, obtain_file_a, clean_data_a
)

warnings.filterwarnings('ignore')


# emotion labeling
def auto_label_emotions_contextaware(tweets_df, sample_size=None):
    """
    Context-aware emotion labeling for Persian protest tweets using
    keyword heuristics and multilingual sentiment analysis.
    """
    print("\n" + "="*70)
    print(" CONTEXT-AWARE EMOTION LABELING FOR PROTEST TWEETS")
    print("="*70)
    print("Loading sentiment analyzer...")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1
        )

    if sample_size:
        tweets_sample = tweets_df.sample(n=min(sample_size, len(tweets_df)),
                                         random_state=42)
    else:
        tweets_sample = tweets_df

    print(f"Labeling {len(tweets_sample)} tweets with context-aware "
          f"approach...")

    # persian protest-specific keywords
    emotion_keywords = {
        'anger': ['خشم', 'عصبانی', 'نفرت', 'ظلم', 'رژیم', 'kill'],
        'sadness': ['غم', 'اندوه', 'گریه', 'فقدان', 'loss', 'death'],
        'fear': ['ترس', 'وحشت', 'بازداشت', 'اعدام', 'prison', 'threat'],
        'disgust': ['چندش', 'منزجر', 'شرم', 'shameful', 'disgrace'],
        'joy': ['شادی', 'امید', 'پیروزی', 'آزادی', 'freedom'],
        'surprise': ['تعجب', 'شگفت', 'باورنکردنی', 'shocking', 'wow']
    }

    labeled_tweets = []
    batch_size = 50

    for i in range(0, len(tweets_sample), batch_size):
        batch = tweets_sample.iloc[i:i+batch_size]
        texts = batch['text'].tolist()
        if i % 5000 == 0:
            print(f"Processed {i}/{len(tweets_sample)} tweets...")

        try:
            sentiments = sentiment_analyzer(texts, truncation=True,
                                            max_length=512)

            for text, sentiment in zip(texts, sentiments):
                text_lower = text.lower()
                emotion_scores = {}
                for emotion, keywords in emotion_keywords.items():
                    score = sum(1 for kw in keywords if kw in text_lower)
                    emotion_scores[emotion] = score

                max_keyword_emotion = max(emotion_scores.items(),
                                          key=lambda x: x[1])

                if max_keyword_emotion[1] > 0:
                    emotion = max_keyword_emotion[0]
                    confidence = 0.8
                else:
                    label = sentiment['label'].lower()

                    if 'negative' in label or 'neg' in label:
                        emotion = np.random.choice(
                            ['anger', 'sadness', 'fear'],
                            p=[0.6, 0.25, 0.15])
                    elif 'positive' in label or 'pos' in label:
                        emotion = np.random.choice(['joy', 'surprise'],
                                                   p=[0.7, 0.3])
                    else:
                        emotion = 'neutral'
                    confidence = sentiment['score']

                labeled_tweets.append({
                    'text': text, 'emotion': emotion, 'confidence': confidence
                })

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            for text in texts:
                labeled_tweets.append({
                    'text': text, 'emotion': 'anger', 'confidence': 0.5
                })
            continue

    labeled_df = pd.DataFrame(labeled_tweets)
    print("\nEmotion distribution (context-aware):")
    print(labeled_df['emotion'].value_counts())
    return labeled_df


def create_training_data_automatically(tweets_df, sample_size=5000):
    """
    Create training data automatically using the context-aware
    labeling heuristic.
    """
    print("\n" + "="*70)
    print(" AUTOMATIC TRAINING DATA GENERATION")
    print("="*70)
    print(f"Sample size: {sample_size}")

    labeled_df = auto_label_emotions_contextaware(tweets_df, sample_size)

    min_samples = labeled_df['emotion'].value_counts().min()
    if min_samples < 100:
        print(f"\nWarning: Some emotions have very few samples "
              f"(min={min_samples})")

    labeled_df.to_csv('../data/labeled_tweets_auto.csv', index=False)
    print("\nSaved automatically labeled data to "
          "'../data/labeled_tweets_auto.csv'")

    return labeled_df


def train_emotion_classifier(labeled_tweets_path,
                             output_dir='./models/parsbert_emotion_model'):
    """
    Train ParsBERT for emotion classification on auto-labeled data.
    """
    print("\n" + "="*60)
    print("TRAINING EMOTION CLASSIFIER")
    print("="*60)

    labeled_df = pd.read_csv(labeled_tweets_path)
    labeled_df['label'] = labeled_df['emotion'].map(Config.LABEL2ID)
    labeled_df = labeled_df.dropna(subset=['label', 'text'])

    print(f"Total labeled samples: {len(labeled_df)}")
    print("\nEmotion distribution:")
    print(labeled_df['emotion'].value_counts())

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        labeled_df['text'].values, labeled_df['label'].values, test_size=0.3,
        random_state=42, stratify=labeled_df['label'].values)

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42,
        stratify=temp_labels)

    print(f"\nTrain: {len(train_texts)}, Val: {len(val_texts)}, "
          f"Test: {len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=len(Config.LABELS),
        id2label=Config.ID2LABEL, label2id=Config.LABEL2ID)

    train_dataset = HFDataset.from_dict({'text': train_texts,
                                         'label': train_labels})
    val_dataset = HFDataset.from_dict({'text': val_texts,
                                       'label': val_labels})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length',
                         truncation=True, max_length=Config.MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        warmup_steps=Config.WARMUP_STEPS, weight_decay=Config.WEIGHT_DECAY,
        learning_rate=Config.LEARNING_RATE, logging_steps=50,
        evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        save_total_limit=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    trainer.train()

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    test_predictions = trainer.predict(HFDataset.from_dict({
        'text': test_texts, 'label': test_labels
    }).map(tokenize_function, batched=True))

    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    print(classification_report(test_labels, y_pred,
                                target_names=Config.LABELS, digits=4))

    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.LABELS, yticklabels=Config.LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./outputs/confusion_matrix.png', dpi=300)
    print("\nConfusion matrix saved as './outputs/confusion_matrix.png'")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")

    return trainer


def predict_emotions(tweets_df, model_path='./models/parsbert_emotion_model'):
    """Predict emotions for tweets using trained ParsBERT model"""
    print("\n" + "="*60)
    print("PREDICTING EMOTIONS ON FULL DATASET")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []
    probabilities = []

    print(f"Processing {len(tweets_df)} tweets...")

    for idx, text in enumerate(tweets_df['text'].tolist()):
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(tweets_df)} tweets...")

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=Config.MAX_LENGTH,
                           padding='max_length').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

        predictions.append(Config.ID2LABEL[pred.item()])
        probabilities.append(probs.cpu().numpy()[0])

    tweets_df['predicted_emotion'] = predictions
    for idx, emotion in enumerate(Config.LABELS):
        tweets_df[f'prob_{emotion}'] = [prob[idx] for prob in probabilities]

    print("\nEmotion distribution in predictions:")
    print(tweets_df['predicted_emotion'].value_counts())

    return tweets_df


# main pipeline
def run_emotion_analysis(test_mode=True, subset_size: int = 10) -> None:
    """Run the complete emotion analysis pipeline.

    :param test_mode: for testing only on a small subset (debugging).
    :param subset_size: number of samples to use in test mode.
    """
    print("\n" + "="*70)
    mode_str = "TEST MODE" if test_mode else "FULL_RUN"
    print(f" MAHSA AMINI MOVEMENT: EMOTION CLASSIFICATION (NLP) - {mode_str} ")
    print("="*70)

    # separate files for testing and real run
    data_dir = '../data'
    model_dir = './models' if not test_mode else './models_test'
    output_dir = './outputs' if not test_mode else './outputs_test'

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # data prep
    print("\n[STAGE 1] DATA PREPROCESSING (Dataset A)")
    cleaned_file = '../data/clean_data_a.csv'

    if os.path.exists(cleaned_file):
        tweets_a = pd.read_csv(cleaned_file)
        print(f"Loaded {len(tweets_a)} cleaned tweets")
    else:
        try:
            print("Obtaining the data file...")
            tweets_a = obtain_file_a()
            if test_mode:
                print(f"Selecting a subset of {subset_size} samples for testing")
                tweets_a = tweets_a.head(subset_size)

            print("Cleaning the data...")
            tweets_a = clean_data_a(tweets_a)
            tweets_a.to_csv(cleaned_file, index=False)
            print(f"Saved to '{cleaned_file}'")
        except Exception as e:
            print(f"Error in STAGE 1: {e}")
            return

    # emotion labeling
    labeled_file = '../data/labeled_tweets_auto.csv'
    if not os.path.exists(labeled_file):

        train_sample = 20 if test_mode else Config.TRAIN_SAMPLES

        # Ensure we don't request more samples than we have
        train_sample = min(train_sample, len(tweets_a))

        labeled_df = create_training_data_automatically(
            tweets_a, sample_size=train_sample)
        print(f"\nCreated {len(labeled_df)} labeled training samples")
    else:
        print(f"\n[STAGE 2] Loading existing labeled data from "
              f"'{labeled_file}'")

    # train parsbert
    model_path = './models/parsbert_emotion_model'
    if not os.path.exists(model_path) or not os.path.exists(
            os.path.join(model_path, 'config.json')):
        print("\n[STAGE 3] TRAINING PARSBERT EMOTION CLASSIFIER")
        try:
            # low epochs if the function supports kwargs
            train_emotion_classifier(labeled_file, output_dir=model_path, num_train_epochs=1)
        except TypeError:
            # fallback
            train_emotion_classifier(labeled_file, output_dir=model_path)
    else:
        print("\n[STAGE 3] MODEL ALREADY TRAINED - Skipping training")

    # preict
    print("\n[STAGE 4] PREDICTING EMOTIONS")
    emotion_file = '../data/tweets_with_emotions.csv'

    if not os.path.exists(emotion_file):
        tweets_a = predict_emotions(tweets_a, model_path=model_path)
        tweets_a.to_csv(emotion_file, index=False)
        print(f"Saved predictions to '{emotion_file}'")
    else:
        print(f"Loading existing predictions from '{emotion_file}'")

    print("\n" + "="*70)
    print(" EMOTION ANALYSIS COMPLETE ")
    print("="*70)
    print(f"Final output: {emotion_file}")
    print("Run network_analysis.py next.")


if __name__ == "__main__":
    run_emotion_analysis(test_mode=True, subset_size=10)
