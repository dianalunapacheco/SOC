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
from log_utils import setup_logger
warnings.filterwarnings('ignore')


logger = setup_logger('./logs/sentiment_analysis.log')


# emotion labeling
def auto_label_emotions_contextaware(tweets_df, sample_size=None):
    """
    Context-aware emotion labeling for Persian protest tweets using
    keyword heuristics and multilingual sentiment analysis.
    """
    logger.info("\n" + "="*70)
    logger.info(" CONTEXT-AWARE EMOTION LABELING FOR PROTEST TWEETS")
    logger.info("="*70)
    logger.info("Loading sentiment analyzer...")
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

    logger.info(f"Labeling {len(tweets_sample)} tweets with context-aware approach...")

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
            logger.info(f"Processed {i}/{len(tweets_sample)} tweets...")

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
            logger.error(f"Error processing batch {i}: {e}")
            for text in texts:
                labeled_tweets.append({
                    'text': text, 'emotion': 'anger', 'confidence': 0.5
                })
            continue

    labeled_df = pd.DataFrame(labeled_tweets)
    logger.info("\nEmotion distribution (context-aware):")
    logger.info(labeled_df['emotion'].value_counts())
    return labeled_df


def create_training_data_automatically(tweets_df, sample_size=5000):
    """
    Create training data automatically using the context-aware
    labeling heuristic.
    """
    logger.info("\n" + "="*70)
    logger.info(" AUTOMATIC TRAINING DATA GENERATION")
    logger.info("="*70)
    logger.info(f"Sample size: {sample_size}")

    labeled_df = auto_label_emotions_contextaware(tweets_df, sample_size)

    min_samples = labeled_df['emotion'].value_counts().min()
    if min_samples < 100:
        logger.info(f"\nWarning: Some emotions have very few samples (min={min_samples})")

    return labeled_df


def train_emotion_classifier(labeled_tweets_path,
                             output_dir='./models/parsbert_emotion_model'):
    """
    Train ParsBERT for emotion classification on auto-labeled data.
    """
    logger.info("\n" + "="*60)
    logger.info("TRAINING EMOTION CLASSIFIER")
    logger.info("="*60)

    labeled_df = pd.read_csv(labeled_tweets_path)
    labeled_df['label'] = labeled_df['emotion'].map(Config.LABEL2ID)
    labeled_df = labeled_df.dropna(subset=['label', 'text'])

    logger.info(f"Total labeled samples: {len(labeled_df)}")
    logger.info("\nEmotion distribution:")
    logger.info(labeled_df['emotion'].value_counts())

    min_class_count = labeled_df['label'].value_counts().min()

    if min_class_count < 5:
        logger.info(f"\n[!] WARNING: Smallest class has only {min_class_count} samples.")
        logger.info("    Disabling stratification to prevent crash.")
        stratify_strategy = None
        stratify_val = None
    else:
        stratify_strategy = labeled_df['label'].values
        stratify_val = 'infer'

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        labeled_df['text'].values, labeled_df['label'].values, test_size=0.3,
        random_state=42, stratify=stratify_strategy)

    if stratify_strategy is not None:
        temp_counts = pd.Series(temp_labels).value_counts()
        if temp_counts.min() < 2:
            stratify_strategy = None
        else:
            stratify_val = temp_labels

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42,
        stratify=stratify_val)

    logger.info(f"\nTrain: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

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
        eval_strategy="epoch", save_strategy="epoch",
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

    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)

    test_predictions = trainer.predict(HFDataset.from_dict({
        'text': test_texts, 'label': test_labels
    }).map(tokenize_function, batched=True))

    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    logger.info(classification_report(
        test_labels, y_pred,
        labels=list(range(len(Config.LABELS))),
        target_names=Config.LABELS,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.LABELS, yticklabels=Config.LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    logger.info(f"\nConfusion matrix saved as {os.path.join(output_dir, 'confusion_matrix.png')}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"\nModel saved to {output_dir}")

    return trainer


def predict_emotions(tweets_df, model_path='./models/parsbert_emotion_model'):
    """Predict emotions for tweets using trained ParsBERT model"""
    logger.info("\n" + "="*60)
    logger.info("PREDICTING EMOTIONS ON FULL DATASET")
    logger.info("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # prepare the data
    texts = tweets_df['text'].tolist()
    logger.info(f"Processing {len(texts)} tweets...")

    predictions = []
    probabilities = []
    BATCH_SIZE = 64

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_texts = [str(t) for t in batch_texts]

            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding=True
            ).to(device)

            # inference
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # store results
            predictions.extend([Config.ID2LABEL[p.item()] for p in preds])
            probabilities.extend(probs.cpu().numpy().tolist())

            # log progress
            if (i // BATCH_SIZE) % 500 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} tweets...({i/len(texts)*100:.1f}%)")

    tweets_df['predicted_emotion'] = predictions

    # unpack probabilities
    prob_array = np.array(probabilities)
    for idx, emotion in enumerate(Config.LABELS):
        tweets_df[f'prob_{emotion}'] = prob_array[:, idx]

    logger.info("\nEmotion distribution in predictions:")
    logger.info(tweets_df['predicted_emotion'].value_counts())

    return tweets_df


# main pipeline
def run_emotion_analysis(test_mode: bool = True, subset_size: int = 10) -> None:
    """Run the complete emotion analysis pipeline.

    :param test_mode: for testing only on a small subset (debugging).
    :param subset_size: number of samples to use in test mode.
    """
    logger.info("\n" + "="*70)
    mode_str = "TEST MODE" if test_mode else "FULL_RUN"
    logger.info(f" MAHSA AMINI MOVEMENT: EMOTION CLASSIFICATION (NLP) - {mode_str} ")
    logger.info("="*70)

    # separate files for testing and real run
    if test_mode:
        data_dir = '../data_test'
        model_dir = './models_test'
        output_dir = './outputs_test'
        cleaned_file = os.path.join(data_dir, 'clean_data_test.csv')
        labeled_file = os.path.join(data_dir, 'labeled_tweets_test.csv')
        emotion_file = os.path.join(data_dir, 'tweets_with_emotions_test.csv')
    else:
        data_dir = '../data'
        model_dir = './models'
        output_dir = './outputs'
        cleaned_file = os.path.join(data_dir, 'clean_data_a.csv')
        labeled_file = os.path.join(data_dir, 'labeled_tweets_auto.csv')
        emotion_file = os.path.join(data_dir, 'tweets_with_emotions.csv')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # data prep
    logger.info("\n[STAGE 1] DATA PREPROCESSING (Dataset A)")
    if os.path.exists(cleaned_file):
        tweets_a = pd.read_csv(cleaned_file)
        logger.info(f"Loaded {len(tweets_a)} cleaned tweets")
    else:
        try:
            tweets_a = obtain_file_a()
            if test_mode:
                logger.info(f"Selecting a subset of {subset_size} samples for testing")
                # slice the data before cleaning and saving
                tweets_a = tweets_a.head(subset_size)

            tweets_a = clean_data_a(tweets_a)
            tweets_a.to_csv(cleaned_file, index=False)
            logger.info(f"Saved cleaned data to '{cleaned_file}'")
        except Exception as e:
            logger.info(f"Error in STAGE 1: {e}")
            return

    # emotion labeling
    if not os.path.exists(labeled_file):
        # in test mode, use subset_size data
        # in full run, use Config.TRAIN_SAMPLES
        sample_n = len(tweets_a) if test_mode else Config.TRAIN_SAMPLES

        logger.info(f"\n[STAGE 2] Generating labels for {sample_n} tweets...")
        labeled_df = create_training_data_automatically(tweets_a, sample_size=sample_n)
        labeled_df.to_csv(labeled_file, index=False)
    else:
        logger.info(f"\n[STAGE 2] Loading existing labeled data from '{labeled_file}'")

    # train parsbert
    logger.info("\n[STAGE 3] TRAINING PARSBERT EMOTION CLASSIFIER")
    epochs = 1 if test_mode else Config.NUM_EPOCHS

    if not os.path.exists(os.path.join(model_dir, 'config.json')):
        if test_mode:
            Config.NUM_EPOCHS = epochs
            Config.TRAIN_SAMPLES = subset_size
        # pass the specific labeled_file path to the trainer
        train_emotion_classifier(labeled_file, output_dir=model_dir)
    else:
        logger.info("\n[STAGE 3] MODEL ALREADY TRAINED - Skipping training")

    # predict
    logger.info("\n[STAGE 4] PREDICTING EMOTIONS")
    if not os.path.exists(emotion_file):
        tweets_predict = predict_emotions(tweets_a, model_path=model_dir)
        tweets_predict.to_csv(emotion_file, index=False)
        logger.info(f"Saved predictions to '{emotion_file}'")
    else:
        logger.info(f"Loading existing predictions from '{emotion_file}'")

    logger.info("\n" + "="*70)
    logger.info(" EMOTION ANALYSIS COMPLETE ")
    logger.info("="*70)
    logger.info(f"Final output: {emotion_file}")
    logger.info("Run network_analysis.py next.")


if __name__ == "__main__":
    run_emotion_analysis(test_mode=False, subset_size=50)
