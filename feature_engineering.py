"""
Data Loader & Synthetic Dataset Generator
==========================================
Generates a realistic synthetic dataset of genuine and fake reviews,
or loads an external dataset from CSV.

The synthetic generator creates reviews with distinct linguistic profiles:
- Genuine reviews: specific product details, moderate sentiment, natural language
- Fake reviews: exaggerated language, generic praise/criticism, extreme ratings
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIG, RAW_DATA_DIR, RANDOM_SEED


# ============================================================
# TEMPLATES & VOCABULARY for Synthetic Generation
# ============================================================

# Genuine review components
GENUINE_PRODUCT_DETAILS = [
    "the battery life", "the screen quality", "the build quality",
    "the camera", "the sound system", "the keyboard", "the touchpad",
    "the display brightness", "the processing speed", "the storage capacity",
    "the design", "the weight", "the color options", "the port selection",
    "the software", "the customer service", "the packaging", "the value",
    "the delivery", "the setup process", "the performance", "the durability",
    "the connectivity", "the app integration", "the ergonomics",
]

GENUINE_OPINIONS = [
    "is decent for the price", "works well in most situations",
    "could be better but is acceptable", "met my expectations",
    "exceeded what I expected for this price range",
    "is okay but nothing exceptional", "performs adequately",
    "is solid and reliable", "has improved since the last version",
    "is a good middle ground", "is quite impressive actually",
    "took some getting used to but I like it now",
    "is not perfect but does the job", "is better than competitors",
    "leaves a bit to be desired", "is surprisingly good",
    "is consistent and dependable", "works as advertised",
    "is average compared to similar products", "has room for improvement",
]

GENUINE_CONTEXTS = [
    "I've been using this for about {weeks} weeks now.",
    "After {weeks} weeks of daily use, here are my thoughts.",
    "I bought this {reason} and have used it extensively.",
    "My {person} recommended this and I decided to try it.",
    "I compared several options before settling on this one.",
    "I was hesitant at first but decided to give it a try.",
    "This is my {ordinal} time buying from this brand.",
    "I needed this for {use_case} and it works well.",
    "I've owned similar products before and this one is comparable.",
    "After researching online for days, I went with this product.",
]

GENUINE_CONCLUSIONS = [
    "Overall, I'd give it a {rating} out of 5.",
    "In summary, it's a reasonable purchase.",
    "Would I buy it again? Probably, with some reservations.",
    "Not perfect, but I'm satisfied with my purchase.",
    "It does what it needs to do. No complaints.",
    "I'd recommend it to someone looking for a budget option.",
    "Solid product overall. Minor issues but nothing deal-breaking.",
    "If you need something reliable, this is a good choice.",
    "Happy with my purchase overall.",
    "Good value for money considering the features.",
]

# Fake review components (overly positive / overly negative)
FAKE_POSITIVE_OPENERS = [
    "ABSOLUTELY AMAZING!!!", "This is THE BEST product EVER!!!",
    "WOW WOW WOW!!!! I can't believe how PERFECT this is!!!",
    "FIVE STARS is NOT ENOUGH for this INCREDIBLE product!!!",
    "Best purchase I've EVER made in my ENTIRE life!!!",
    "If I could give 10 stars I would!!!",
    "PERFECT PERFECT PERFECT!!!", "UNBELIEVABLE quality!!!",
    "This changed my life completely!!!", "MUST BUY IMMEDIATELY!!!",
    "Everyone needs this!! Best ever!!", "Greatest product on earth!!!",
]

FAKE_POSITIVE_BODIES = [
    "Everything about it is absolutely perfect and flawless.",
    "I love everything about this product without any exception.",
    "The quality is beyond amazing incredible stunning.",
    "This product has zero flaws and is completely perfect in every way.",
    "I have never seen anything so perfect in my entire life.",
    "Every single feature works perfectly without any issues at all.",
    "The build quality is the best I have ever experienced ever.",
    "This product exceeded my expectations by a thousand percent.",
    "I bought five more for all my friends and family members.",
    "Best quality best price best everything you could ever want.",
    "Perfect product perfect delivery perfect everything all perfect.",
    "Cannot find a single negative thing to say absolutely nothing.",
]

FAKE_NEGATIVE_OPENERS = [
    "WORST PRODUCT EVER!! DO NOT BUY!!!",
    "TERRIBLE!! HORRIBLE!! AVOID AT ALL COSTS!!!",
    "COMPLETE GARBAGE!! TOTAL WASTE OF MONEY!!!",
    "SCAM!! This is a SCAM!! FAKE PRODUCT!!!",
    "ZERO STARS!! This deserves NEGATIVE stars!!!",
    "WORST purchase in my ENTIRE LIFE!!!",
    "DO NOT WASTE YOUR MONEY on this JUNK!!!",
    "DISGUSTING quality!! RETURN IMMEDIATELY!!!",
    "AWFUL AWFUL AWFUL!! Biggest regret ever!!!",
    "NEVER buy this!! NEVER EVER!!!",
]

FAKE_NEGATIVE_BODIES = [
    "Everything about this product is terrible horrible awful.",
    "Nothing works nothing is good nothing is acceptable at all.",
    "Complete and total failure in every possible way imaginable.",
    "The worst quality I have ever seen in my entire life period.",
    "Broke immediately after opening the box completely useless.",
    "Total scam they should be sued for selling this garbage.",
    "Every feature is broken and nothing works as described.",
    "Worst customer service worst product worst experience ever.",
    "I cannot believe this is legal to sell this garbage product.",
    "Save your money buy literally anything else instead seriously.",
]

# Filler phrases for fake reviews (to add bulk)
FAKE_FILLERS = [
    "I'm telling you", "trust me on this", "believe me",
    "you won't regret it", "seriously", "honestly",
    "I promise you", "take my word for it", "no doubt about it",
    "hands down", "without question", "absolutely",
    "100 percent", "guaranteed", "for real",
]


def _generate_genuine_review():
    """Generate a single genuine-looking review with realistic patterns."""
    parts = []

    # Context / opening (60% chance)
    if random.random() < 0.6:
        context = random.choice(GENUINE_CONTEXTS).format(
            weeks=random.randint(1, 24),
            reason=random.choice(["for work", "for personal use", "as a gift",
                                   "for school", "for my home office"]),
            person=random.choice(["friend", "colleague", "family member", "coworker"]),
            ordinal=random.choice(["second", "third", "first"]),
            use_case=random.choice(["work", "gaming", "studying", "everyday use",
                                     "travel", "home entertainment"]),
        )
        parts.append(context)

    # Main body - mention 2-4 specific features
    num_features = random.randint(2, 4)
    features = random.sample(GENUINE_PRODUCT_DETAILS, num_features)
    opinions = random.sample(GENUINE_OPINIONS, num_features)
    for feat, opinion in zip(features, opinions):
        parts.append(f"{feat.capitalize()} {opinion}.")

    # Conclusion (70% chance)
    if random.random() < 0.7:
        conclusion = random.choice(GENUINE_CONCLUSIONS).format(
            rating=random.choice(["3", "3.5", "4", "4", "4.5"])
        )
        parts.append(conclusion)

    review = " ".join(parts)
    # Add some natural typos occasionally (5% chance per review)
    if random.random() < 0.05:
        words = review.split()
        if len(words) > 5:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 3:
                # Swap two adjacent characters
                pos = random.randint(1, len(word) - 2)
                word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
                words[idx] = word
            review = " ".join(words)

    # Ratings: genuine reviews tend toward 3-4 stars
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.25, 0.40, 0.20])

    return review, int(rating)


def _generate_fake_review():
    """Generate a single fake review with telltale deceptive patterns."""
    is_positive = random.random() < 0.65  # 65% fake-positive

    parts = []

    if is_positive:
        parts.append(random.choice(FAKE_POSITIVE_OPENERS))
        # Add 2-3 body sentences
        num_sentences = random.randint(2, 3)
        bodies = random.sample(FAKE_POSITIVE_BODIES, min(num_sentences, len(FAKE_POSITIVE_BODIES)))
        parts.extend(bodies)
        # Add filler phrases
        if random.random() < 0.5:
            parts.append(random.choice(FAKE_FILLERS) + "!!")
        rating = np.random.choice([4, 5], p=[0.15, 0.85])
    else:
        parts.append(random.choice(FAKE_NEGATIVE_OPENERS))
        num_sentences = random.randint(2, 3)
        bodies = random.sample(FAKE_NEGATIVE_BODIES, min(num_sentences, len(FAKE_NEGATIVE_BODIES)))
        parts.extend(bodies)
        if random.random() < 0.5:
            parts.append(random.choice(FAKE_FILLERS) + "!!")
        rating = np.random.choice([1, 2], p=[0.85, 0.15])

    review = " ".join(parts)

    # Fake reviews often repeat exclamation marks
    if random.random() < 0.3:
        review = review.replace(".", "!!")
    if random.random() < 0.2:
        review = review.upper()

    return review, int(rating)


def generate_synthetic_dataset(n_samples=None, fake_ratio=None, save=True):
    """
    Generate a synthetic dataset of genuine and fake reviews.

    Parameters
    ----------
    n_samples : int, optional
        Total number of reviews. Defaults to config value.
    fake_ratio : float, optional
        Proportion of fake reviews. Defaults to config value.
    save : bool
        Whether to save the dataset to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: review_text, rating, label, review_length
    """
    if n_samples is None:
        n_samples = DATASET_CONFIG["n_samples"]
    if fake_ratio is None:
        fake_ratio = DATASET_CONFIG["fake_ratio"]

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    n_fake = int(n_samples * fake_ratio)
    n_genuine = n_samples - n_fake

    print(f"Generating {n_genuine} genuine and {n_fake} fake reviews...")

    reviews = []

    # Generate genuine reviews
    for _ in tqdm(range(n_genuine), desc="Generating genuine reviews"):
        text, rating = _generate_genuine_review()
        reviews.append({
            "review_text": text,
            "rating": rating,
            "label": 0,  # 0 = genuine
        })

    # Generate fake reviews
    for _ in tqdm(range(n_fake), desc="Generating fake reviews"):
        text, rating = _generate_fake_review()
        reviews.append({
            "review_text": text,
            "rating": rating,
            "label": 1,  # 1 = fake
        })

    df = pd.DataFrame(reviews)

    # Add derived fields
    df["review_length"] = df["review_text"].apply(lambda x: len(x.split()))

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save:
        filepath = os.path.join(RAW_DATA_DIR, "reviews_dataset.csv")
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"  Total reviews: {len(df)}")
        print(f"  Genuine: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
        print(f"  Fake: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")

    return df


def load_dataset(filepath=None):
    """
    Load a dataset from CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. If None, loads the generated dataset.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, "reviews_dataset.csv")

    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}. Generating synthetic dataset...")
        return generate_synthetic_dataset()

    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath}: {len(df)} reviews")

    # Ensure required columns exist
    required_cols = ["review_text", "rating", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "review_length" not in df.columns:
        df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print("\nSample genuine review:")
    print(df[df["label"] == 0].iloc[0]["review_text"])
    print("\nSample fake review:")
    print(df[df["label"] == 1].iloc[0]["review_text"])
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRating distribution:\n{df.groupby('label')['rating'].value_counts().unstack(fill_value=0)}")
