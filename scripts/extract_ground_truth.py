"""
Extract ground truth answers from SQuAD v2 and match with queries_500.json.

This script handles two scenarios:
1. If dev-v2.0.json exists: Extract from original SQuAD format
2. Otherwise: Try to download it from SQuAD website

The squad_v2_train_documents.json is already processed and lacks answers.
"""

import json
import urllib.request
from pathlib import Path

from loguru import logger

from src.utils import LoggerConfig, ensure_dir


def download_squad_v2(output_path: str, url: str = None):
    """Download SQuAD v2 dev or train set if not present."""
    if url is None:
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

    logger.info(f"Downloading SQuAD v2 from {url}")
    logger.info("This may take a minute (train set is ~30MB)...")

    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def load_squad_v2_original(path: str):
    """Load SQuAD v2 in original format and build QA map."""
    logger.info(f"Loading SQuAD v2 from {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build mapping: question → answer
    qa_map = {}

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]
                qa_id = qa["id"]

                # Skip impossible questions
                if qa.get("is_impossible", False):
                    continue

                # Get first answer
                answers = qa.get("answers", [])
                if answers:
                    answer_text = answers[0]["text"]

                    # Store by both ID and question text for matching
                    qa_map[qa_id] = {
                        "question": question,
                        "answer": answer_text,
                        "context": context,
                    }

                    # Also store by question for fuzzy matching
                    qa_map[question.lower()] = {
                        "id": qa_id,
                        "answer": answer_text,
                        "context": context,
                    }

    logger.info(f"Loaded {len(qa_map)} QA entries")
    return qa_map


def match_queries_with_answers(queries_path: str, qa_map: dict):
    """Match queries_500.json with ground truth answers."""
    logger.info(f"Loading queries from {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    matched_count = 0
    unmatched_count = 0

    for query in queries:
        query_id = query.get("id")
        question = query.get("query", "")

        matched = False

        # Try matching by ID first
        if query_id and query_id in qa_map:
            query["answer"] = qa_map[query_id]["answer"]
            matched = True
            matched_count += 1

        # Try matching by question text (case-insensitive)
        elif question.lower() in qa_map:
            qa_data = qa_map[question.lower()]
            query["answer"] = qa_data["answer"]
            if "id" in qa_data:
                query["id"] = qa_data["id"]  # Update ID if found
            matched = True
            matched_count += 1

        # Try fuzzy matching (remove punctuation)
        else:
            question_normalized = question.lower().strip("?.,!").strip()

            for _qa_key, qa_value in qa_map.items():
                if isinstance(qa_value, dict) and "question" in qa_value:
                    qa_question = qa_value["question"].lower().strip("?.,!").strip()

                    if question_normalized == qa_question:
                        query["answer"] = qa_value["answer"]
                        matched = True
                        matched_count += 1
                        break

        if not matched:
            query["answer"] = ""  # No match found
            unmatched_count += 1
            logger.debug(f"Unmatched: {question[:60]}...")

    logger.info(f"Matched: {matched_count}, Unmatched: {unmatched_count}")
    return queries


def main():
    """Main extraction script."""
    LoggerConfig.setup(level="INFO")

    # Paths - USE TRAIN SET since queries_500.json was generated from train split
    SQUAD_TRAIN = "data/squad/train-v2.0.json"
    QUERIES_IN = "data/squad/queries_500.json"
    QUERIES_OUT = "data/squad/queries_500_with_answers.json"

    # Ensure data directory exists
    ensure_dir("data/squad")

    # Step 1: Get SQuAD v2 train set
    if not Path(SQUAD_TRAIN).exists():
        logger.warning(f"{SQUAD_TRAIN} not found. Attempting to download...")
        train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        if not download_squad_v2(SQUAD_TRAIN, url=train_url):
            logger.error("Failed to download SQuAD v2 train set. Please download manually from:")
            logger.error("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
            logger.error(f"Save it to: {SQUAD_TRAIN}")
            return

    # Step 2: Load SQuAD and build QA map
    qa_map = load_squad_v2_original(SQUAD_TRAIN)

    # Step 3: Match with queries
    queries_with_answers = match_queries_with_answers(QUERIES_IN, qa_map)

    # Step 4: Save updated queries
    logger.info(f"Saving updated queries to {QUERIES_OUT}")
    with open(QUERIES_OUT, "w", encoding="utf-8") as f:
        json.dump(queries_with_answers, f, indent=2, ensure_ascii=False)

    # Statistics
    with_answers = sum(1 for q in queries_with_answers if q.get("answer"))
    logger.info(f"\n{'='*60}")
    logger.info(f"Total queries: {len(queries_with_answers)}")
    logger.info(f"With answers: {with_answers}")
    logger.info(f"Coverage: {with_answers/len(queries_with_answers)*100:.1f}%")
    logger.info(f"{'='*60}")

    if with_answers < 400:  # Expect ~500 matches from train set
        logger.warning("\n⚠️  WARNING: Low match rate!")
        logger.warning("Expected ~100% coverage with train-v2.0.json")
        logger.warning("Check if questions were modified or filtered")


if __name__ == "__main__":
    main()
