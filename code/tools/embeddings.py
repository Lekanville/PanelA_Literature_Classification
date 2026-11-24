import os
# Legacy path also required: import re
import pandas as pd
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

# Verify the token is active
if os.getenv("HF_TOKEN"):
    print("HF_TOKEN successfully loaded from security environment.")
else:
    print("Warning: HF_TOKEN not found in environment!")

# Legacy preprocessing path retained for reference. It applied hand-picked
# spelling, boilerplate, and statistical substitutions before PLM encoding.
# def clean_academic_text(text):
#     if not isinstance(text, str):
#         return ""
#
#     text = text.replace('\xa0', ' ')
#     spelling_map = {
#         r'\brandomized\b': 'randomised',
#         r'\borganization\b': 'organisation',
#         r'\bbehavior\b': 'behaviour',
#         r'\bcenter\b': 'centre',
#         r'\bmodeling\b': 'modelling'
#     }
#     for us_pattern, uk_word in spelling_map.items():
#         text = re.sub(us_pattern, uk_word, text, flags=re.IGNORECASE)
#
#     boilerplate_patterns = [
#         r'©', r'\bcopyright\b', r'\ball rights reserved\b',
#         r'\bjohn wiley\b', r'\bspringer nature\b', r'\bwiley sons\b',
#         r'\bamerican psychological association\b', r'\bamerican psychological\b',
#         r'\bpsychological association\b', r'\babstract available\b',
#         r'\belsevier\b', r'\bpublisher\b', r'\bmacmillan publishers\b',
#         r'\b\d{4}\s+author\b', r'\bamerican society\b',
#         r'\bfindings suggest\b', r'\bnational academy\b',
#         r'\b\d{4}\s+[A-Z][a-z]+',
#         r'\b\d{4}\s+the\s+cochrane\s+collaboration\b',
#         r'\b\d{4}\s+author\w*'
#     ]
#     for pattern in boilerplate_patterns:
#         text = re.sub(pattern, '', text, flags=re.IGNORECASE)
#
#     noise = [
#         r'\b95%?\s?ci\b', r'\b95%?\s?confidence interval\b',
#         r'\bet al\b', r'\bp\s?[<=]\s?0?\.05\b', r'\binf\b'
#     ]
#     for pattern in noise:
#         text = re.sub(pattern, '', text, flags=re.IGNORECASE)
#
#     return re.sub(r'\s+', ' ', text).strip()

def prepare_embedding_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace('\xa0', ' ').split())


MODEL_REPOSITORIES = {
    "all-mpnet": "all-mpnet-base-v2",
    "Bio_clinical_BERT": "emilyalsentzer/Bio_ClinicalBERT",
    "Biomed_BERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "PubMed_BERT": "neuml/pubmedbert-base-embeddings",
}


def _encode_frame(df, model, pool):
    df = df.copy()
    df["embedding_text"] = (
        df["Title"].fillna('') + " " + df["Abstract"].fillna('')
    ).apply(prepare_embedding_text)
    docs = df["embedding_text"].tolist()
    print(f"Encoding {len(docs)} abstracts across 4 GPUs...")
    corpus_embeddings = model.encode_multi_process(
        docs, pool, show_progress_bar=True, batch_size=512
    )
    embedding_df = pd.DataFrame(
        corpus_embeddings,
        columns=[f"corpus_embeddings_{index}" for index in range(corpus_embeddings.shape[1])],
        index=df.index,
    )
    return pd.concat([df, embedding_df], axis=1)


def _compute_embeddings_for_frames(frames, model_name):
    try:
        repository = MODEL_REPOSITORIES[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported SBERT model: {model_name}") from error

    model = SentenceTransformer(repository, model_kwargs={"use_safetensors": True})
    pool = model.start_multi_process_pool(target_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    try:
        return tuple(_encode_frame(frame, model, pool) for frame in frames)
    finally:
        print("Shutting down GPU Pool cleanly...")
        model.stop_multi_process_pool(pool)


def compute_embeddings(df, model_name):
    """Compute embeddings for one dataframe using a pretrained sentence model."""
    return _compute_embeddings_for_frames((df,), model_name)[0]


def compute_embeddings_for_splits(train_df, test_df, model_name):
    """Encode train and test data separately using one loaded model and GPU pool."""
    return _compute_embeddings_for_frames((train_df, test_df), model_name)




