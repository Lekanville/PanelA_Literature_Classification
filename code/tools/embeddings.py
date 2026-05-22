import re
import pandas as pd
from sentence_transformers import SentenceTransformer

def clean_academic_text(text):
    """Clean academic text, including legal boilerplate, noise, and US/UK spelling normalization."""
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\xa0', ' ')

    # 1. Standardize US/UK English variations to protect topic cohesion
    # This ensures 'randomized' and 'randomised' merge into a single semantic token
    spelling_map = {
        r'\brandomized\b': 'randomised',
        r'\borganization\b': 'organisation',
        r'\bbehavior\b': 'behaviour',
        r'\bcenter\b': 'centre',
        r'\bmodeling\b': 'modelling'
    }
    for us_pattern, uk_word in spelling_map.items():
        text = re.sub(us_pattern, uk_word, text, flags=re.IGNORECASE)
    
    # 2. Clean specific boilerplate phrases (No text loss following them)
    boilerplate_patterns = [
        r'©', r'\bcopyright\b', r'\ball rights reserved\b', 
        r'\bjohn wiley\b', r'\bspringer nature\b', r'\bwiley sons\b',
        r'\bamerican psychological association\b', r'\bamerican psychological\b',
        r'\bpsychological association\b', r'\babstract available\b',
        r'\belsevier\b', r'\bpublisher\b', r'\bmacmillan publishers\b',
        r'\b\d{4}\s+author\b', r'\bamerican society\b', r'\bfindings suggest\b',
        r'\bnational academy\b', 
        
        r'\b\d{4}\s+[A-Z][a-z]+',                      # Catches "2019 Melatti", "2020 The"
        r'\b\d{4}\s+the\s+cochrane\s+collaboration\b', # Catches Cochrane header explicitly
        r'\b\d{4}\s+author\w*'                         # Keeps your standard fallback safety net
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 3. Remove specific statistical noise
    noise = [
        r'\b95%?\s?ci\b', r'\b95%?\s?confidence interval\b', 
        r'\bet al\b', r'\bp\s?[<=]\s?0?\.05\b',
        r'\binf\b'
    ]
    for pattern in noise:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text).strip()

    # x = re.search("2019.*author.*", text)
    # if x:
    #     print(text)

    return text

def compute_embeddings(df, model_name):
    """
    Compute SBERT embeddings for a list of texts.
    
    Args:
        texts (list of str): List of input texts to embed.
        model_name (str): The name of the SBERT model to use.
    Returns:
        np.ndarray: Array of shape (n_texts, embedding_dim) containing the embeddings.
    """

    # APPLY CLEANING BEFORE EMBEDDING
    df["Titl_and_Abs_Clean"] = (df["Title"].fillna('') + " " + df["Abstract"].fillna('')).apply(clean_academic_text)
    docs = df["Titl_and_Abs_Clean"].tolist()

    # Encode using the pool of 4 visible GPUs
    if model_name == "all-mpnet":
        sbert_model_name = 'all-mpnet-base-v2'
    elif model_name == "Bio_clinical_BERT":
        sbert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    elif model_name == "Biomed_BERT":
        sbert_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    elif model_name == "PubMed_BERT":
        sbert_model_name = "neuml/pubmedbert-base-embeddings"

    model = SentenceTransformer(sbert_model_name)
    pool = model.start_multi_process_pool(target_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

    try:
        print("Encoding abstracts across 4 GPUs...")
        corpus_embeddings = model.encode_multi_process(docs, pool, show_progress_bar=True, batch_size=512)
    finally:
        # CRITICAL: This executes even if encoding errors out, 
        # ensuring the 4 background GPU daemons are instantly killed.
        print("Shutting down GPU Pool cleanly...")
        model.stop_multi_process_pool(pool)

    # When encoding, SBERT will now use the GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SentenceTransformer('all-mpnet-base-v2', device=device)
    # corpus_embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)

    embedding_df = pd.DataFrame(
        corpus_embeddings, 
        columns=[f"corpus_embeddings_{i}" for i in range(corpus_embeddings.shape[1])],
        index=df.index  # Crucial: keep the indices aligned
    )

    # 2. Concatenate the original DF with the new embedding DF
    # axis=1 means "add columns next to each other"
    df = pd.concat([df, embedding_df], axis=1)

    return df




