import os
import re
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from adjustText import adjust_text
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def normalize_topic_text(value):
    if not isinstance(value, str):
        return ""

    text_value = value.replace('\xa0', ' ')
    dash = r"[-\u2010-\u2015]"
    text_value = re.sub(
        rf"\bgenome\s*{dash}?\s*wide\s+associations?\b",
        "genome wide association",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\bgenome\s*{dash}?\s*wide\s+analys(?:is|es)\b",
        "genome wide analysis",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b1\s*,\s*25\s*{dash}?\s*\(?\s*oh\s*\)?\s*2\s*d\s*([23])\b",
        r"1,25 dihydroxyvitamin d\1",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b1\s*,\s*25\s*{dash}?\s*\(?\s*oh\s*\)?\s*2\s*d\b",
        "1,25 dihydroxyvitamin d",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b1\s*,\s*25\s*{dash}?\s*dihydroxy\s*vitamin\s*{dash}?\s*d\s*([23])\b",
        r"1,25 dihydroxyvitamin d\1",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b1\s*,\s*25\s*{dash}?\s*dihydroxy\s*vitamin\s*{dash}?\s*d\b",
        "1,25 dihydroxyvitamin d",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b25\s*{dash}?\s*\(?\s*oh\s*\)?\s*{dash}?\s*d\s*([23])\b",
        r"25 hydroxyvitamin d\1",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b25\s*{dash}?\s*\(?\s*oh\s*\)?\s*{dash}?\s*d\b",
        "25 hydroxyvitamin d",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b25\s*{dash}?\s*hydroxy\s*vitamin\s*{dash}?\s*d\s*([23])\b",
        r"25 hydroxyvitamin d\1",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b25\s*{dash}?\s*hydroxy\s*vitamin\s*{dash}?\s*d\b",
        "25 hydroxyvitamin d",
        text_value,
        flags=re.IGNORECASE,
    )
    text_value = re.sub(
        rf"\b25\s*{dash}?\s*hydroxy\s*cholecalciferol\b",
        "25 hydroxyvitamin d3",
        text_value,
        flags=re.IGNORECASE,
    )
    return ' '.join(text_value.split())


def clean_topic_boilerplate(value):
    if not isinstance(value, str):
        return ""

    patterns = (
        r"\bet\s+al\.?",
        r"\ball\s+rights\s+reserved\b",
        r"\brights\s+reserved\b",
        r"\bspringer\s+nature\b",
        r"\bjohn\s+wiley(?:\s+(?:and|&)\s+sons|\s+sons)?\b",
        r"\bmacmill(?:an|ian)\s+publishers?(?:\s+limited)?\b",
        r"\bpublishers?\s+limited\b",
        r"\bamerican\s+psychological(?:\s+association)?\b",
        r"\bpsychological\s+association\b",
        r"\b95\s*%?\s*(?:confidence\s+intervals?|confidence|ci)\b",
        r"©\s*\d{4},?\s+(?:the\s+)?author(?:s|\(s\))?(?:\s*\(or their employer\(s\)\))?\.?",
        r"©\s*\d{4}\s+american\s+society\s+(?:for\s+nutrition|of\s+human\s+genetics|for\s+clinical\s+investigation)\b\.?",
        r"©\s*\d{4}\s+american\s+(?:heart|chemical)\s+association\b\.?",
        r"©\s*\d{4}\s+american\s+chemical\s+society\b\.?",
        r"\b\d{4}\s+author(?:s|\(s\))?(?!\w)",
        r"\b(?:proceedings\s+of\s+the\s+)?national\s+academy(?:\s+of)?\s+sciences\b",
        r"\bpoorly\s+understood\b",
        r"\bpublished\s+by\s+[^.;]{1,120}",
        r"\bunder\s+exclusive\s+licen[cs]e\s+to\s+[^.;]{1,120}",
        r"\b(?:this\s+is\s+an?\s+)?open[-\s]access\s+article\b",
        r"\b(?:distributed|published)\s+under\s+the\s+terms\s+of\s+(?:the\s+)?creative\s+commons\b.*?\b(?:credited|cited)\b[^.]*\.?",
        r"\b(?:distributed|published)\s+under\s+the\s+terms\s+of\s+(?:the\s+)?creative\s+commons\b.*?\blicen[cs]e\b",
        r"\bcreative\s+commons(?:\s+attribution)?(?:\s+\d(?:\.\d)?(?:\s+international)?)?\s+licen[cs]e\b",
        r"\bcc\s+by(?:[-\s]nc)?\s*\d(?:\.\d)?\s+licen[cs]e\b",
        r"\b(?:distributed|published)\s+under\s+(?:the\s+)?terms\s+of\b",
        r"\bthe\s+use,\s*distribution(?:\s+and|\s+or)\s+reproduction\b[^.]*\.?",
        r"\bprovided\s+(?:that\s+)?(?:the\s+)?original\s+author(?:\(s\)|s)?\b[^.]*\.?",
        r"\b(?:this\s+)?trial\s+is\s+registered\s+with\s+clinicaltrials\.gov,?\s*number\s+[a-z0-9-]+",
        r"\bopen[-\s]access(?:\s+article)?\b",
        r"\bclinicaltrials\s*\.?\s*gov\b",
        r"\buniversity\s+press\b",
    )
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return " ".join(value.split())


def extract_uoa_research_areas(
    df,
    text_col='topic_text',
    uoa_col='Unit_of_assessment_number',
    top_n=10,
    min_prevalence=0.01,
):
    df = df.copy()
    if text_col not in df.columns:
        df[text_col] = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')
    df[text_col] = df[text_col].map(normalize_topic_text).map(clean_topic_boilerplate)

    if not 0 <= min_prevalence <= 1:
        raise ValueError('min_prevalence must be between 0 and 1')

    concept_map = {
        "controlled trial": "randomised controlled trial",
        "randomised controlled": "randomised controlled trial",
        "transcription factors": "transcription factor",
        "stem cells": "stem cell",
        "type diabetes": "type 2 diabetes",
        "people dementia": "people with dementia",
        "cl co transport": "chloride cotransport",
        "cl co transporter": "chloride cotransport",
        "cl cotransport": "chloride cotransport",
        "k cl cotransport": "chloride cotransport",
        "k cl co transport": "chloride cotransport",
        "na cl co transport": "chloride cotransport",
        "genome wide association": "genome-wide association",
        "genome wide analysis": "genome-wide analysis",
    }
    excluded_phrases = {
        "long term",
        "genome wide",
        "wide association",
        "wide analysis",
        "findings suggest",
        "american society",
        "american association",
        "american chemical",
        "american chemical society",
        "chemical society",
        "clinicaltrials gov",
        "open access",
        "open access article",
        "access article",
        "university press",
        "oxford university",
        "author exclusive",
        "exclusive licence",
        "exclusive license",
        "commons attribution",
        "commons attribution license",
        "cc by",
        "cc by 4",
        "creative commons",
        "terms creative commons",
        "creative commons attribution",
        "attribution license",
        "attribution licence",
        "use distribution",
        "use distribution and",
        "use distribution reproduction",
        "distribution reproduction",
        "provided original author",
        "original author",
        "original author source",
        "permits unrestricted",
        "permits unrestricted use",
        "unrestricted use",
        "author source",
        "source credited",
        "25 hydroxyvitamin",
        "hydroxyvitamin d",
        "hydroxyvitamin d2",
        "hydroxyvitamin d3",
    }
    domain_stopwords = {'study', 'results', 'conclusions', 'background', 'methods'}
    stop_words = list(text.ENGLISH_STOP_WORDS.union(domain_stopwords))
    count_vectorizer = CountVectorizer(
        preprocessor=lambda value: value.lower() if isinstance(value, str) else '',
        stop_words=stop_words,
        ngram_range=(2, 3),
    )
    analyze = count_vectorizer.build_analyzer()

    uoa_groups = list(df.groupby(uoa_col, sort=True))
    class_documents = [
        ' '.join(group[text_col])
        for _, group in uoa_groups
    ]
    class_tfidf = TfidfVectorizer(
        preprocessor=lambda value: value.lower() if isinstance(value, str) else '',
        stop_words=stop_words,
        ngram_range=(2, 3),
        max_features=10000,
    )
    tfidf_matrix = class_tfidf.fit_transform(class_documents)
    tfidf_features = class_tfidf.get_feature_names_out()

    uoa_themes = {}
    prevalence_rows = []
    for row_index, (uoa, group) in enumerate(uoa_groups):
        article_count = len(group)
        minimum_count = max(1, math.ceil(article_count * min_prevalence))
        phrase_counts = {}

        for document in group[text_col]:
            article_phrases = {
                concept_map.get(phrase, phrase)
                for phrase in analyze(document)
            }
            article_phrases.difference_update(excluded_phrases)
            article_phrases = {
                phrase for phrase in article_phrases
                if not re.fullmatch(r'\d{4}\s+author(?:s|\(s\))?', phrase)
            }
            for phrase in article_phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        class_scores = {}
        for phrase, score in zip(tfidf_features, tfidf_matrix[row_index].toarray().ravel()):
            canonical_phrase = concept_map.get(phrase, phrase)
            if canonical_phrase not in excluded_phrases:
                class_scores[canonical_phrase] = class_scores.get(canonical_phrase, 0.0) + float(score)

        eligible_phrases = [
            phrase
            for phrase, count in phrase_counts.items()
            if count >= minimum_count and phrase in class_scores
        ]
        ranked_phrases = sorted(
            eligible_phrases,
            key=lambda phrase: (
                -class_scores[phrase],
                -phrase_counts[phrase],
                phrase,
            ),
        )[:top_n]

        uoa_themes[uoa] = ranked_phrases
        for phrase in ranked_phrases:
            count = phrase_counts[phrase]
            prevalence_rows.append(
                {
                    'uoa': uoa,
                    'phrase': phrase,
                    'article_count': count,
                    'training_articles': article_count,
                    'prevalence_percent': round(100 * count / article_count, 2),
                    'minimum_article_count': minimum_count,
                    'ctfidf_score': round(class_scores[phrase], 8),
                }
            )

    return uoa_themes, prevalence_rows

# Legacy class-document TF-IDF ranking retained for comparison.
# def extract_uoa_research_areas(df, text_col='topic_text', uoa_col='Unit_of_assessment_number', top_n=10):
#     # 1. Group all abstracts by UoA to create "Class Documents"
#     # This treats each UoA as one single giant text block
#     uoa_docs = df.groupby(uoa_col)[text_col].apply(lambda x: ' '.join(x)).reset_index()

#     # 2. Define a clean, year-agnostic dynamic cleaner for the vectorizer
#     def dynamic_vectorizer_cleaner(doc):
#         if not isinstance(doc, str):
#             return ""
#         # Lowercase to match vectorizer behavior
#         doc = doc.lower()
#         # Dynamically erase ANY 4-digit year followed by author/authors/authorship
#         # e.g., "2018 author", "2022 authors", "2025 author" become completely blank spaces
#         doc = re.sub(r'\b\d{4}\s+author\w*', ' ', doc)

#         # Collapse synonymous phrases into unified, underscored tokens
#         # Underscores prevent the vectorizer from splitting them up incorrectly later
#         return doc

#     # 3. Initialize TfidfVectorizer with N-gram support
#     # stop_words='english' is vital to remove "the", "and", etc.
#     # ngram_range=(2, 3) forces it to look for phrases/areas specifically
#     domain_stopwords = [
#     'rights reserved', 'et al', '95 ci', '95 confidence', 'confidence interval',
#     'john wiley', 'wiley sons', 'springer nature', 'elsevier', 'publisher',
#     'american psychological', 'psychological association', 'abstract available',
#     'study', 'results', 'conclusions', 'background', 'methods'
#     ]
#     final_stopwords = text.ENGLISH_STOP_WORDS.union(domain_stopwords)

#     # 4. Initialize TfidfVectorizer using the dynamic preprocessor hook
#     vectorizer = TfidfVectorizer(
#         preprocessor=dynamic_vectorizer_cleaner,
#         stop_words=list(final_stopwords), 
#         ngram_range=(2, 3), 
#         max_features=10000
#     )

#     # 5. Fit and transform the Class Documents
#     tfidf_matrix = vectorizer.fit_transform(uoa_docs[text_col])
#     feature_names = vectorizer.get_feature_names_out()

#     # 6. Extract top phrases per UoA
#     uoa_themes = {}
#     for i, row in uoa_docs.iterrows():
#         uoa_name = row[uoa_col]
#         # Get the tf-idf scores for this UoA's "document"
#         scores = tfidf_matrix[i].toarray().flatten()
#         # Sort and get the indices of the top N scores
#         top_indices = scores.argsort()[-top_n:][::-1]
#         # Map indices to the actual phrases
#         top_phrases = [feature_names[idx] for idx in top_indices]
#         uoa_themes[uoa_name] = top_phrases
        
#     return uoa_themes



def plot_uoa_network(themes_dict, output_dir):
    """
    Generates a publication-quality interdisciplinary network plot.
    Fixes node coloration, darkens edge transparency, and handles overlapping labels.
    """

    G = nx.Graph()
    
    # Cohesive, bright palette for the 6 UoA Hubs
    uoa_colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3']
    uoa_to_color = {uoa: color for uoa, color in zip(themes_dict.keys(), uoa_colors)}
    
    # 1. Build the graph dynamically
    for uoa, phrases in themes_dict.items():
        G.add_node(uoa, type='uoa', color=uoa_to_color[uoa])
        for phrase in phrases:
            if not G.has_node(phrase):
                G.add_node(phrase, type='phrase')
            G.add_edge(uoa, phrase)

    phrase_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'phrase']
    uoa_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'uoa']
    
    shared_phrases = [n for n in phrase_nodes if G.degree(n) > 1]
    unique_phrases = [n for n in phrase_nodes if G.degree(n) == 1]

    # Map colors precisely to every node type
    node_colors = {}
    for node in G.nodes():
        if G.nodes[node]['type'] == 'uoa':
            node_colors[node] = uoa_to_color[node]
        elif node in shared_phrases:
            node_colors[node] = '#444444' # Deep gray for shared nodes
        else:
            parent_uoa = list(G.neighbors(node))[0]
            node_colors[node] = uoa_to_color[parent_uoa]

    # 2. Layout Optimization
    # pos = nx.spring_layout(G, k=0.18, iterations=150, seed=42)
    pos = nx.spring_layout(G, k=0.38, iterations=100, seed=42)  # Adjusted k for better separation
    if 1 in pos and 2 in pos:
        pos[1] = pos[1] + np.array([0.32, 0.03])
        pos[2] = pos[2] + np.array([0.10, 0.45])
        pos = nx.spring_layout(
            G,
            pos=pos,
            fixed=uoa_nodes,
            k=0.38,
            iterations=100,
            seed=42,
        )
    
    # OPTIONAL FINE-TUNING: Scaling factor normalization 
    # If the layout still feels slightly wide, we can squash coordinates globally:
    # pos = {node: np.array([coords[0] * 0.85, coords[1]]) for node, coords in pos.items()}
    plt.figure(figsize=(22, 14), dpi=300)
    
    # FIX: Darkened edge settings for visibility (alpha=0.3, width=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#BDBDBD', width=0.8)
    
    # FIX: Explicitly passing the custom color lists to all node types
    nx.draw_networkx_nodes(G, pos, nodelist=unique_phrases, node_size=55,
                           node_color=[node_colors[n] for n in unique_phrases], alpha=0.85)
    nx.draw_networkx_nodes(G, pos, nodelist=shared_phrases, node_size=55,
                           node_color=[node_colors[n] for n in shared_phrases], alpha=0.95)
    hub_nodes = nx.draw_networkx_nodes(G, pos, nodelist=uoa_nodes, node_size=4400, 
                                       node_color=[node_colors[n] for n in uoa_nodes], alpha=0.95)
    
    # 3. Draw Hub Labels
    nx.draw_networkx_labels(G, pos, labels={n: n for n in uoa_nodes}, font_size=22, font_weight='bold')
    
    # 4. Draw Central Shared Labels with layout jitter to fix overlapping text
    # np.random.seed(42)  

    # for node in shared_phrases:
    #     nx_val, ny_val = pos[node]
    #     jitter_x = np.random.uniform(-0.015, 0.015)
    #     jitter_y = np.random.uniform(-0.015, 0.015)
        
    #     plt.text(nx_val + jitter_x, ny_val + jitter_y, s=node,
    #              fontsize=9, fontweight='bold', family='sans-serif',
    #              horizontalalignment='center', verticalalignment='center',
    #              bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.85))
    
    texts = []
    target_x = []
    target_y = []

    # 4. Draw Central Shared Labels & log tracking positions
    for node in shared_phrases:
        nx_val, ny_val = pos[node]
        t = plt.text(nx_val, ny_val, s="cryo-EM" if node == "cryo em" else node,
                     fontsize=15, fontweight='bold', family='sans-serif',
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.85))
        texts.append(t)
        target_x.append(nx_val)
        target_y.append(ny_val)

    # 5. Draw Peripheral Unique Labels
    for node in unique_phrases:
        nx_val, ny_val = pos[node]
        t_unique = plt.text(nx_val, ny_val, s="cryo-EM" if node == "cryo em" else node, fontsize=15, fontweight='semibold', color='#222222',
                            horizontalalignment='center', verticalalignment='center')
        texts.append(t_unique)
        target_x.append(nx_val)
        target_y.append(ny_val)

    # 6. Execute global layout solver with anchor targets passed explicitly
    adjust_text(texts, 
                x=target_x,
                y=target_y,
                objects=hub_nodes,
                expand=(1.1, 1.3),
                force_text=(0.25, 0.4),
                force_static=(0.2, 0.3),
                force_pull=(0.25, 0.25),
                iter_lim=300,
                arrowprops=dict(arrowstyle="-", color="#E0E0E0", lw=0.6), # Clean indicators if needed
                autoalign='xy')

    plt.title("Network of UoA Research Areas and Interdisciplinary Overlap", fontsize=24, fontweight='bold', pad=25)
    plt.axis('off')
    
    filename = "uoa_network_analysis.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()



# def plot_uoa_network(themes_dict, output_dir):
#     """
#     Generates a high-density, publication-quality interdisciplinary network plot
#     inspired by radial, organic cluster maps. Handles dynamic data structures.
#     """
#     G = nx.Graph()
    
#     # Define cohesive, bright palette for the 6 UoA Hubs
#     uoa_colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3']
#     uoa_to_color = {uoa: color for uoa, color in zip(themes_dict.keys(), uoa_colors)}
    
#     # 1. Build the graph dynamically from your data
#     for uoa, phrases in themes_dict.items():
#         G.add_node(uoa, type='uoa', color=uoa_to_color[uoa])
#         for phrase in phrases:
#             if not G.has_node(phrase):
#                 G.add_node(phrase, type='phrase')
#             G.add_edge(uoa, phrase)

#     # 2. AUTOMATIC SEMANTIC SEPARATION LOGIC
#     phrase_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'phrase']
#     uoa_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'uoa']
    
#     # Automatically categorize nodes based on connections (degrees)
#     shared_phrases = [n for n in phrase_nodes if G.degree(n) > 1]
#     unique_phrases = [n for n in phrase_nodes if G.degree(n) == 1]

#     # Assign colors to phrases based on their interdisciplinary profile
#     node_colors = {}
#     for node in G.nodes():
#         if G.nodes[node]['type'] == 'uoa':
#             node_colors[node] = uoa_to_color[node]
#         elif node in shared_phrases:
#             # Shared words get a neutral dark grey to hold the center together
#             node_colors[node] = '#555555' 
#         else:
#             # Unique outer words take the color of their singular parent UoA!
#             parent_uoa = list(G.neighbors(node))[0]
#             node_colors[node] = uoa_to_color[parent_uoa]

#     # 3. FORCE-DIRECTED LAYOUT OPTIMIZATION
#     # k regulates distance; a smaller k squeezes shared phrases tightly into the core, 
#     # while letting unique filaments fan outward wildly.
#     pos = nx.spring_layout(G, k=0.18, iterations=150, seed=42)
    
#     plt.figure(figsize=(20, 16), dpi=300) # Crisp, high-resolution canvas
    
#     # 4. PRECISION DRAWING
#     # Draw Edge lines (very light and thin for that delicate, string-art network feel)
#     nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='#CCCCCC', width=0.6)
    
#     # Draw Unique Filament Nodes (Small, matching parent color)
#     nx.draw_networkx_nodes(G, pos, nodelist=unique_phrases, node_size=35, 
#                            node_color=[node_colors[n] for n in unique_phrases], alpha=0.7)
    
#     # Draw Shared Core Nodes (Slightly larger, dark grey)
#     nx.draw_networkx_nodes(G, pos, nodelist=shared_phrases, node_size=60, 
#                            node_color=[node_colors[n] for n in shared_phrases], alpha=0.8)
    
#     # Draw the Main UoA Core Hubs (Massive, vibrant anchor points)
#     nx.draw_networkx_nodes(G, pos, nodelist=uoa_nodes, node_size=2500, 
#                            node_color=[node_colors[n] for n in uoa_nodes], alpha=0.95)
    
#     # 5. TYPOGRAPHY & LABELING
#     labels = {}
#     # Base labels for the big 6 Hubs
#     for node in uoa_nodes:
#         labels[node] = str(node)
        
#     # Dynamically label ALL shared tokens making up the center web
#     for sp in shared_phrases:
#         labels[sp] = str(sp)
        
#     # Draw Hub Labels (Huge, bold, impactful)
#     nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in uoa_nodes}, 
#                             font_size=16, font_weight='bold', font_family='sans-serif')
    
#     # Draw Central Shared Word Labels (Clean, legible, bounded by a soft background pill)
#     nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in shared_phrases}, 
#                             font_size=8.5, font_weight='semibold', font_family='sans-serif',
#                             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15', alpha=0.65))
    
#     plt.title("Network of UoA Research Areas and Interdisciplinary Overlap", fontsize=24, fontweight='bold', pad=20)
#     plt.axis('off')

#     # 6. EXPORT
#     filename = "uoa_network_analysis.png"
#     full_destination = os.path.join(output_dir, filename)
#     plt.savefig(full_destination, dpi=300, bbox_inches='tight', facecolor='white')
#     logger.info(f"High-impact network plot saved to {full_destination}")
#     plt.close()


# def plot_uoa_network(themes_dict, output_dir):
#     G = nx.Graph()
    
#     # Define colors for the 6 UoAs
#     uoa_colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3']
#     uoa_to_color = {uoa: color for uoa, color in zip(themes_dict.keys(), uoa_colors)}
    
#     # Build the graph
#     for uoa, phrases in themes_dict.items():
#         G.add_node(uoa, type='uoa', color=uoa_to_color[uoa])
#         for phrase in phrases:
#             if not G.has_node(phrase):
#                 G.add_node(phrase, type='phrase', color='#A9A9A9') # Default grey for phrases
#             G.add_edge(uoa, phrase)

#     # Positioning using a force-directed layout (similar to image_85d235.jpg)
#     pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
#     plt.figure(figsize=(14, 10))
    
#     # Draw Phrase Nodes (small and grey)
#     phrase_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'phrase']
#     nx.draw_networkx_nodes(G, pos, nodelist=phrase_nodes, node_size=20, node_color='#A9A9A9', alpha=0.6)
    
#     # Draw UoA Hubs (larger and colored)
#     uoa_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'uoa']
#     for node in uoa_nodes:
#         nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color=uoa_to_color[node])
        
#     # Draw edges with transparency
#     nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='grey')
    
#     # Add Labels only for UoAs and the most "connected" phrases (interdisciplinary areas)
#     labels = {node: node for node in uoa_nodes}
#     # Find phrases connected to more than one UoA
#     shared_phrases = [n for n in phrase_nodes if G.degree(n) > 1]
#     for sp in shared_phrases:
#         labels[sp] = sp
        
#     nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
#     plt.title("Network of UoA Research Areas and Interdisciplinary Overlap", fontsize=15)
#     plt.axis('off')

#     filename = f"uoa_network_analysis.png"
#     full_destination = os.path.join(output_dir, filename)
#     plt.savefig(full_destination, dpi=300, bbox_inches='tight', facecolor='white')
#     logger.info(f"Network plot saved to {full_destination}")
#     plt.close()

    # plt.show()

