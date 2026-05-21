import os
import re
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from adjustText import adjust_text
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer



def extract_uoa_research_areas(df, text_col='Titl_and_Abs_Clean', uoa_col='Unit_of_assessment_number', top_n=10):
    # 1. Group all abstracts by UoA to create "Class Documents"
    # This treats each UoA as one single giant text block
    uoa_docs = df.groupby(uoa_col)[text_col].apply(lambda x: ' '.join(x)).reset_index()

    # 2. Define a clean, year-agnostic dynamic cleaner for the vectorizer
    def dynamic_vectorizer_cleaner(doc):
        if not isinstance(doc, str):
            return ""
        # Lowercase to match vectorizer behavior
        doc = doc.lower()
        # Dynamically erase ANY 4-digit year followed by author/authors/authorship
        # e.g., "2018 author", "2022 authors", "2025 author" become completely blank spaces
        doc = re.sub(r'\b\d{4}\s+author\w*', ' ', doc)
        return doc

    # 3. Initialize TfidfVectorizer with N-gram support
    # stop_words='english' is vital to remove "the", "and", etc.
    # ngram_range=(2, 3) forces it to look for phrases/areas specifically
    domain_stopwords = [
    'rights reserved', 'et al', '95 ci', '95 confidence', 'confidence interval',
    'john wiley', 'wiley sons', 'springer nature', 'elsevier', 'publisher',
    'american psychological', 'psychological association', 'abstract available',
    'study', 'results', 'conclusions', 'background', 'methods'
    ]
    final_stopwords = text.ENGLISH_STOP_WORDS.union(domain_stopwords)

    # 4. Initialize TfidfVectorizer using the dynamic preprocessor hook
    vectorizer = TfidfVectorizer(
        preprocessor=dynamic_vectorizer_cleaner,
        stop_words=list(final_stopwords), 
        ngram_range=(2, 3), 
        max_features=10000
    )

    # 5. Fit and transform the Class Documents
    tfidf_matrix = vectorizer.fit_transform(uoa_docs[text_col])
    feature_names = vectorizer.get_feature_names_out()

    # 6. Extract top phrases per UoA
    uoa_themes = {}
    for i, row in uoa_docs.iterrows():
        uoa_name = row[uoa_col]
        # Get the tf-idf scores for this UoA's "document"
        scores = tfidf_matrix[i].toarray().flatten()
        # Sort and get the indices of the top N scores
        top_indices = scores.argsort()[-top_n:][::-1]
        # Map indices to the actual phrases
        top_phrases = [feature_names[idx] for idx in top_indices]
        uoa_themes[uoa_name] = top_phrases
        
    return uoa_themes



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
    
    # OPTIONAL FINE-TUNING: Scaling factor normalization 
    # If the layout still feels slightly wide, we can squash coordinates globally:
    # pos = {node: np.array([coords[0] * 0.85, coords[1]]) for node, coords in pos.items()}
    plt.figure(figsize=(22, 14), dpi=300)
    
    # FIX: Darkened edge settings for visibility (alpha=0.3, width=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#BDBDBD', width=0.8)
    
    # FIX: Explicitly passing the custom color lists to all node types
    nx.draw_networkx_nodes(G, pos, nodelist=unique_phrases, node_size=50, 
                           node_color=[node_colors[n] for n in unique_phrases], alpha=0.85)
    nx.draw_networkx_nodes(G, pos, nodelist=shared_phrases, node_size=90, 
                           node_color=[node_colors[n] for n in shared_phrases], alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=uoa_nodes, node_size=3200, 
                           node_color=[node_colors[n] for n in uoa_nodes], alpha=0.95)
    
    # 3. Draw Hub Labels
    nx.draw_networkx_labels(G, pos, labels={n: n for n in uoa_nodes}, font_size=18, font_weight='bold')
    
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
        t = plt.text(nx_val, ny_val, s=node,
                     fontsize=8.5, fontweight='bold', family='sans-serif',
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.85))
        texts.append(t)
        target_x.append(nx_val)
        target_y.append(ny_val)

    # 5. Draw Peripheral Unique Labels
    for node in unique_phrases:
        nx_val, ny_val = pos[node]
        t_unique = plt.text(nx_val, ny_val, s=node, fontsize=8, fontweight='semibold', color='#222222',
                            horizontalalignment='center', verticalalignment='center')
        texts.append(t_unique)
        target_x.append(nx_val)
        target_y.append(ny_val)

    # 6. Execute global layout solver with anchor targets passed explicitly
    adjust_text(texts, 
                x=target_x,
                y=target_y,
                expand_points=(1.8, 1.8), 
                expand_text=(1.3, 1.5),
                force_text=(0.15, 0.3),
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

