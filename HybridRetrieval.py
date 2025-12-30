#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# --- 1. Privacy Knowledge Graph) ---


with open('KG.json', 'r', encoding='utf-8') as f:

    PRIVACY_KG = json.load(f)



class HybridPrivacyRetriever:
    def __init__(self, kg):
        self.kg = kg
        # Extract descriptions and trigger words from the knowledge graph as a retrieval library
        self.corpus = []
        self.pattern_map = []
        for pattern_id, info in kg.items():
            content = f"{info['description']} {' '.join(info['entities'])} {' '.join(info['triggers'])}"
            self.corpus.append(content)
            self.pattern_map.append(pattern_id)

        #  BM25
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Initialize the semantic model (using a lightweight Chinese model)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def _kg_pattern_match(self, pattern_id):

        data = self.kg[pattern_id]
        return {
            "pattern": pattern_id,
            "entities": data["entities"],
            "triggers": data["triggers"],
            "description": data["description"]
        }

    def retrieve(self, query, bm25_threshold=1.5, semantic_threshold=0.6):
        print(f"\n[Start retrieval] Search term: '{query}'")

        # --- Step 1: BM25 Text Preliminary Retrieval ---
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argmax(bm25_scores)

        # --- Step 2: Semantic Embedding Matching Verification ---
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_sim = util.cos_sim(query_embedding, self.corpus_embeddings[top_idx]).item()


        if bm25_scores[top_idx] > bm25_threshold and cos_sim > semantic_threshold:

            pattern_id = self.pattern_map[top_idx]
            return self._kg_pattern_match(pattern_id)

        # ---Step 3: Word Embedding Retrieval Algorithm (Semantic Completion) ---

        semantic_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        best_semantic_idx = np.argmax(semantic_scores).item()

        if semantic_scores[best_semantic_idx] > semantic_threshold:

            pattern_id = self.pattern_map[best_semantic_idx]
            return self._kg_pattern_match(pattern_id)

        return None


# --- 4. Main  ---
def main():

    retriever = HybridPrivacyRetriever(PRIVACY_KG)


    case1 = ""


    case2 = ""


    case3 = ""

    test_cases = [case1, case2, case3]

    for i, query in enumerate(test_cases):
        print("-" * 50)
        result = retriever.retrieve(query)

        if result:
            print(f"[Decision]Similar privacy risk detected!")
            print(f"  - Risk pattern,: {result['pattern']} ({result['description']})")
            print(f"  - Related entity: {', '.join(result['entities'])}")
            print(f"  - Privacy Trigger: {', '.join(result['triggers'])}")
        else:
            print("No significant privacy breach risk was identified")


if __name__ == "__main__":
    main()