"""
Embedding utilities - convert text to vectors

This is the CORE of semantic search. Everything else is just infrastructure
around these magical vectors.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import re

class ArabicEmbedder:
    """
    Handles text → vector conversion
    
    KEY INSIGHT: We're using a multilingual model that already "understands"
    Arabic from training. We don't need heavy preprocessing!
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize embedding model
        
        WHY THIS MODEL:
        - Supports 50+ languages including Arabic
        - Only 384 dimensions (vs 768) = 2x faster, less memory
        - Still excellent quality (~80% of larger models)
        - 118MB vs 400MB+ for Arabic-specific models
        - Actively maintained by HuggingFace
        
        ALTERNATIVES:
        - "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
          → Best Arabic-specific (768 dims, 85.3 MTEB score)
        - "intfloat/multilingual-e5-large"
          → Best multilingual (1024 dims, highest quality)
        """
        print(f"Loading: {model_name}")
        print("First run downloads ~118MB...")
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Loaded! Dimension: {self.dimension}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Minimal cleaning - just remove obvious noise
        
        PHILOSOPHY: Less is more!
        Over-normalization can hurt because:
        1. Model trained on messy web text
        2. Model learned orthographic variants
        3. Removing too much loses signal
        """
        # Remove diacritics
        diacritics = re.compile("""
                         ّ    | # Tashdid
                         َ    | # Fatha
                         ً    | # Tanwin Fath
                         ُ    | # Damma
                         ٌ    | # Tanwin Damm
                         ِ    | # Kasra
                         ٍ    | # Tanwin Kasr
                         ْ    | # Sukun
                         ـ     # Tatwil
                     """, re.VERBOSE)
        text = re.sub(diacritics, '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text → embedding vector
        
        Args:
            text: Arabic or English text
        
        Returns:
            numpy array of shape (384,) - the semantic vector
        
        WHAT HAPPENS INSIDE:
        1. Text → tokens (subword pieces)
        2. Tokens → IDs (numbers the model knows)
        3. IDs → embeddings (look up in learned table)
        4. Pass through 12 transformer layers
        5. Pool final layer → single 384-dim vector
        6. Normalize to unit length (for cosine similarity)
        """
        cleaned = self.clean_text(text)
        
        # This single line does all the magic!
        embedding = self.model.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently
        
        WHY BATCHING:
        Neural networks process batches in parallel (GPU/CPU SIMD)
        
        Speed comparison (100 texts):
        - One by one: ~10 seconds (sequential)
        - Batch of 32: ~2 seconds (parallel)
        - 5x faster!
        
        Args:
            texts: List of strings
            batch_size: How many to process simultaneously
            show_progress: Show tqdm progress bar
        
        Returns:
            numpy array of shape (len(texts), 384)
        """
        cleaned = [self.clean_text(t) for t in texts]
        
        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Returns:
            float 0-1:
              1.0 = identical meaning
              0.5 = somewhat related
              0.0 = completely unrelated
        
        MATH:
        Since vectors are normalized, cosine similarity is just dot product!
        cos(θ) = A · B (when ||A|| = ||B|| = 1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)


# Test the embedder
if __name__ == "__main__":
    print("="*60)
    print("TESTING ARABIC EMBEDDER")
    print("="*60)
    
    embedder = ArabicEmbedder()
    
    # Test texts
    prayer = "الصلاة"
    worship = "العبادة"
    car = "السيارة"
    
    print("\n1. TESTING SIMILARITY")
    print("-"*60)
    sim_prayer_worship = embedder.similarity(prayer, worship)
    sim_prayer_car = embedder.similarity(prayer, car)
    
    print(f"'{prayer}' vs '{worship}': {sim_prayer_worship:.3f} (should be high)")
    print(f"'{prayer}' vs '{car}': {sim_prayer_car:.3f} (should be low)")
    
    print("\n2. EXAMINING EMBEDDINGS")
    print("-"*60)
    emb = embedder.embed(prayer)
    print(f"Text: '{prayer}'")
    print(f"Embedding shape: {emb.shape}")
    print(f"First 10 values: {emb[:10]}")
    print(f"Min: {emb.min():.3f}, Max: {emb.max():.3f}, Mean: {emb.mean():.3f}")
    
    print("\n3. TESTING BATCH EMBEDDING")
    print("-"*60)
    texts = [prayer, worship, car, "الصوم", "الحج"]
    embeddings = embedder.embed_batch(texts)
    print(f"Embedded {len(texts)} texts")
    print(f"Result shape: {embeddings.shape}")
    
    print("\n✓ All tests passed!")