# src/arabic_utils.py
import re

def simple_arabic_clean(text: str) -> str:
    """
    Minimal Arabic text cleaning
    
    Modern embeddings handle most normalization internally!
    We just remove the most obvious noise.
    
    WHY THIS WORKS:
    - Sentence-transformers models are trained on noisy web text
    - They've already learned: أ = إ = آ = ا
    - They understand diacritics are optional
    - Over-normalization can actually hurt performance!
    """
    
    # Remove diacritics (tashkeel)
    # These are usually noise for search
    arabic_diacritics = re.compile("""
                         ّ    | # Tashdid (shadda)
                         َ    | # Fatha
                         ً    | # Tanwin Fath
                         ُ    | # Damma
                         ٌ    | # Tanwin Damm
                         ِ    | # Kasra
                         ٍ    | # Tanwin Kasr
                         ْ    | # Sukun
                         ـ     # Tatwil/Kashida (elongation)
                     """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    
    # Normalize alef variants (optional - model often handles this)
    # إ أ آ ٱ → ا
    text = re.sub("[إأآٱ]", "ا", text)
    
    # Normalize ya (optional)
    # ى → ي
    text = re.sub("ى", "ي", text)
    
    # Normalize teh marbuta (optional)
    # ة → ه
    text = re.sub("ة", "ه", text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Test it
if __name__ == "__main__":
    test_texts = [
        "الصَّلاَةُ عِماد الدِّين",
        "الإســــــلام",
        "إِنَّ اللَّهَ مَعَ الصَّابِرِينَ"
    ]
    
    for text in test_texts:
        clean = simple_arabic_clean(text)
        print(f"Original: {text}")
        print(f"Clean:    {clean}")
        print()