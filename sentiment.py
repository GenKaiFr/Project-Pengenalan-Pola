import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import math

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Dataset kata-kata sentimen
kata_positif = [
    "senang", "bahagia", "menyenangkan", "bagus", "suka", "cinta", "gembira", "puas", "hebat", "mantap",
    "indah", "cantik", "luar biasa", "fantastis", "menakjubkan", "sempurna", "terbaik", "keren", "asyik", "menarik",
    "berhasil", "sukses", "bangga", "optimis", "antusias", "bersemangat", "excited", "amazing", "wonderful", "excellent",
    "lega", "tenang", "damai", "nyaman", "fresh", "segar", "cerah", "positif", "beruntung", "grateful"
]

kata_negatif = [
    "sedih", "buruk", "kecewa", "marah", "benci", "jelek", "tidak", "bosan", "lelah", "susah",
    "menyebalkan", "kesal", "jengkel", "stress", "depresi", "putus asa", "hopeless", "terrible", "awful", "bad",
    "gagal", "rugi", "sakit", "pusing", "mual", "muntah", "demam", "flu", "batuk", "pilek",
    "takut", "cemas", "khawatir", "nervous", "panik", "gelisah", "resah", "hancur", "rusak", "patah"
]

def hitung_skor(text):
    kata = text.lower().split()
    skor_positif = sum(1 for k in kata if k in kata_positif)
    skor_negatif = sum(1 for k in kata if k in kata_negatif)
    return skor_positif, skor_negatif

def prediksi_sentimen(text):
    pos, neg = hitung_skor(text)
    
    if pos == 0 and neg == 0:
        return "netral", 0.5, 0.5
    
    # Hitung probabilitas langsung tanpa baseline yang membingungkan
    total = pos + neg
    if total == 0:
        prob_pos = 0.5
        prob_neg = 0.5
    else:
        prob_pos = pos / total
        prob_neg = neg / total
    
    if pos > neg:
        prediksi = "positif"
    elif neg > pos:
        prediksi = "negatif"
    else:
        prediksi = "netral"
    
    return prediksi, prob_pos, prob_neg

# NLTK preprocessing functions
def preprocess_text(text):
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords (Indonesian + English)
        stop_words = {'dan', 'atau', 'yang', 'ini', 'itu', 'adalah', 'dengan', 'untuk', 'dari', 'ke', 'di', 'pada', 'dalam', 'akan', 'telah', 'sudah', 'juga', 'dapat', 'bisa', 'harus', 'sangat', 'sekali', 'banget'}
        tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        # Simple stemming for Indonesian
        tokens = [token.replace('nya', '').replace('kan', '').replace('an', '') for token in tokens]
        return ' '.join(tokens)
    except:
        # Fallback if NLTK fails
        words = text.lower().split()
        return ' '.join([w for w in words if w.isalpha()])

# Manual Naive Bayes classifier
class NaiveBayesSentiment:
    def __init__(self):
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocabulary = set()
        self.trained = False
    
    def train(self, texts, labels):
        for text, label in zip(texts, labels):
            processed_text = preprocess_text(text)
            words = processed_text.split()
            
            self.class_counts[label] += 1
            
            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)
        
        self.trained = True
    
    def predict(self, text):
        if not self.trained:
            return "netral", 0.33, 0.33, 0.33
        
        processed_text = preprocess_text(text)
        words = processed_text.split()
        
        class_scores = {}
        total_docs = sum(self.class_counts.values())
        
        for class_name in self.class_counts:
            # Prior probability
            prior = self.class_counts[class_name] / total_docs
            
            # Likelihood
            likelihood = 1.0
            total_words_in_class = sum(self.word_counts[class_name].values())
            vocab_size = len(self.vocabulary)
            
            for word in words:
                word_count = self.word_counts[class_name][word]
                # Laplace smoothing
                word_prob = (word_count + 1) / (total_words_in_class + vocab_size)
                likelihood *= word_prob
            
            class_scores[class_name] = prior * likelihood
        
        # Normalize probabilities
        total_score = sum(class_scores.values())
        if total_score == 0:
            return "netral", 0.33, 0.33, 0.33
        
        probabilities = {k: v/total_score for k, v in class_scores.items()}
        
        # Get prediction
        prediction = max(probabilities, key=probabilities.get)
        
        return prediction, probabilities.get('positif', 0), probabilities.get('negatif', 0), probabilities.get('netral', 0)

# Training data for Naive Bayes
training_texts = [
    "saya sangat senang hari ini", "aku bahagia sekali", "ini menyenangkan", "bagus banget",
    "aku suka ini", "cinta banget", "gembira sekali", "puas dengan hasil", "hebat sekali",
    "mantap jiwa", "indah sekali", "cantik banget", "luar biasa", "fantastis",
    "aku sedih sekali", "ini buruk", "kecewa banget", "marah sekali", "benci ini",
    "jelek banget", "tidak suka", "bosan sekali", "lelah banget", "susah sekali",
    "menyebalkan", "kesal banget", "jengkel sekali", "stress banget", "depresi",
    "hari ini biasa saja", "tidak ada yang spesial", "standar", "lumayan", "cukup"
]

training_labels = [
    "positif", "positif", "positif", "positif", "positif", "positif", "positif", "positif", "positif", "positif",
    "positif", "positif", "positif", "positif",
    "negatif", "negatif", "negatif", "negatif", "negatif", "negatif", "negatif", "negatif", "negatif", "negatif",
    "negatif", "negatif", "negatif", "negatif", "negatif",
    "netral", "netral", "netral", "netral", "netral"
]

# Initialize and train Naive Bayes model
nb_model = NaiveBayesSentiment()
try:
    nb_model.train(training_texts, training_labels)
    print("Naive Bayes model trained successfully!")
except Exception as e:
    print(f"Warning: Naive Bayes training failed: {e}")
    nb_model.trained = False

def pecah_kalimat(kalimat):
    pemecah = r"(,|\babis itu\b|\babis tu\b|\btapi\b|\btetapi\b|\bnamun\b|\blalu\b|\bkemudian\b|\bdan\b)"
    parts = re.split(pemecah, kalimat, flags=re.IGNORECASE)
    
    kalimat_bagian = []
    temp = ""
    
    for p in parts:
        if re.match(pemecah, p, re.IGNORECASE):
            if temp.strip():
                kalimat_bagian.append(temp.strip())
            temp = ""
        else:
            temp += " " + p
    
    if temp.strip():
        kalimat_bagian.append(temp.strip())
    
    return [b.strip() for b in kalimat_bagian if b.strip()]

# === MAIN PROGRAM ===
kalimat = "Saya merasa senang, bahagia, dan bangga karena hasil kerja yang bagus, keren, dan berhasil, bahkan terasa luar biasa dan mantap, namun di sisi lain sempat muncul perasaan lelah, cemas, dan khawatir karena beberapa kendala yang menyebalkan, membuat situasi terasa stress dan hampir putus asa, meskipun akhirnya saya tetap optimis, bersemangat, dan grateful karena masalah tersebut tidak berujung gagal atau buruk."

print("Kalimat asli:", kalimat)
print("\n" + "="*60)

# Pecah kalimat
bagian_kalimat = pecah_kalimat(kalimat)

print("\n=== HASIL ANALISIS PER BAGIAN (LEXICON-BASED) ===")
positif_count = 0
negatif_count = 0

for i, bagian in enumerate(bagian_kalimat, 1):
    prediksi, prob_pos, prob_neg = prediksi_sentimen(bagian)
    
    if prediksi == "positif":
        positif_count += 1
    elif prediksi == "negatif":
        negatif_count += 1
    
    print(f"\nBagian {i}: '{bagian}'")
    print(f"-> Prediksi: {prediksi.upper()}")
    print(f"-> Probabilitas Positif: {prob_pos:.1%}")
    print(f"-> Probabilitas Negatif: {prob_neg:.1%}")

print("\n=== HASIL ANALISIS NAIVE BAYES ===")
for i, bagian in enumerate(bagian_kalimat, 1):
    prediksi_nb, prob_pos_nb, prob_neg_nb, prob_net_nb = nb_model.predict(bagian)
    
    print(f"\nBagian {i}: '{bagian}'")
    print(f"-> Prediksi: {prediksi_nb.upper()}")
    print(f"-> Probabilitas Positif: {prob_pos_nb:.1%}")
    print(f"-> Probabilitas Negatif: {prob_neg_nb:.1%}")
    print(f"-> Probabilitas Netral: {prob_net_nb:.1%}")

print("\n" + "="*60)
print("=== RINGKASAN SENTIMEN ===")
print(f"Total bagian positif: {positif_count}")
print(f"Total bagian negatif: {negatif_count}")

if positif_count > negatif_count:
    kesimpulan = "POSITIF"
elif negatif_count > positif_count:
    kesimpulan = "NEGATIF"
else:
    kesimpulan = "SEIMBANG"

print(f"\nKesimpulan: Secara keseluruhan cerita ini memiliki sentimen {kesimpulan}.")