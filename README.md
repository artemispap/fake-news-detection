# Fake News Detection

## Abstract
This project develops a machine learning system to distinguish between **real and fake news** articles.  
Using transformer-based models (BERT, DistilBERT) and an LSTM baseline, we evaluate trade-offs between accuracy, efficiency, and generalization.  
The final solution combines **BERT with a CNN classification head**, achieving strong performance and enabling deployment through **Gradio**.



## Dataset
- Source: [Fake News Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)  
- Total articles: **44,898**
  - Fake: 23,481  
  - Real: 21,417 (Reuters)  
- Features: `title` + `text` combined into **content**  
- Split: **80% train / 10% validation / 10% test** (stratified)  
- Preprocessing:
  - Lowercasing, URL/HTML removal, punctuation cleaning  
  - Removed `Reuters` prefix (to prevent data leakage)  
  - Tokenized with `bert-base-uncased` (max length = 128)



## Models

### 1) **BERT + CNN (Final Model)**
- Frozen `bert-base-uncased` encoder + Conv1D + GlobalMaxPooling + Dense layers  
- Validation accuracy: **97%+**  
- Test accuracy: **96.6%**  
- Balanced precision/recall across both classes  

### 2) **BERT + CNN — 5‑Fold Cross‑Validation**
- **Setup:** Stratified **k=5** folds, EarlyStopping, ReduceLROnPlateau (BERT frozen).  
- **Cross‑val results:** Accuracy **0.9792 ± 0.0010**, Loss **0.0594 ± 0.0027**.  
- **Held‑out test (10%)**: Accuracy **>97%**.  
- **Confusion matrix (test):** Real correct **2048** (FN **94**), Fake correct **2270** (FP **78**).  
- **Why it matters:** More **robust** estimate with **low variance** across splits.  
- **Trade‑off:** Higher compute (≈ **5,000 s per fold** in runs).

### 3) **BERT (Pooled Classifier)**
- Uses pooled `[CLS]` output + MLP  
- Test accuracy: **87.4%**  
- Tends to over‑flag real articles as fake (higher false positives than BERT–CNN)  

### 4) **DistilBERT**
- Lightweight backbone with GlobalAveragePooling + MLP  
- Test accuracy: **93.9%**  
- Faster and smaller—good for latency‑sensitive deployments  

### 5) **NLTK + LSTM (Baseline)**
- Tokenizer + LSTM trained from scratch (no pretrained embeddings)  
- Test accuracy: **99.0%** (strong on this dataset; may generalize less beyond domain)



## Tools
- **Python** (TensorFlow/Keras, HuggingFace Transformers)  
- **pandas, scikit-learn** (data handling & evaluation)  
- **matplotlib, seaborn** (visualization)  
- **Google Colab** (experiments)  
- **Gradio** (deployment demo)  



## Deployment
A **Gradio interface** allows users to paste text and receive real-time predictions:  
- Classifies text as **Real** or **Fake**  
- Displays prediction confidence  
- Includes a **user feedback flag button**  



## Conclusion
- **BERT–CNN** provided the best balance between accuracy and efficiency.  
- **5‑fold cross‑validation** confirms robustness and low variance.  
- **DistilBERT** is promising for lightweight deployment.  
- **LSTM** reached very high accuracy here but may not generalize as well as transformers.  
- Overall, transformer + CNN hybrids are strong candidates for **fake news detection** tasks.



## Authors
- Παπασπύρου Αρτεμισία (p2822424)  
- Παριανού Ζωή-Αικατερίνη (p2822425)  
