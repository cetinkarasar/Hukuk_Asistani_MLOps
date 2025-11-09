import streamlit as st
import torch
import os
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SABİTLER ve AYARLAR ---

# ====> BURAYI DEĞİŞTİRİN <====
HF_REPO_NAME = "Cetin003/hukuk_model_tck_v1_lora"
# ===============================

PDF_PATH = "data/tck.pdf"
FAISS_INDEX_PATH = "faiss_tck_kutuphanesi"
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v1"

ALPACA_PROMPT = """Aşağıda, bir görevi açıklayan bir talimat (instruction) ile daha fazla bağlam sağlayan bir girdi (input) bulunmaktadır. İsteği uygun şekilde tamamlayan bir yanıt (output) yazın.

### Talimat:
{}

### Girdi:
{}

### Yanıt:
{}"""

# --- 2. UYGULAMA FONKSİYONLARI (Streamlit Önbellekleme ile) ---
# Bu fonksiyonlar, ağır modellerin sadece bir kez yüklenmesini sağlar.

@st.cache_resource
def load_generator_model():
    st.info(f"Uzman modeliniz ({HF_REPO_NAME}) Hugging Face'ten yükleniyor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = HF_REPO_NAME,
        load_in_4bit = True,
        dtype = None,
    )
    
    generation_params = {
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.3,
        "repetition_penalty": 1.15,
        "top_p": 0.95,
        "top_k": 40
    }
    st.success("Uzman model (Generator) başarıyla yüklendi!")
    return model, tokenizer, generation_params

@st.cache_resource
def load_retriever():
    st.info(f"Embedding modeli ({EMBEDDING_MODEL_NAME}) yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    if not os.path.exists(FAISS_INDEX_PATH):
        st.warning("Kütüphane (FAISS index) bulunamadı. PDF'ten yeni bir tane oluşturuluyor...")
        
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success("Kütüphane oluşturuldu ve diske kaydedildi!")
    
    else:
        st.info(f"Hazır kütüphane ({FAISS_INDEX_PATH}) diskten yükleniyor...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.success("Kütüphane (Retriever) başarıyla yüklendi!")
        
    return vector_store

def hukukAsistani(kullanici_sorusu, vector_store, model, tokenizer, generation_params):
    # === ADIM 1: RAG (Retriever) ===
    ilgili_metinler = vector_store.similarity_search(kullanici_sorusu, k=3)
    context_input = "\n\n".join([doc.page_content for doc in ilgili_metinler])

    # === ADIM 2: PROMPT MÜHENDİSLİĞİ (Bağlam Önce!) ===
    asistan_talimati = """Görevin, 'Bağlam' olarak verilen kanun metinlerini okumak ve bu metinlere dayanarak 'Kullanıcı Sorusu'nu cevaplamaktır.
    
    KURALLAR:
    1. Cevabın SADECE VE SADECE 'Bağlam' içindeki bilgilere dayanmalıdır.
    2. 'Bağlam' dışında ASLA bilgi verme veya yorum yapma.
    3. Eğer cevap 'Bağlam' içinde yoksa, 'Verilen metinde bu bilgi bulunmamaktadır.' de."""
    
    girdi_metni = f"""Aşağıda 'Bağlam' ve 'Kullanıcı Sorusu' bulunmaktadır.

    [Bağlam Başlangıcı]
    {context_input}
    [Bağlam Sonu]

    [Kullanıcı Sorusu]
    {kullanici_sorusu}
    """
    
    prompt = ALPACA_PROMPT.format(asistan_talimati, girdi_metni, "")

    # === ADIM 3: GENERATOR (Uzman Model) ===
    inputs = tokenizer([prompt], return_tensors = "pt", padding = False, truncation = True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)
    
    # === ADIM 4: CEVAP ===
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    response_start_index = decoded_output.find("### Yanıt:")
    if response_start_index != -1:
        response = decoded_output[response_start_index + len("### Yanıt:"):]
        return response.strip()
    else:
        return decoded_output

# --- 3. STREAMLIT ARAYÜZÜ ---

def main():
    st.set_page_config(page_title="TCK Hukuk Asistanı", layout="wide")
    st.title("⚖️ TCK Hukuk Asistanı (RAG + Fine-Tuned LLM)")
    
    try:
        vector_store = load_retriever()
        model, tokenizer, generation_params = load_generator_model()
    except Exception as e:
        st.error(f"Modeller yüklenirken kritik bir hata oluştu: {e}")
        st.error("Lütfen HF Spaces ayarlarından T4 GPU'nun seçili olduğundan emin olun.")
        return

    kullanici_sorusu = st.text_input("Lütfen TCK ile ilgili sorunuzu buraya yazın:", placeholder="Kasten yaralamanın cezası nedir?")

    if kullanici_sorusu:
        with st.spinner("Asistanınız RAG kütüphanesinde arama yapıyor ve uzman modelinizle cevap üretiyor..."):
            cevap = hukukAsistani(kullanici_sorusu, vector_store, model, tokenizer, generation_params)
            st.markdown("---")
            st.subheader("Asistanın Cevabı:")
            st.write(cevap)

if __name__ == "__main__":
    main()