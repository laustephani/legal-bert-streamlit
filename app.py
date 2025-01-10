import streamlit as st
from sentence_transformers import SentenceTransformer

# Título do aplicativo
st.title("Legal-BERT PT-BR: Transformador de Sentenças Jurídicas")

# Carregar o modelo
@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer("ulysses-camara/legal-bert-pt-br")
    return model

model = load_model()

# Entrada do usuário
sentence = st.text_input("Digite uma sentença para gerar o embedding:")

if sentence:
    # Gerar o embedding
    embedding = model.encode(sentence)
    st.write("Embedding gerado:")
    st.write(embedding)
