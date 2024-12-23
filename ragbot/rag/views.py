from django.shortcuts import render, redirect
from .models import Documentizer
import PyPDF2
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine
import google.generativeai as genai
from django.http import JsonResponse

genai.configure(api_key="")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def preprocess_text(text):
    return simple_preprocess(text)

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def train_word2vec(documents):
    tokenized_docs = [preprocess_text(doc) for doc in documents]
    model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
    return model

def make_vec(text, word2vec_model):
    tokens = preprocess_text(text)
    vector = word2vec_model.wv[tokens].mean(axis=0)
    return vector.flatten()

def find_best_match(query, word2vec_model, document_vectors):
    query_tokens = preprocess_text(query)
    query_tokens = [token for token in query_tokens if token in word2vec_model.wv]

    if not query_tokens:
        return None  

    query_vector = word2vec_model.wv[query_tokens].mean(axis=0) 
    query_vector = query_vector.flatten()
    scores = [(doc, 1 - cosine(query_vector, doc['vector'])) for doc in document_vectors]
    return sorted(scores, key=lambda x: x[1], reverse=True)[0]


def generate_response(query, context):
    response = gemini_model.generate_content(f"Context: {context}\nQuestion: {query}")
    return response.text

def upload_pdf(request):
    if request.method == "POST":
        name = request.POST.get("name")
        pdf_file = request.FILES.get("file")

        if name and pdf_file:
            text = extract_text(pdf_file)

            document = Documentizer.objects.create(name=name, file=pdf_file, content=text)

            documents = Documentizer.objects.values_list('content', flat=True)
            word2vec_model = train_word2vec(documents)

            vector = make_vec(text, word2vec_model)
            document.vector = vector.tolist()
            document.save()

            return redirect('chat')

    return render(request, "upload_pdf.html")

def chat_with_pdf(request):
    if request.method == "POST":
        user_query = request.POST.get("question")

        documents = Documentizer.objects.values_list('content', flat=True)
        word2vec_model = train_word2vec(documents)

        document_vectors = [
            {'name': doc.name, 'vector': make_vec(doc.content, word2vec_model), 'content': doc.content}
            for doc in Documentizer.objects.all()
        ]

        best_match = find_best_match(user_query, word2vec_model, document_vectors)
        if not best_match:
            return JsonResponse({'response': "No relevant document found or query tokens not in vocabulary."})
        best_match_doc, score = best_match
        
        context = best_match_doc['content'][:1000]  
        response_text = generate_response(user_query, context)

        return JsonResponse({'response': response_text})

    return render(request, "chat.html")
