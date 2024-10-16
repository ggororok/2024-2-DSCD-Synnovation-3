from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import subprocess
from transformers import pipeline
import gradio as gr

# 1. PDF 파일에서 텍스트 로드 및 청크 분할
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks

# 2. 로컬 GGUF 모델을 사용하여 임베딩 생성 함수
def generate_embedding_with_llama(text):
    result = subprocess.run([
        "C:/Users/SAMSUNG/AppData/Local/Programs/Ollama/ollama app.exe",
        "-m", "C:/Users/SAMSUNG/Desktop/동국대학교/2024년 4학년 2학기/데이터사이언스캡스톤디자인/dgu_team/ggml-model-Q5_K_M.gguf",
        "--input", text
    ], capture_output=True, text=True)

    embedding = result.stdout.strip().split()
    return [float(x) for x in embedding]

# 3. 미리 PDF를 임베딩하여 벡터스토어 생성 및 저장
def prepare_faiss_vectorstore(pdf_files):
    all_embeddings = []
    all_chunks = []

    for pdf_path in pdf_files:  # pdf_files가 여러 PDF 경로를 담고 있음
        chunks = load_and_split_pdf(pdf_path)
        for chunk in chunks:
            embedding = generate_embedding_with_llama(chunk.page_content)
            all_embeddings.append(embedding)
            all_chunks.append(chunk)

    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_embeddings(all_embeddings, all_chunks)
    return vectorstore

# PDF 파일 경로 리스트 (필수로 지정)
pdf_files = ['report1.pdf', 'report2.pdf']  # 이 부분에서 PDF 경로를 명확히 설정

# 벡터스토어 생성 (PDF 파일 리스트를 함수로 전달)
vectorstore = prepare_faiss_vectorstore(pdf_files)

# 4. 질문을 임베딩하고 관련 문서 검색
def search_similar_documents(vectorstore, query, top_k=3):
    query_embedding = generate_embedding_with_llama(query)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    
    return results

# 5. LLM을 사용해 검색된 문서와 질문에 대한 응답 생성
def generate_rag_response(query, search_results):
    generator = pipeline('text-generation', model='gpt2')
    
    # 검색된 문서와 질문을 결합하여 입력으로 사용
    context = " ".join([doc.page_content for doc in search_results])
    input_text = f"질문: {query}\n문서 내용: {context}\n답변:"
    
    response = generator(input_text, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# 6. Gradio 인터페이스에서 질문 입력 및 응답 생성
def chatbot_response(query):
    # FAISS에서 유사 문서 검색
    search_results = search_similar_documents(vectorstore, query)
    
    # LLM을 통해 최종 답변 생성
    response = generate_rag_response(query, search_results)
    
    return response

# 7. Gradio UI 구성
with gr.Blocks() as interface:
    gr.Markdown("# PDF 기반 RAG 챗봇")
    
    # 질문 입력창과 응답 출력창
    query_input = gr.Textbox(label="질문을 입력하세요", placeholder="질문 입력")
    output_box = gr.Textbox(label="챗봇 응답", placeholder="결과가 여기 표시됩니다.", lines=10)
    
    # 버튼 설정
    submit_button = gr.Button("보내기")
    
    # 버튼 클릭 시 처리
    def submit_fn(query):
        return chatbot_response(query)
    
    submit_button.click(submit_fn, query_input, output_box)

# Gradio 인터페이스 실행
interface.launch()
