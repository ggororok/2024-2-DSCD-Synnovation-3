# 필요한 라이브러리 임포트
import torch
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# 1. pdfplumber로 PDF 텍스트 추출 함수 정의
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# 텍스트 추출 및 Document 객체로 변환
pdf_text = extract_text_from_pdf("한화생명 간편가입 시그니처 암보험(갱신형) 무배당_2055-001_002_약관_20220601_(2).pdf")
documents = [Document(page_content=pdf_text)]

# 문서를 적절한 크기로 청크 나누기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 2. BM25 Retriever 설정
bm25_retriever = BM25Retriever.from_documents(chunks)

# 3. LLM과 결합하여 답변 생성
# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")

# 모델 메모리 설정 및 로드
model = AutoModelForCausalLM.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B",
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16  # fp16으로 로드하여 메모리 사용량 절감
)

# QA 파이프라인 생성
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# 요약 파이프라인 생성
summarization_pipeline = pipeline("summarization", model="hyunwoongko/kobart")

# 질문에 대한 답변을 생성하는 함수 정의
def query_bm25(query):
    # BM25로 관련 문서 검색 (상위 3개 결과 사용)
    try:
        results = bm25_retriever.invoke(query)[:3]
    except Exception as e:
        print(f"An error occurred while retrieving documents: {e}")
        return "Error occurred during document retrieval."
    
    if not results:
        return "No relevant documents found."
    
    summaries = []
    for doc in results:
        try:
            content = doc.page_content.strip()[:512] if doc.page_content else ""
            if len(content) > 10:
                summary = summarization_pipeline(content, max_length=100, min_length=30, do_sample=True)[0]['summary_text']
                summaries.append(summary)
            else:
                print("Document content is too short or empty.")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"An error occurred while summarizing: {e}")
            continue
    
    if not summaries:
        return "Summarization failed for all documents."
    
    context = " ".join(summaries)
    input_text = f"{context}"
    
    # LLM을 사용하여 답변 생성
    llm_response = qa_pipeline(input_text, max_new_tokens=100)[0]['generated_text']
    torch.cuda.empty_cache()
    
    return llm_response

# 5. Gradio Blocks 인터페이스 생성
with gr.Blocks() as iface:
    gr.Markdown("# 보험 문서 챗봇\n보험 문서에 대해 물어보면 답하는 챗봇입니다.")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="뭐든지 물어보세요.", label="챗 입력")
    
    with gr.Row():
        submit_btn = gr.Button("보내기")
        retry_btn = gr.Button("다시보내기 ↩")
        undo_btn = gr.Button("이전챗 삭제 ❌")
        clear_btn = gr.Button("전챗 삭제 💫")
    
    # 채팅 제출 시 응답 생성
    def submit(message, history):
        response_message = query_bm25(message)  # 질문에 대한 답변 생성
        history.append((message, response_message))  # 대화 기록에 추가
        return history, ""

    # 버튼 기능 연결
    submit_btn.click(submit, [msg, chatbot], [chatbot, msg])
    retry_btn.click(lambda: None, None, chatbot)  # 다시 보내기 기능 구현 필요
    undo_btn.click(lambda history: history[:-1], [chatbot], chatbot)  # 이전 메시지 삭제
    clear_btn.click(lambda: [], None, chatbot)  # 전체 메시지 삭제

iface.launch()
