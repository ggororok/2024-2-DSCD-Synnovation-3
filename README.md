<h1 align="center"> 2024-2-DSCD-Synnovation-3 </h1>
<div align="center"> <b>동국대학교 데이터 사이언스 캡스톤 디자인(졸업 프로젝트)을 진행하는 3조 Synnovation 팀 입니다.</b> </div> </br>
<div align="center"> <img src="https://github.com/user-attachments/assets/ba0220e7-d89c-41cb-b3ed-8f9d44a05267" alt="메인이미지" width="900"> </div>
</br>
</br>

## 🤖 주제: RAG를 활용한 보험문서 챗봇 만들기 프로젝트

**참여기간**  </br>
24.09.02~12.15 </br>

**참여인원**  </br>

|이름|역할|깃헙 아이디|
|------|---|---|
| 김용태 | 팀장, 파인튜닝 모델 | [YongTae0](https://github.com/YongTae0) |
| 김민지 | RAG 모델 | [ggororok](https://github.com/ggororok) |
| 박준하 | 서버 관리, 파인튜닝 모델 |[joonhai](https://github.com/joonhai)|
| 원종철 | RAG 모델 | [JongCheolWon](https://github.com/JongCheolWon) | 
</br>

## ✨ 프로젝트 목표
### 1️⃣ RAG를 활용한 LLM 챗봇 모델 개발 
**1. RAG 사용의 적합성 평가** 
  - 해당 모델(RAG)과 파인튜닝 LLM모델 성능 비교
  - 보험 관련 질의에 대한 답변 정확도 평가 </br>

**2. 학습 비용 및 시간 문제 해결**
  - 파인튜닝 모델 대비 RAG의 비율, 시간 절감 효과 확인 </br>

**3. RAG 성능 일관성 평가**
  - 여러 보험문서(손해보험, 자동차보험)에도 RAG의 성능이 보장되는지 확인 </br>
  
### 2️⃣ Gradio를 통한 웹 애플리케이션 개발 
**1. 보험에 대해 잘 모르는 사람을 위한 기능** 
- 용어사전 링크 제공

**2. 전체 고객을 위한 기능**
- 여러 보험에 대한 RAG 챗봇 기능 
- 답변 근거 문서 하이라이트 기능
</br>

## 🧑🏻‍💻 분석 및 모델링 과정
**1️⃣ RAG 모델 구축**  </br>
+ **문서 저장 기능**: FAISS 벡터 데이터베이스를 통해 저장 </br>
+ **문서 검색 기능**: 의미 검색 방식과 키워드 검색 방식을 결합한 하이브리드 검색 방식 활용  </br>
   +  의미 검색: 보험 문서 임베딩 -> 유클리드 거리와 KNN을 활용하여 유사도  </br> 
   +  키워드 검색: bm25_retriever를 통해 유사한 문서 추출  </br>
+ **검색 성능 평가**: 최적의 검색 비율 설정  </br>
   +  context(청크)별로 200개의 QA 데이터셋 생성  </br>
   +  예측된 답변이 같은 context를 참조하는지 평가  </br>
   +  Hit Rate(적중률), MAP(평균 정밀도), MRR(관련성), NDCG 지표 활용 평가  </br>
  > 최종적으로 의미 검색과 키워드 검색 비율 7:3 선정
   
+ **답변 생성 기능**: LLM + 프롬프트 + RAG 파이프라인 구성  </br>
+ **답변 성능 평가**  </br> 
   +  QA 데이터셋을 활용하여 예측답변와 ANSWER답변 유사도 비교  </br>
   +  Bert-Score, Cross-Encoder 지표 활용 평가  </br>
+ **gradio 연결**: gradio를 통해 데모 UI 구축  </br>
   
**2️⃣ 파인튜닝 모델 구축** </br>
+ **학습 데이터 전환**: pdf를 JSON 파일로 변환 </br>
+ **학습 파라미터 설정**: 제한된 환경 내에서 최적의 값으로 설정 </br>
+ **답변 성능 평가** </br>
   +  QA 데이터셋을 활용하여 예측답변와 ANSWER답변 유사도 비교 </br>
   +  Bert-Score, Cross-Encoder 지표 활용 평가 </br>
   
**3️⃣ RAG모델과 파인튜닝 모델 성능 비교**  </br>
+ **답변 평가**: </br>
+ **시간 절감 효과**: </br>

**4️⃣ 타보험문서에 RAG 챗봇 적용**  </br>
+ 타보험문서(손해, 자동차)를 RAG 모델 챗봇에 적용  </br>
+ 답변 성능 평가: 파인튜닝 모델과도 비교 </br>
</br>

## 데이터 분석 언어 및 라이브러리
![Azure](https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white) &nbsp;![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)&nbsp; ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) &nbsp;<br>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) &nbsp; ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) &nbsp; ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) &nbsp; </br>

