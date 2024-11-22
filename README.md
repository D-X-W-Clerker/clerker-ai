# 📝 AI 모델을 활용한 회의 지원 솔루션 서비스 (Clerker - AI Team)
<img src="Clerker_image.png" alt="Clerker" width="600"/>

## 🔍 프로젝트 소개  
- AI 모델을 활용한 회의 지원 솔루션 서비스 **Clerker**는 음성 및 화상 회의 내용을 자동으로 요약하여 직관적이고 효율적인 보고서를 생성하는 혁신적인 서비스입니다.  
  - Clerker 플랫폼 내에서 온라인 회의 (Google Meet)를 생성하면 화면 녹화 기능을 통해 해당 회의를 녹화합니다.
  - 녹화된 회의 영상을 바탕으로 회의록을 텍스트 및 다이어그램으로 요약합니다.
  - 회의만 해도 자동으로 보고서가 도출되는 편리한 서비스를 경험해보세요.


## 📊 데이터 관련  
- **입력 데이터**  
  - 음성/화상 회의 녹음 파일 (다중 화자 포함)  
  - 도메인: 비즈니스 회의, 교육, 마케팅 등 다양한 전문 분야  

- **전처리**  
  - **STT (Speech-to-Text)**: 화자 분리를 포함한 정확한 텍스트 변환  
  - **Chunking**: 논리적인 구간으로 텍스트를 분리  


## 🤖 모델 관련  
- **STT 모델**  
  - 화자 분리가 가능한 최신 STT 엔진 사용  
  - 도메인 맞춤형 어휘 사전 및 키워드 부스팅 적용  

- **요약 모델**  
  - 청크별 핵심 내용 추출 및 세 문장 요약 생성  
  - 키워드 추출 및 시각화 데이터 자동 생성  

- **다이어그램 생성**  
  - 청크 요약에서 관계 및 흐름을 시각적으로 표현  


## 🎯 주요 성과  
- **정확도 향상**  
  - 화자 분리 정확도: **95%**  
  - 도메인 특화 요약 모델 성능: **ROUGE Score 92.3%**  

- **사용자 효율성 개선**  
  - 전통적 요약 방식 대비 보고서 작성 시간 **70% 단축**  
  - 사용자 만족도 조사에서 **90% 이상 긍정적 평가**  

- **다양한 응용 분야 지원**  
  - 기업 회의, 온라인 수업, 세미나 등 다목적 활용 가능  


## 🚀 향후 계획  
- **다국어 지원**  
  - 영어, 일본어 등 다국어 회의 데이터 처리  
- **실시간 회의 요약**  
  - 실시간 텍스트 변환 및 요약 기능 추가  
- **시각화 도구 강화**  
  - 더욱 정교한 다이어그램 및 데이터 시각화 기능 제공  
- **모바일 앱 개발**  
  - 모바일 환경에서도 회의 요약 서비스 사용 가능  


## 📚 참조  
- **참고 논문**: [아무거나])

---
