# 🐢 Full-stack AI legal assistant app

## 📖 목차

- [✨ 주요 기능](#-주요-기능)
- [🚀 실행 가이드](#-실행-가이드)
- [⚙️ 설치 방법](#-설치-방법)
- [🎯 사용 방법](#-사용-방법)
- [🔧 설정](#-설정)
- [📁 프로젝트 구조](#-프로젝트-구조)

## ✨ 주요 기능

### 🔍 **계약서 분석**
- 📄 PDF, 이미지, 문서 파일 자동 텍스트 추출
- 🚨 위험 조항 자동 감지 및 경고
- 💯 안전도 점수 제공 (0-100점)
- 🕵️ 임대인 상습 채무불이행자 명단 조회

### 🤖 **AI 법률 상담**
- 💬 실시간 부동산 법률 질의응답
- 📚 주택임대차보호법 기반 정확한 답변
- 🔍 RAG 기술로 근거 있는 답변 제공
- ✅ Groundedness Check로 신뢰성 검증

### 🌐 **다국어 지원**
- 🇰🇷 한국어
- 🇺🇸 영어
- 🇯🇵 일본어  
- 🇨🇳 중국어
- 🇺🇦 우크라이나어
- 🇻🇳 베트남어

### 🎵 **부가 기능**
- 🎧 텍스트 음성 변환 (TTS)
- 📸 분석 결과 PNG 이미지 저장
- 🌍 실시간 번역
- 📊 아름다운 HTML 리포트 생성

## 🚀 실행 가이드

### 1️⃣ **환경 준비**
```bash
# Python 3.8 이상 설치 확인
python --version

# Git으로 프로젝트 다운로드
git clone https://github.com/your-username/shellter.git
cd shellter
```

### 2️⃣ **의존성 설치**
```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 3️⃣ **API 키 설정**
```bash
#.env 파일 생성
#API 키 입력 (선택사항)

UPSTAGE_API_KEY=your_upstage_api_key
DEEPL_API_KEY=your_deepl_api_key  
GOOGLE_API_KEY=your_google_api_key
```

### 4️⃣ **실행**
```bash
python shellter_gradio.py
```

### 5️⃣ **브라우저에서 접속**
```
http://localhost:7860
```

## ⚙️ 설치 방법

### 🐍 **방법 1: Venv (가상환경) 사용시**

#### Windows
```bash
# 1. Python 설치 확인
python --version

# 2. 가상환경 생성
python -m venv venv

# 3. 가상환경 활성화
venv\Scripts\activate

# 4. pip 업그레이드
python -m pip install --upgrade pip

# 5. 패키지 설치
pip install -r requirements.txt

# 6. 실행
python shellter_gradio.py
```

#### macOS/Linux
```bash
# 1. Python 설치 확인
python3 --version

# 2. 가상환경 생성
python3 -m venv venv

# 3. 가상환경 활성화
source venv/bin/activate

# 4. pip 업그레이드
python -m pip install --upgrade pip

# 5. 패키지 설치
pip install -r requirements.txt

# 6. 실행
python shellter_gradio.py
```

### 🐍 **방법 2: Conda 사용시**

#### 모든 운영체제
```bash
# 1. Conda 환경 생성
conda create -n shellter_env python=3.10 -y

# 2. Conda 환경 활성화
conda activate shellter_env

# 3. pip 업그레이드
python -m pip install --upgrade pip

# 4. 패키지 설치
pip install -r requirements.txt

# 5. 실행
python shellter_gradio.py
```

<!-- ### 📥 **수동 설치 (requirements.txt 없을 경우)**
```bash
# 핵심 패키지만 설치
pip install gradio==5.43.1 requests python-dotenv langchain-upstage langchain-core langchain-community chromadb Pillow tqdm
``` -->

## 🎯 사용 방법

### 📄 **계약서 분석하기**

1. **파일 업로드**
   - 지원 형식: PDF, JPG, PNG, DOC, DOCX, HWP, TXT
   - 파일 크기: 최대 50MB

2. **분석 시작**
   - `🔍 분석 시작` 버튼 클릭
   - 자동으로 텍스트 추출 및 분석 진행

3. **결과 확인**
   - 📊 안전도 점수 확인
   - 🚨 위험 조항 경고 확인
   - 🤖 AI 심층 분석 결과 확인

### 💬 **AI 상담하기**

1. **질문 입력**
   - 채팅창에 부동산 관련 질문 입력
   - 예시 질문 제공됨

2. **답변 확인**
   - 실시간으로 AI 답변 생성
   - 법적 근거와 함께 상세 설명

### 🌐 **부가 기능 활용**

#### 번역하기
- 분석 결과나 답변을 6개 언어로 번역
- DeepL API 사용으로 정확한 번역

#### 음성 변환
- 텍스트를 자연스러운 음성으로 변환
- Google TTS API 사용

#### 이미지 저장
- 분석 결과를 PNG 이미지로 저장
- 공유나 인쇄용으로 활용

## 🔧 설정

### 🔑 **API 키 설정**

#### 1. Upstage API (AI 모델)
```bash
# .env 파일에 추가
UPSTAGE_API_KEY=your_upstage_api_key
```

#### 2. DeepL API (번역)
```bash
# .env 파일에 추가
DEEPL_API_KEY=your_deepl_api_key
```

#### 3. Google API (음성)
```bash
# .env 파일에 추가
GOOGLE_API_KEY=your_google_api_key
```

## 📁 프로젝트 구조

```
shellter/
├── 📄 shellter_gradio.py       # 메인 애플리케이션
├── 📁 fonts/                   # 다국어 폰트
│   ├── NotoSans-Regular.ttf
│   ├── NotoSansKR-Regular.ttf
│   └── ...
├── 📁 data/                   # 법률 데이터
│   ├── easylaw_qa_data.json
│   ├── 특약문구_합본.csv
│   └── ...
├── 📁 Image/                  # 이미지 리소스
│   └── logo.png
├── 📄 .env                    # 환경변수 (API KEY)
├── 📄 requirements.txt        # 필요 패키지 모음
└── 📄 README.md               # 이 파일
```

### 🔍 **핵심 파일 설명**

- **`shellter_gradio.py`**: 메인 애플리케이션 파일
- **`fonts/`**: 다국어 텍스트 렌더링용 폰트
- **`data/`**: AI 학습용 법률 데이터
- **`requirements.txt`**: 필요한 Python 패키지 목록