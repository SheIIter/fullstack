import gradio as gr
import os
import json
import csv
import mimetypes
import requests
import re
import base64
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수
try:
    load_dotenv()
    DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
except:
    DEEPL_API_KEY = None
    GOOGLE_API_KEY = None
    UPSTAGE_API_KEY = None

# API
TTS_API_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}" if GOOGLE_API_KEY else None
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# Upstage Chat API 
try:
    from langchain_upstage import ChatUpstage, UpstageDocumentParseLoader
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    UPSTAGE_AVAILABLE = True
except ImportError:
    UPSTAGE_AVAILABLE = False

# 🌍 번역 함수 
def deepl_translate_text(text, target_lang):
    """DeepL API를 사용한 텍스트 번역"""
    if not DEEPL_API_KEY:
        # API 키가 없으면 간단한 메시지 반환
        lang_names = {
            "EN": "영어", "JA": "일본어", "ZH": "중국어", 
            "UK": "우크라이나어", "VI": "베트남어"
        }
        return f"[{lang_names.get(target_lang, target_lang)} 번역 기능]\n\nDeepL API 키가 설정되지 않아 실제 번역은 불가능합니다.\n\n원본 텍스트:\n{text[:500]}..."
        
    try:
        headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
        data = {"text": [text], "target_lang": target_lang}
        response = requests.post(DEEPL_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        return response.json()["translations"][0]["text"]
    except Exception as e:
        return f"번역 오류: {e}\n\n원본 텍스트:\n{text[:500]}..."

# 🔊 TTS 함수 
def split_text_for_tts(text, max_bytes=4500):
    """긴 텍스트를 Google TTS 5000바이트 제한에 맞게 안전하게 분할"""
    if len(text.encode('utf-8')) <= max_bytes:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?다])\s+', text)
    
    for sentence in sentences:
        test_chunk = current_chunk + sentence + " "
        
        if len(test_chunk.encode('utf-8')) <= max_bytes:
            current_chunk = test_chunk
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # 단일 문장이 너무 긴 경우 강제로 자르기
            if len(sentence.encode('utf-8')) > max_bytes:
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    test_word_chunk = temp_chunk + word + " "
                    if len(test_word_chunk.encode('utf-8')) <= max_bytes:
                        temp_chunk = test_word_chunk
                    else:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                
                current_chunk = temp_chunk
            else:
                current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def google_text_to_speech(text, lang_code="KO"):
    """Google TTS API로 음성 파일 생성 (긴 텍스트 자동 분할 처리)"""
    if not GOOGLE_API_KEY:
        return None, "Google API 키가 설정되지 않아 음성 생성이 불가능합니다."
    
    # 특수문자 제거
    text = re.sub(r"[^\w\s가-힣]", "", text)
    
    # 텍스트 길이 체크 및 분할
    text_chunks = split_text_for_tts(text)
    
    # 음성 설정
    voice_map = {
        "KO": {"languageCode": "ko-KR", "name": "ko-KR-Wavenet-A"},
        "EN": {"languageCode": "en-US", "name": "en-US-Wavenet-F"},
        "JA": {"languageCode": "ja-JP", "name": "ja-JP-Wavenet-A"},
        "ZH": {"languageCode": "cmn-CN", "name": "cmn-CN-Wavenet-A"},
    }
    
    if lang_code not in voice_map:
        return None, f"지원하지 않는 언어: {lang_code}"
    
    try:
        # 단일 청크인 경우
        if len(text_chunks) == 1:
            request_body = {
                "input": {"text": text_chunks[0]},
                "voice": voice_map[lang_code],
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": 0.85,
                    "pitch": -5
                }
            }
            
            response = requests.post(TTS_API_URL, data=json.dumps(request_body), timeout=30)
            
            if response.status_code == 200:
                audio_content = base64.b64decode(response.json()['audioContent'])
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_content)
                    return tmp_file.name, "음성 생성 완료!"
            else:
                return None, f"TTS 오류: {response.text}"
        
        # 여러 청크인 경우 첫 번째 청크만 처리 (Gradio 제한)
        else:
            first_chunk = text_chunks[0]
            request_body = {
                "input": {"text": first_chunk},
                "voice": voice_map[lang_code],
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": 0.85,
                    "pitch": -5
                }
            }
            
            response = requests.post(TTS_API_URL, data=json.dumps(request_body), timeout=30)
            
            if response.status_code == 200:
                audio_content = base64.b64decode(response.json()['audioContent'])
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_content)
                    return tmp_file.name, f"음성 생성 완료! (긴 텍스트로 인해 {len(text_chunks)}개 중 첫 번째 부분만 재생)"
            else:
                return None, f"TTS 오류: {response.text}"
            
    except Exception as e:
        return None, f"TTS 오류: {e}"

# 📄 파일 처리 함수 (간단한 텍스트 추출)
def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """파일에서 텍스트 추출 (간단 버전)"""
    if not file_path or not os.path.exists(file_path):
        return "", "파일을 찾을 수 없습니다"
    
    file_extension = Path(file_path).suffix.lower()
    
    # 텍스트 파일인 경우
    if file_extension in ['.txt', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), "성공"
        except:
            return "", "텍스트 파일 읽기 실패"
    
    # 이미지나 PDF인 경우 (Upstage 라이브러리 사용 시도)
    try:
        if UPSTAGE_AVAILABLE:
            pages = UpstageDocumentParseLoader(file_path, ocr="force").load()
            extracted_text = "\n\n".join([p.page_content for p in pages])
            if not extracted_text.strip():
                return "", "텍스트 추출 실패"
            return extracted_text, "성공"
        else:
            # Upstage 라이브러리가 없는 경우 샘플 텍스트 반환
            sample_text = f"""
이는 {file_extension} 파일에서 추출된 샘플 텍스트입니다.

【 부동산 임대차 계약서 】

임대인: 홍길동 (주소: 서울시 강남구 테헤란로 123)
임차인: 김철수 (주소: 서울시 서초구 서초대로 456)

1. 임대차 목적물
   - 주소: 서울시 강남구 역삼동 123-45 아파트 101동 1004호
   - 면적: 84.5㎡ (전용면적 59.5㎡)

2. 임대차 조건
   - 보증금: 2억원
   - 월세: 없음 (전세계약)
   - 계약기간: 2024년 3월 1일 ~ 2026년 2월 28일

3. 특약사항
   - 계약 만료 시 보증금은 즉시 반환한다.
   - 임차인은 계약 종료 1개월 전까지 갱신 의사를 통지한다.

실제 텍스트 추출을 위해서는 Upstage API 키가 필요합니다.
"""
            return sample_text, "샘플 텍스트 (실제 추출을 위해서는 Upstage API 키 필요)"
    except Exception as e:
        return f"파일 처리 중 오류: {str(e)}", "오류"

# 📊 텍스트 품질 분석 (외부 출력X)
def analyze_text_quality(text: str) -> dict:
    """텍스트 품질 분석 (UI에 표시하지 않음)"""
    if not text:
        return {"quality_level": "EMPTY", "confidence": 0, "details": {}}
    
    word_count = len(text.split())
    char_count = len(text)
    korean_ratio = sum(1 for c in text if '가' <= c <= '힣') / len(text) if len(text) > 0 else 0
    essential_keywords = ['임대차', '계약', '보증금', '임대인', '임차인', '월세', '전세']
    found_keywords = sum(1 for k in essential_keywords if k in text)
    
    score = (
        (word_count >= 100) * 30 + 
        (korean_ratio >= 0.3) * 25 + 
        (found_keywords >= 3) * 25 + 
        (len(text) >= 500) * 20
    )
    
    level = "HIGH" if score >= 80 else "MEDIUM" if score >= 50 else "LOW"
    
    return {
        "quality_level": level,
        "confidence": int(score),
        "details": {
            "총 단어 수": f"{word_count:,}개",
            "총 문자 수": f"{char_count:,}개",
            "한글 비율": f"{korean_ratio:.1%}",
            "필수 키워드": f"{found_keywords}/{len(essential_keywords)}개 발견"
        }
    }

# 🏠 계약서 분석 엔진 (간단)
def extract_landlord_name(contract_text: str) -> str:
    """임대인 이름 추출 (간단 버전)"""
    # 정규표현식으로 임대인 이름 찾기
    patterns = [
        r'임대인[:\s]*([가-힣]{2,4})',
        r'집주인[:\s]*([가-힣]{2,4})',
        r'소유자[:\s]*([가-힣]{2,4})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, contract_text)
        if match:
            return match.group(1)
    
    return "추출 실패"

def perform_rule_based_analysis(contract_text: str) -> dict:
    """규칙 기반 계약서 분석"""
    alerts, safety_score = [], 100
    
    try:
        categories = {
            "보증금_반환": {"keywords": ["보증금", "반환", "즉시", "계약종료"], "risk": "CRITICAL"},
            "권리관계_유지": {"keywords": ["권리관계", "익일", "근저당", "대항력"], "risk": "CRITICAL"},
            "전세자금대출": {"keywords": ["대출", "불가", "무효", "전세자금"], "risk": "WARNING"},
            "수선_의무": {"keywords": ["수선", "하자", "파손", "수리"], "risk": "ADVISORY"},
            "특약사항": {"keywords": ["특약", "기타사항", "추가조건"], "risk": "ADVISORY"}
        }
        
        for cat_name, info in categories.items():
            display_name = cat_name.replace('_', ' ')
            keyword_count = sum(1 for kw in info['keywords'] if kw in contract_text)
            keyword_ratio = keyword_count / len(info['keywords'])
            
            if keyword_ratio < 0.3:
                if info['risk'] == "CRITICAL":
                    safety_score -= 40
                    alerts.append(f"🚨 [치명적!] {display_name}: 관련 조항이 누락되어 위험할 수 있습니다!")
                elif info['risk'] == "WARNING":
                    safety_score -= 20
                    alerts.append(f"⚠️ [위험] {display_name}: 관련 조항이 미비합니다.")
                else:
                    safety_score -= 10
                    alerts.append(f"💡 [권장] {display_name}: 관련 조항 추가를 권장합니다.")
            else:
                alerts.append(f"✅ [{display_name}] 관련 조항이 확인되었습니다. ({keyword_count}/{len(info['keywords'])}개 키워드)")

        safety_score = max(0, safety_score)
        
        # 임대인 이름 추출
        landlord_name = extract_landlord_name(contract_text)
        if landlord_name != "추출 실패":
            alerts.append(f"✅ [임대인 정보] 임대인 이름: '{landlord_name}'")
        else:
            alerts.append("⚠️ [임대인 정보] 임대인 이름을 자동으로 찾지 못했습니다.")
            
    except Exception as e:
        alerts.append(f"⚠️ 분석 중 오류 발생: {str(e)}")
        safety_score = -1
    
    return {"alerts": alerts, "safety_score": safety_score}

def perform_ai_analysis(contract_text: str) -> dict:
    """AI 기반 계약서 분석 (실제 API 사용)"""
    try:
        if UPSTAGE_AVAILABLE and UPSTAGE_API_KEY:
            prompt = ChatPromptTemplate.from_template(
                """한국 부동산 법률 전문가로서 다음 [계약서]를 분석해주세요.

[계약서]
{contract}

다음 사항들을 중점적으로 분석해주세요:
1. 임차인에게 불리한 조항
2. 누락된 중요 조항
3. 개선 방안
4. 주의사항

친절하고 이해하기 쉽게 설명해주세요."""
            )
            
            chain = prompt | ChatUpstage(model="solar-pro") | StrOutputParser()
            analysis_result = chain.invoke({"contract": contract_text})
            return {"analysis": analysis_result}
        
        else:
            # API 키가 없는 경우 간단한 분석 제공
            analysis_result = f"""
**🔍 계약서 간단 분석 결과**

**📋 계약서 개요**
- 계약서 길이: {len(contract_text):,}자
- 주요 키워드 확인: {'임대차' in contract_text}, {'보증금' in contract_text}, {'계약' in contract_text}

**⚠️ 주의사항**
1. **보증금 반환 조항**: {'보증금 반환' in contract_text if '보증금' in contract_text else '확인 필요'}
2. **계약 해지 조건**: {'해지' in contract_text if '해지' in contract_text else '명시되지 않음'}
3. **특약사항**: {'특약' in contract_text if '특약' in contract_text else '확인 필요'}

**💡 권장사항**
- 전문 법무사와 상담을 받으시기 바랍니다
- 계약서의 모든 조항을 꼼꼼히 검토하세요
- 불분명한 조항은 반드시 명확히 하고 계약하세요

*실제 AI 분석을 위해서는 Upstage API 키가 필요합니다.*
"""
            return {"analysis": analysis_result}
            
    except Exception as e:
        return {"analysis": f"분석 중 오류 발생: {str(e)}"}

def generate_report(file_name, rule_analysis, ai_analysis):
    """종합 분석 리포트 생성 (간소화된 버전)"""
    score = rule_analysis['safety_score']
    if score >= 80:
        safety_grade = f"✅ 매우 안전 ({score}점)"
    elif score >= 50:
        safety_grade = f"⚠️ 보통 ({score}점)"
    elif score >= 0:
        safety_grade = f"🚨 위험! ({score}점)"
    else:
        safety_grade = "⚠️ 점수 계산 오류"
    
    alerts_text = "\n".join([f"- {alert}" for alert in rule_analysis['alerts']])
    
    return f"""# 🏠 AI 부동산 계약서 종합 분석 리포트

## 📋 분석 대상
**파일**: 📄 {file_name}

## 🕵️ 계약서 안전도 검사
**종합 안전도**: {safety_grade}

{alerts_text}

## 🧠 AI 심층 분석
{ai_analysis['analysis']}

---
***본 분석은 참고용이며 법적 효력이 없습니다. 중요한 결정 전 반드시 전문가와 상담하시기 바랍니다.***
"""

# 🎯 Gradio module
def analyze_contract(file):
    """계약서 분석 메인 함수"""
    if file is None:
        return "❌ 파일을 업로드해주세요.", ""
    
    try:
        print(f"파일 분석 시작: {file.name}")
        
        # 파일에서 텍스트 추출
        text, status = extract_text_from_file(file.name)
        if not text:
            return f"❌ 텍스트 추출 실패: {status}", ""
        
        print(f"텍스트 추출 성공: {len(text)}자")
        
        # 텍스트 품질 분석 (내부, UI에 표시하지 않음)
        quality = analyze_text_quality(text)
        print(f"품질 분석 완료: {quality['quality_level']}")
        
        # 규칙 기반 분석
        rule_analysis = perform_rule_based_analysis(text)
        print("규칙 기반 분석 완료")
        
        # AI 분석
        ai_analysis = perform_ai_analysis(text)
        print("AI 분석 완료")
        
        # 리포트 생성 (품질 정보 제외)
        report = generate_report(
            os.path.basename(file.name),
            rule_analysis,
            ai_analysis
        )
        
        return report, text
        
    except Exception as e:
        error_msg = f"❌ 분석 중 오류 발생: {str(e)}"
        print(error_msg)
        return error_msg, ""

def chat_with_ai(message, history):
    """AI 법률 비서 채팅 (실제 API 사용)"""
    if not message.strip():
        return history, ""
    
    try:
        if UPSTAGE_AVAILABLE and UPSTAGE_API_KEY:
            # 실제 AI API 사용
            prompt = ChatPromptTemplate.from_template(
                """당신은 한국 부동산 법률 전문가입니다. 사용자의 질문에 친절하고 정확하게 답변해주세요.

사용자 질문: {question}

답변 시 다음 사항을 고려해주세요:
- 한국 부동산 관련 법률과 실무에 기반한 정확한 정보 제공
- 이해하기 쉬운 용어와 구체적인 예시 포함
- 필요시 주의사항과 권장사항도 함께 안내
- 법적 효력이 없음을 명시하고 전문가 상담 권유"""
            )
            
            chain = prompt | ChatUpstage(model="solar-pro") | StrOutputParser()
            response = chain.invoke({"question": message})
            
        else:
            # API 키가 없는 경우 간단한 규칙 기반 응답
            responses = {
                "보증금": "보증금 관련 질문이시군요! 보증금은 계약 종료 시 반드시 반환되어야 하며, 지연 시 이자도 지급해야 합니다.",
                "전세": "전세 계약에서는 확정일자를 받고, 전입신고를 하여 대항력을 확보하는 것이 중요합니다.",
                "월세": "월세 계약에서는 임대료 인상 한도(5%)와 계약갱신청구권을 확인하세요.",
                "계약서": "계약서에는 당사자 정보, 목적물 정보, 임대조건, 특약사항이 명확히 기재되어야 합니다.",
                "임대인": "임대인의 신원을 확인하고, 등기부등본을 통해 실제 소유자인지 확인하는 것이 중요합니다."
            }
            
            response = "안녕하세요! 부동산 관련 질문에 답변드리겠습니다.\n\n"
            
            found_keyword = False
            for keyword, answer in responses.items():
                if keyword in message:
                    response += f"**{keyword} 관련 답변:**\n{answer}\n\n"
                    found_keyword = True
            
            if not found_keyword:
                response += """다음과 같은 부동산 관련 주제로 질문해주세요:
- 보증금 반환
- 전세 vs 월세
- 계약서 작성
- 임대인 확인
- 계약 갱신
- 특약 조항"""
            
            response += "\n*실제 AI 상담을 위해서는 Upstage API 키가 필요합니다.*"
        
        response += "\n\n💡 **전문적인 법률 상담이 필요한 경우 변호사나 법무사와 상담받으시기 바랍니다.**"
        
        history.append((message, response))
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ 답변 생성 중 오류: {str(e)}"
        history.append((message, error_msg))
        return history, ""

def translate_text(text, target_lang):
    """텍스트 번역"""
    if not text.strip():
        return "번역할 텍스트가 없습니다."
    
    return deepl_translate_text(text, target_lang)

def generate_speech(text, language):
    """텍스트를 음성으로 변환"""
    if not text.strip():
        return None, "음성으로 변환할 텍스트가 없습니다."
    
    lang_map = {
        "한국어": "KO",
        "영어": "EN",
        "일본어": "JA",
        "중국어": "ZH"
    }
    
    if language not in lang_map:
        return None, "지원하지 않는 언어입니다."
    
    return google_text_to_speech(text, lang_map[language])

# 🎨 Gradio 인터페이스 구성
def create_interface():
    """Gradio 인터페이스 생성"""
    
    # CSS 스타일링 (초록색 배경으로 변경)
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    """
    
    with gr.Blocks(
        title="AI 부동산 법률 비서",
        css=css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo"
        )
    ) as interface:
        
        # 헤더
        with gr.Row():
            gr.HTML("""
            <div class="main-header">
                <h1>🐢 쉘터 AI 법률 비서 🗯️</h1>
                <p>부동산 계약서를 안전하게 분석하고, 법률 전문가의 조언을 받아보세요!</p>
            </div>
            """)
        
        with gr.Row(equal_height=True):
            # 왼쪽 컬럼: 파일 분석
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("## 📋 계약서 분석")
                
                # 파일 업로드 (항상 표시)
                file_input = gr.File(
                    label="📁 계약서 파일을 업로드하세요",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx", ".hwp", ".txt"],
                    type="filepath"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 초기화", variant="secondary")
                
                # 분석 결과 (항상 표시)
                analysis_output = gr.Markdown(
                    value="📤 파일을 업로드하고 **분석 시작** 버튼을 클릭하세요.",
                    line_breaks=True
                )
                
                # 분석 결과 번역 및 음성 기능
                gr.Markdown("### 🌍 분석 결과 번역 & 음성")
                with gr.Row():
                    analysis_translate_lang = gr.Dropdown(
                        choices=["원본", "EN", "JA", "ZH", "UK", "VI"],
                        label="번역할 언어",
                        value="원본"
                    )
                    analysis_speech_lang = gr.Dropdown(
                        choices=["한국어", "영어", "일본어", "중국어"],
                        label="음성 언어",
                        value="한국어"
                    )
                
                with gr.Row():
                    analysis_translate_btn = gr.Button("🌍 번역하기")
                    analysis_speech_btn = gr.Button("🔊 음성 생성")
                
                analysis_translation_output = gr.Textbox(
                    label="번역된 분석 결과",
                    lines=5,
                    max_lines=15,
                    show_copy_button=True
                )
                
                analysis_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                analysis_audio_output = gr.Audio(label="분석 결과 음성", type="filepath")
            
            # 오른쪽 컬럼: AI 채팅
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("## 🤖 AI 법률 비서")
                
                chatbot = gr.Chatbot(
                    value=[("안녕하세요!", "안녕하세요! 부동산 관련 질문이 있으시면 언제든 물어보세요. 보증금, 전세, 월세, 계약서 작성 등 다양한 주제로 도움드릴 수 있습니다!")],
                    height=400,
                    show_label=False,
                    container=False,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="부동산 관련 질문을 입력하세요... (예: 보증금 반환 조건은?)",
                        scale=4,
                        show_label=False,
                        container=False
                    )
                    send_btn = gr.Button("📤", scale=1, variant="primary")
                
                # 채팅 답변 번역 및 음성 기능
                gr.Markdown("### 🌍 채팅 답변 번역 & 음성")
                with gr.Row():
                    chat_translate_lang = gr.Dropdown(
                        choices=["원본", "EN", "JA", "ZH", "UK", "VI"],
                        label="번역할 언어",
                        value="원본"
                    )
                    chat_speech_lang = gr.Dropdown(
                        choices=["한국어", "영어", "일본어", "중국어"],
                        label="음성 언어",
                        value="한국어"
                    )
                
                with gr.Row():
                    chat_translate_btn = gr.Button("🌍 최근 답변 번역")
                    chat_speech_btn = gr.Button("🔊 최근 답변 음성")
                
                chat_translation_output = gr.Textbox(
                    label="번역된 채팅 답변",
                    lines=3,
                    max_lines=10,
                    show_copy_button=True
                )
                
                chat_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                chat_audio_output = gr.Audio(label="채팅 답변 음성", type="filepath")
                
                gr.Examples(
                    examples=[
                        "전세 계약 시 주의사항은?",
                        "보증금 반환을 위한 조건은?",
                        "월세 계약과 전세 계약의 차이점은?",
                        "계약서에 반드시 포함되어야 할 내용은?",
                        "임대인 신원 확인 방법은?"
                    ],
                    inputs=msg_input,
                    label="💡 질문 예시"
                )
        
        # 🔥 상태 관리 (수정된 부분)
        extracted_text = gr.State("")
        analysis_report = gr.State("")  # 분석 리포트 저장용
        last_chat_response = gr.State("")
        
        # 🔥 이벤트 핸들러 (수정된 부분)
        def clear_all():
            return (None, "📤 파일을 업로드하고 **분석 시작** 버튼을 클릭하세요.", 
                   "", "", None, "", "", None, "", "", "")
        
        def analyze_and_store_report(file):
            """계약서 분석하고 리포트 저장"""
            if file is None:
                return "❌ 파일을 업로드해주세요.", "", ""
            
            # 기존 분석 함수 호출
            report, text = analyze_contract(file)
            return report, text, report  # 리포트를 별도로 저장
        
        def store_chat_response(message, history):
            """채팅 응답을 저장하고 반환"""
            if not message.strip():
                return history, "", ""
            
            # 새로운 응답 생성
            new_history, _ = chat_with_ai(message, history)
            
            # 마지막 AI 응답만 추출 (사용자 입력 제외)
            if new_history and len(new_history) > 0:
                last_response = new_history[-1][1]  # AI의 답변 부분
            else:
                last_response = ""
                
            return new_history, "", last_response
        
        # 파일 분석 (리포트 저장 포함)
        analyze_btn.click(
            fn=analyze_and_store_report,
            inputs=[file_input],
            outputs=[analysis_output, extracted_text, analysis_report]
        )
        
        # 초기화
        clear_btn.click(
            fn=clear_all,
            outputs=[file_input, analysis_output, analysis_translation_output, 
                    analysis_speech_status, analysis_audio_output,
                    chat_translation_output, chat_speech_status, chat_audio_output, 
                    msg_input, extracted_text, analysis_report]
        )
        
        # 🔥 분석 결과(리포트) 번역 - 수정된 부분
        analysis_translate_btn.click(
            fn=lambda report, lang: translate_text(report, lang) if lang != "원본" and report else ("번역할 분석 결과가 없습니다." if not report else report),
            inputs=[analysis_report, analysis_translate_lang],
            outputs=[analysis_translation_output]
        )
        
        # 🔥 분석 결과(리포트) 음성 생성 - 수정된 부분
        def generate_analysis_speech(report, lang, translate_lang):
            if not report.strip():
                return None, "분석 결과가 없습니다."
            
            # 번역이 필요한 경우
            if translate_lang != "원본":
                translated = translate_text(report, translate_lang)
                speech_text = translated
            else:
                speech_text = report
            
            return generate_speech(speech_text, lang)
        
        analysis_speech_btn.click(
            fn=generate_analysis_speech,
            inputs=[analysis_report, analysis_speech_lang, analysis_translate_lang],
            outputs=[analysis_audio_output, analysis_speech_status]
        )
        
        # 채팅 (응답 저장 포함)
        send_btn.click(
            fn=store_chat_response,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, last_chat_response]
        )
        
        msg_input.submit(
            fn=store_chat_response,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, last_chat_response]
        )
        
        # 🔥 채팅 답변 번역 - 수정된 부분
        chat_translate_btn.click(
            fn=lambda response, lang: translate_text(response, lang) if lang != "원본" and response.strip() else ("번역할 답변이 없습니다." if not response.strip() else response),
            inputs=[last_chat_response, chat_translate_lang],
            outputs=[chat_translation_output]
        )
        
        # 🔥 채팅 답변 음성 생성 - 수정된 부분
        def generate_chat_speech(response, lang, translate_lang):
            if not response.strip():
                return None, "채팅 답변이 없습니다."
            
            # 번역이 필요한 경우
            if translate_lang != "원본":
                translated = translate_text(response, translate_lang)
                speech_text = translated
            else:
                speech_text = response
            
            return generate_speech(speech_text, lang)
        
        chat_speech_btn.click(
            fn=generate_chat_speech,
            inputs=[last_chat_response, chat_speech_lang, chat_translate_lang],
            outputs=[chat_audio_output, chat_speech_status]
        )
    
    return interface

# 🚀 메인 실행부
def main():
    """메인 실행 함수"""
    print("🚀 AI 부동산 법률 비서를 시작합니다...")
    print("📋 기능 목록:")
    print("  - 📄 계약서 파일 분석 (PDF, 이미지, 문서)")
    print("  - 🤖 AI 법률 상담 챗봇 (실제 API 연동)")
    print("  - 🌍 분석결과 & 채팅답변 번역 (DeepL)")
    print("  - 🔊 분석결과 & 채팅답변 음성변환 (Google TTS)")
    print("  - 📊 계약서 위험도 분석")
    print()
    
    try:
        # 인터페이스 생성 및 실행
        app = create_interface()
        
        print("✅ 인터페이스 생성 완료!")
        print("🌐 웹 브라우저에서 자동으로 열립니다...")
        print("🔗 수동 접속: http://localhost:7860")
        print()
        print("💡 사용 팁:")
        print("  1. 왼쪽에서 계약서 파일을 업로드하고 분석하세요")
        print("  2. 분석 결과를 번역하고 음성으로 들어보세요")
        print("  3. 오른쪽에서 AI와 채팅할 수 있습니다")
        print("  4. 채팅 답변도 번역하고 음성으로 들어보세요")
        print("  5. Ctrl+C로 종료할 수 있습니다")
        print()
        
        # 서버 시작
        app.launch(
            server_name="0.0.0.0",  # 모든 IP에서 접속 가능
            server_port=7860,       # 포트 설정
            share=False,            # 공개 링크 생성 안함 (로컬 테스트용)
            debug=True,             # 디버그 모드
            show_error=True,        # 오류 표시
            quiet=False,            # 로그 출력
            inbrowser=True          # 자동으로 브라우저 열기
        )
        
    except KeyboardInterrupt:
        print("\n👋 사용자가 서버를 중단했습니다.")
        print("🛑 서버를 안전하게 종료합니다...")
        
    except Exception as e:
        print(f"❌ 서버 시작 중 오류 발생: {e}")
        print("🔧 다음을 확인해보세요:")
        print("  1. 필요한 패키지가 모두 설치되었는지")
        print("  2. 포트 7860이 사용 중인지")
        print("  3. 방화벽 설정이 차단하고 있는지")

if __name__ == "__main__":
    main()