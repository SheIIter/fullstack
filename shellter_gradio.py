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
import io
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datetime import datetime
from tqdm import tqdm 

from langchain_upstage import (
    UpstageDocumentParseLoader,
    UpstageEmbeddings,
    ChatUpstage,
    UpstageGroundednessCheck,
)
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Groundedness 체크용 컨텍스트 직렬화 유틸
def docs_to_text(docs):
    try:
        return "\n\n---\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    except Exception:
        return str(docs)

# Groundedness 컨텍스트 빌더: 검색 컨텍스트 + 계약/질문 원문 결합
def build_grounded_context_for_contract(contract_text: str) -> str:
    try:
        retrieved = RETRIEVER.invoke(contract_text) if RETRIEVER else []
    except Exception:
        retrieved = []
    retrieved_text = docs_to_text(retrieved)
    return f"[참고 자료]\n{retrieved_text}\n\n[계약서]\n{contract_text}"

def build_grounded_context_for_question(question_text: str) -> str:
    try:
        retrieved = RETRIEVER.invoke(question_text) if RETRIEVER else []
    except Exception:
        retrieved = []
    retrieved_text = docs_to_text(retrieved)
    return f"[참고 자료]\n{retrieved_text}\n\n[질문]\n{question_text}"

# 선택 의존성 (HTML -> PNG 변환용)
HTML2IMAGE_AVAILABLE = False
try:
    from html2image import Html2Image
    HTML2IMAGE_AVAILABLE = True
except Exception:
    HTML2IMAGE_AVAILABLE = False

# 선택 의존성 (Markdown -> HTML 변환) - FIXED
MARKDOWN_AVAILABLE = False
try:
    import markdown2
    MARKDOWN_AVAILABLE = True
except Exception:
    try:
        import markdown
        MARKDOWN_AVAILABLE = True
    except Exception:
        MARKDOWN_AVAILABLE = False

# 환경 변수
try:
    if load_dotenv():
        print("🔑 API 키를 성공적으로 불러왔습니다.")
    DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
except:
    DEEPL_API_KEY = None
    GOOGLE_API_KEY = None
    UPSTAGE_API_KEY = None

# API 엔드포인트
TTS_API_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}" if GOOGLE_API_KEY else None
STT_API_URL = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_API_KEY}" if GOOGLE_API_KEY else None
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# 데이터 경로 설정
EASYLAW_QA_PATH = "./data/easylaw_qa_data.json"
SPECIAL_CLAUSES_PATH = "./data/특약문구 합본_utf8bom.csv"
LAW_PARSED_PATH = "./data/주택임대차보호법(법률)(제19356호)_parsed.json"
DEFAULTER_LIST_PATH = "./data/상습채무불이행자.CSV"
CHROMA_DB_PATH = "./chroma_db_real_estate_gradio"

# 다국어 폰트 자동 다운로드 로직, TTF만으로 정확한 링크로 수정 진행.
FONTS_DIR = Path("./fonts")
FONT_URLS = {
    # Noto Sans (라틴/키릴/기본영문)
    "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "NotoSans-Bold.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    # Noto Sans KR (한국어)
    "NotoSansKR-Regular.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf",
    "NotoSansKR-Bold.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Bold.otf",
    # Noto Sans JP (일본어) - TTF 파일로 수정
    "NotoSansJP-Regular.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
    "NotoSansJP-Bold.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Bold.otf",
    # Noto Sans SC (중국어 간체) - TTF 파일로 수정
    "NotoSansSC-Regular.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "NotoSansSC-Bold.ttf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
    # Noto Color Emoji (이모지 지원)
    "NotoColorEmoji-Regular.ttf": "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf",
    # Noto Sans (우크라이나어 키릴 문자 지원) - 추가
    "NotoSans-{style}.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
}

def setup_fonts():
    """
    필요한 다국어 폰트를 ./fonts 폴더에 자동으로 다운로드하고, OTF 파일을 TTF로 변환합니다.
    """
    print("🖋️ 다국어 폰트 설정을 시작합니다...")
    FONTS_DIR.mkdir(exist_ok=True)

    for font_name, url in FONT_URLS.items():
        font_path = FONTS_DIR / font_name
        if font_path.exists():
            print(f"  - '{font_name}' 폰트가 이미 존재합니다. (건너뛰기)")
            continue

        try:
            print(f"  - '{font_name}' 폰트 다운로드 중... ({url})")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(font_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=f"    {font_name}"
            ) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
            
            if total_size != 0 and pbar.n != total_size:
                 raise Exception("다운로드 중 오류 발생")

            print(f"  🎉 '{font_name}' 폰트 다운로드 완료!")

        except Exception as e:
            print(f"  ❌ '{font_name}' 폰트 다운로드 실패: {e}")
            if font_path.exists():
                font_path.unlink() # 실패 시 불완전한 파일 삭제
    
    # OTF 파일을 TTF로 변환 시도
    print("  - OTF 파일을 TTF로 변환 시도 중...")
    try:
        from fontTools.ttLib import TTFont
        from fontTools.ttx import makeOutputFileName
        
        for font_name in FONT_URLS.keys():
            if font_name.endswith('.otf'):
                otf_path = FONTS_DIR / font_name
                ttf_name = font_name.replace('.otf', '.ttf')
                ttf_path = FONTS_DIR / ttf_name
                
                if otf_path.exists() and not ttf_path.exists():
                    try:
                        print(f"    - '{font_name}' → '{ttf_name}' 변환 중...")
                        font = TTFont(str(otf_path))
                        font.save(str(ttf_path))
                        print(f"    - '{ttf_name}' 변환 완료!")
                    except Exception as e:
                        print(f"    - '{font_name}' 변환 실패: {e}")
    except ImportError:
        print("    - fontTools가 설치되지 않아 OTF→TTF 변환을 건너뜁니다.")
        print("    - pip install fonttools로 설치 가능합니다.")
    
    print("✅ 모든 폰트 설정이 완료되었습니다.")


def build_ai_brain_if_needed():
    """AI의 지식 베이스(Vector DB)를 구축합니다. 이미 존재하면 건너뜁니다."""
    if os.path.exists(CHROMA_DB_PATH):
        print(f"✅ Vector DB가 이미 존재합니다. ({CHROMA_DB_PATH})")
        return
    print(f"✨ AI의 지식 베이스(Vector DB)를 새로 구축합니다...")
    all_documents = []
    
    # EasyLaw Q&A 데이터 로드
    try:
        with open(EASYLAW_QA_PATH, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                all_documents.append(Document(
                    page_content=f"사례 질문: {item['question']}\n사례 답변: {item['answer']}",
                    metadata={"source": "easylaw_qa"}
                ))
        print(f"  - EasyLaw QA 데이터 로드 완료 ({len(all_documents)}개 문서)")
    except FileNotFoundError:
        print(f"  [경고] '{EASYLAW_QA_PATH}' 파일을 찾을 수 없습니다.")
    
    # 법률 조문 데이터 로드
    try:
        with open(LAW_PARSED_PATH, 'r', encoding='utf-8') as f:
            law_text = json.load(f).get("text", "")
            all_documents.append(Document(
                page_content=law_text,
                metadata={"source": "housing_lease_law"}
            ))
        print(f"  - 주택임대차보호법 데이터 로드 완료")
    except FileNotFoundError:
        print(f"  [경고] '{LAW_PARSED_PATH}' 파일을 찾을 수 없습니다.")
    
    # 특약 조항 데이터 로드
    try:
        clauses_count = 0
        with open(SPECIAL_CLAUSES_PATH, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if clause_content := row.get('특약내용', '').strip():
                    all_documents.append(Document(
                        page_content=f"권장 특약 조항 예시: {clause_content}",
                        metadata={"source": "special_clauses"}
                    ))
                    clauses_count += 1
        print(f"  - 특약 조항 데이터 로드 완료 ({clauses_count}개)")
    except FileNotFoundError:
        print(f"  [경고] '{SPECIAL_CLAUSES_PATH}' 파일을 찾을 수 없습니다.")
    
    if not all_documents:
        print("🔴 DB를 구축할 데이터가 없습니다. RAG 기능이 정상 동작하지 않을 수 있습니다.")
        return
    
    print("  - 텍스트 분할 및 임베딩 진행 중... (시간이 걸릴 수 있습니다)")
    split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(all_documents)
    Chroma.from_documents(
        documents=split_docs,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        persist_directory=CHROMA_DB_PATH
    )
    print(f"🎉 Vector DB 구축 완료! ({CHROMA_DB_PATH})")

# ### MODIFIED FUNCTION ###: 로컬에 다운로드된 폰트를 직접 사용하는 방식으로 변경
def get_multilingual_font(size=16, bold=False, lang_code='KO'):
    """
    로컬 ./fonts 폴더에 다운로드된 Noto 폰트를 사용하여 다국어 텍스트 렌더링을 지원합니다.
    언어 코드에 따라 적절한 폰트 파일을 선택하여 tofu 현상을 방지합니다.
    """
    style = "Bold" if bold else "Regular"
    
    # 언어 코드에 따른 폰트 파일 매핑 (TTF 우선, OTF 폴백)
    font_map = {
        'KO': [f'NotoSansKR-{style}.ttf', f'NotoSansKR-{style}.otf'],
        'JA': [f'NotoSansJP-{style}.ttf', f'NotoSansJP-{style}.otf'],
        'ZH': [f'NotoSansSC-{style}.ttf', f'NotoSansSC-{style}.otf'],
        # 우크라이나어(키릴) - 키릴 문자 지원 폰트 추가
        'UK': [f'NotoSans-{style}.ttf', f'NotoSansKR-{style}.ttf', f'NotoSansKR-{style}.otf'],
        'VI': [f'NotoSans-{style}.ttf'],
        'EN': [f'NotoSans-{style}.ttf'],
    }
    
    # 요청된 언어의 폰트 파일명 가져오기, 없으면 기본 NotoSans 사용
    font_candidates = font_map.get(lang_code.upper(), [f'NotoSans-{style}.ttf'])
    
    # TTF 파일을 우선적으로 찾기
    for font_filename in font_candidates:
        font_path = FONTS_DIR / font_filename
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception as e:
                print(f"⚠️ 폰트 로드 실패 '{font_filename}': {e}")
                continue
    
    # 모든 후보 폰트가 실패한 경우 폴백 시도
    fallback_candidates = [
        FONTS_DIR / "NotoSans-Regular.ttf",
        FONTS_DIR / "NotoSans-Regular.otf"
    ]
    
    for fallback_path in fallback_candidates:
        if fallback_path.exists():
            try:
                print(f"⚠️ 경고: 요청된 폰트를 찾을 수 없어 '{fallback_path.name}'로 대체합니다.")
                return ImageFont.truetype(str(fallback_path), size)
            except Exception as e:
                print(f"⚠️ 폴백 폰트 로드 실패 '{fallback_path.name}': {e}")
                continue
    
    # 최후의 수단
    print("❌ 모든 폰트 로드 실패. PIL 기본 폰트를 사용합니다. 글자가 깨질 수 있습니다.")
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def get_emoji_font(size=16):
    """
    이모지 전용 폰트를 로드합니다.
    """
    emoji_font_path = FONTS_DIR / "NotoColorEmoji-Regular.ttf"
    if emoji_font_path.exists():
        try:
            return ImageFont.truetype(str(emoji_font_path), size)
        except Exception as e:
            print(f"⚠️ 이모지 폰트 로드 실패: {e}")
    return None

def draw_text_with_emoji(draw, text, position, main_font, emoji_font, align='left', color='#000000'):
    """
    이모지와 일반 텍스트를 혼합하여 렌더링합니다.
    align: 'left', 'center', 'right'
    """
    if not emoji_font:
        # 이모지 폰트가 없으면 기본 폰트로 렌더링
        if align == 'center':
            bbox = draw.textbbox((0, 0), text, font=main_font)
            x = position[0] - (bbox[2] - bbox[0]) // 2
            draw.text((x, position[1]), text, fill=color, font=main_font)
        else:
            draw.text(position, text, fill=color, font=main_font)
        return
    
    # 이모지와 일반 텍스트를 분리
    import re
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270]')
    
    # 이모지 위치 찾기
    emoji_positions = []
    for match in emoji_pattern.finditer(text):
        emoji_positions.append((match.start(), match.end(), match.group()))
    
    if not emoji_positions:
        # 이모지가 없으면 기본 렌더링
        if align == 'center':
            bbox = draw.textbbox((0, 0), text, font=main_font)
            x = position[0] - (bbox[2] - bbox[0]) // 2
            draw.text((x, position[1]), text, fill=color, font=main_font)
        else:
            draw.text(position, text, fill=color, font=main_font)
        return
    
    # 텍스트를 이모지와 일반 텍스트로 분할하여 렌더링
    current_x = position[0]
    if align == 'center':
        # 전체 텍스트 너비 계산
        total_width = 0
        last_end = 0
        for start, end, emoji in emoji_positions:
            # 이모지 앞의 일반 텍스트
            if start > last_end:
                text_part = text[last_end:start]
                bbox = draw.textbbox((0, 0), text_part, font=main_font)
                total_width += bbox[2] - bbox[0]
            # 이모지
            bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
            total_width += bbox[2] - bbox[0]
            last_end = end
        
        # 마지막 일반 텍스트
        if last_end < len(text):
            text_part = text[last_end:]
            bbox = draw.textbbox((0, 0), text_part, font=main_font)
            total_width += bbox[2] - bbox[0]
        
        current_x = position[0] - total_width // 2
    
    # 실제 렌더링
    last_end = 0
    for start, end, emoji in emoji_positions:
        # 이모지 앞의 일반 텍스트
        if start > last_end:
            text_part = text[last_end:start]
            draw.text((current_x, position[1]), text_part, fill=color, font=main_font)
            bbox = draw.textbbox((0, 0), text_part, font=main_font)
            current_x += bbox[2] - bbox[0]
        
        # 이모지
        draw.text((current_x, position[1]), emoji, fill=color, font=emoji_font)
        bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
        current_x += bbox[2] - bbox[0]
        last_end = end
    
    # 마지막 일반 텍스트
    if last_end < len(text):
        text_part = text[last_end:]
        draw.text((current_x, position[1]), text_part, fill=color, font=main_font)


def extract_text_from_file(file_path: str) -> tuple[str, str]:
    if not file_path or not os.path.exists(file_path):
        return "", "파일을 찾을 수 없습니다."

    try:
        # Upstage 라이브러리가 이미지와 문서를 처리합니다. JPG도 여기에 포함됩니다.
        pages = UpstageDocumentParseLoader(file_path, ocr="force").load()
        extracted_text = "\n\n".join([p.page_content for p in pages if p.page_content])
        
        if not extracted_text.strip():
            return "", "파일에서 텍스트를 추출할 수 없었습니다. 내용이 비어있거나 인식이 어렵습니다."
            
        return extracted_text, "성공"
    except Exception as e:
        # 오류 발생 시 더 구체적인 메시지 반환
        error_message = f"파일 처리 중 오류가 발생했습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다.\n(서버 오류: {str(e)})"
        print(f"❌ 텍스트 추출 실패: {error_message}")
        return "", error_message

def perform_rule_based_analysis(contract_text: str) -> dict:
    alerts, safety_score = [], 100
    try:
        # 1. 기존의 키워드 기반 분석 (유지)
        categories = {
            "보증금_반환": {"keywords": ["보증금", "반환", "즉시", "계약종료"], "risk": "CRITICAL"},
            "권리관계_유지": {"keywords": ["권리관계", "익일", "근저당", "대항력"], "risk": "CRITICAL"},
            "전세자금대출": {"keywords": ["대출", "불가", "무효", "전세자금"], "risk": "WARNING"},
            "수선_의무": {"keywords": ["수선", "하자", "파손", "수리"], "risk": "ADVISORY"},
            "특약사항": {"keywords": ["특약", "기타사항", "추가조건"], "risk": "ADVISORY"}
        }
        for cat_name, info in categories.items():
            display_name = cat_name.replace('_', ' ').title()
            keyword_count = sum(1 for kw in info['keywords'] if kw in contract_text)
            if keyword_count < len(info['keywords']) * 0.5:
                if info['risk'] == "CRITICAL":
                    safety_score -= 40
                    alerts.append(f"🚨 [치명적!] {display_name}: 관련 조항이 누락되었거나 미비하여 심각한 위험이 발생할 수 있습니다!")
                elif info['risk'] == "WARNING":
                    safety_score -= 20
                    alerts.append(f"⚠️ [위험] {display_name}: 관련 조항이 부족하여 주의가 필요합니다.")
                else:
                    safety_score -= 10
                    alerts.append(f"💡 [권장] {display_name}: 분쟁 예방을 위해 관련 조항 보강을 권장합니다.")
            else:
                alerts.append(f"✅ [{display_name}] 관련 조항이 확인되었습니다.")

        safety_score = max(0, safety_score)

        # 2. 🔥 임대인 이름 추출 및 상습 채무 불이행자 명단 조회 (핵심 기능 추가)
        print("  [임대인 검사] 임대인 신원 조회 시작...")
        landlord_name = extract_landlord_name_robustly(contract_text)
        
        if landlord_name == "이름 자동 추출 실패":
            alerts.append("⚠️ [임대인 검사] 계약서에서 임대인 이름을 자동으로 찾지 못했습니다. 직접 확인이 필요합니다.")
        else:
            found_defaulter = False
            try:
                with open(DEFAULTER_LIST_PATH, 'r', encoding='utf-8-sig') as f:
                    # CSV 파일의 모든 행을 미리 리스트로 로드하여 검색 효율성 증대
                    defaulter_list = list(csv.DictReader(f))
                    for row in defaulter_list:
                        # 이름 비교 시 공백 제거 후 비교
                        defaulter_name = row.get('성명', '').strip().replace(' ', '')
                        if landlord_name == defaulter_name:
                            safety_score = 0  # << 치명적 위험이므로 안전점수 0점으로 조정
                            alerts.append(f"🚨🚨🚨 [치명적 위험!] 임대인 '{landlord_name}'이(가) 상습 채무 불이행자 명단에 포함되어 있습니다! **계약을 즉시 중단하고 전문가와 상담하세요.**")
                            found_defaulter = True
                            break
                if not found_defaulter:
                    alerts.append(f"✅ [임대인 검사] 임대인('{landlord_name}')은(는) 상습 채무 불이행자 명단에 없습니다.")
            except FileNotFoundError:
                alerts.append(f"⚠️ [임대인 검사] 상습 채무불이행자 명단 파일을 찾을 수 없어 조회가 불가능합니다. ({DEFAULTER_LIST_PATH})")
            except Exception as e:
                alerts.append(f"⚠️ [임대인 검사] 명단 파일 처리 중 오류 발생: {e}")

    except Exception as e:
        alerts.append(f"⚠️ 규칙 기반 분석 중 오류 발생: {e}")
        safety_score = -1
    
    # 안전 점수 순으로 정렬하여 중요한 경고가 위로 오게 함
    alerts.sort(key=lambda x: ('🚨' not in x, '⚠️' not in x, '💡' not in x, '✅' not in x))

    return {"alerts": alerts, "safety_score": safety_score}

def google_text_to_speech(text, lang_code="KO"):
    if not GOOGLE_API_KEY:
        return None, "Google API 키가 설정되지 않아 음성 생성이 불가능합니다."
    
    # 특수문자 일부 제거 (음성 변환 품질 향상)
    text = re.sub(r"[^\w\s가-힣.,!?]", "", text, flags=re.UNICODE)
    
    text_chunks = split_text_for_tts(text)
    
    voice_map = {
        "KO": {"languageCode": "ko-KR", "name": "ko-KR-Wavenet-A"},
        "EN": {"languageCode": "en-US", "name": "en-US-Wavenet-F"},
        "JA": {"languageCode": "ja-JP", "name": "ja-JP-Wavenet-A"},
        "ZH": {"languageCode": "cmn-CN", "name": "cmn-CN-Wavenet-A"},
        "UK": {"languageCode": "uk-UA", "name": "uk-UA-Wavenet-A"}, # 우크라이나어
        "VI": {"languageCode": "vi-VN", "name": "vi-VN-Wavenet-A"}  # 베트남어
    }

    if lang_code.upper() not in voice_map:
        return None, f"지원하지 않는 언어 코드: {lang_code}"
        
    try:
        # 긴 텍스트의 경우 첫 번째 청크만 처리하여 샘플 제공 (Gradio에서는 전체를 처리하면 시간이 너무 오래 걸릴 수 있음)
        first_chunk = text_chunks[0] if text_chunks else ""
        if not first_chunk:
             return None, "음성으로 변환할 텍스트가 없습니다."

        request_body = {
            "input": {"text": first_chunk},
            "voice": voice_map[lang_code.upper()],
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": 0.9, "pitch": -2}
        }
        
        response = requests.post(TTS_API_URL, data=json.dumps(request_body), timeout=30)

        if response.status_code == 200:
            audio_content = base64.b64decode(response.json()['audioContent'])
            # Gradio에서는 임시 파일을 사용하는 것이 안정적
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_content)
                # 메시지 개선
                msg = "음성 생성 완료!"
                if len(text_chunks) > 1:
                    msg = f"음성 생성 완료 🎵 "
                return tmp_file.name, msg
        else:
            return None, f"TTS API 오류: {response.text}"
    except Exception as e:
        return None, f"TTS 스크립트 오류: {e}"

RETRIEVER = None
def initialize_retriever():
    """전역 RAG 검색기를 초기화합니다."""
    global RETRIEVER
    if os.path.exists(CHROMA_DB_PATH):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large")
            )
            RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 5})
            print("✅ RAG 검색기(Retriever) 초기화 완료.")
        except Exception as e:
            print(f"❌ RAG 검색기 초기화 실패: {e}")
    else:
        print("⚠️ Vector DB 경로를 찾을 수 없어 RAG 검색기를 초기화할 수 없습니다.")

# 🎨 이미지 생성 및 UI 관련 함수들
# 🎨 이미지 생성을 위한 색상 및 폰트 설정 (PIL 폴백용) - 초록색 테마
COLORS = {
    'bg': '#f0f9ff',
    'white': '#ffffff',
    'primary': '#10b981',
    'success': '#059669',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text': '#1f2937',
    'muted': '#6b7280',
    'border': '#d1fae5',
    'accent': '#047857'
}

EMBED_HEAD = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- 다국어 지원을 위한 Noto Sans 폰트 패밀리 -->
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;700&family=Noto+Sans+KR:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --card-bg: #ffffff;
    --bg: #f0fdf4;
    --border: #d1fae5;
    --text: #1f2937;
    --text-weak: #4b5563;
    --muted: #6b7280;
    --primary: #10b981;
    --primary-dark: #059669;
    --accent: #047857;
    --shadow: rgba(16, 185, 129, 0.1);
    --badge-bg: #ecfdf5;
    --badge-text: #065f46;
    --badge-border: #a7f3d0;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --card-bg: #0f172a;
      --bg: #020617;
      --border: #1e293b;
      --text: #e2e8f0;
      --text-weak: #cbd5e1;
      --muted: #94a3b8;
      --primary: #34d399;
      --primary-dark: #10b981;
      --accent: #059669;
      --shadow: rgba(16, 185, 129, 0.2);
      --badge-bg: #064e3b;
      --badge-text: #6ee7b7;
      --badge-border: #065f46;
    }
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Noto Sans KR', 'Noto Sans', 'Noto Sans JP', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    line-height: 1.7;
  }
</style>
"""

# 🔥 FIXED: 줄바꿈 문제를 해결한 CSS
DEFAULT_EMBED_CSS = """
.report-wrap { max-width: 980px; margin: 0 auto; padding: 32px; }
.report-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 20px; overflow: hidden; box-shadow: 0 10px 35px var(--shadow); }
.report-header { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); padding: 32px; color: #fff; }
.report-header h1 { margin: 0 0 8px 0; font-size: 26px; font-weight: 700; }
.report-header .meta { font-size: 14px; opacity: .9; }
.report-section { padding: 28px 32px; border-top: 1px solid var(--border); }
.report-section:last-child { border-bottom: none; }
.report-section h2 { margin: 0 0 16px 0; font-size: 20px; font-weight: 700; color: var(--text); padding-bottom: 8px; border-bottom: 2px solid var(--primary); display: inline-block;}
.alerts { display: grid; gap: 12px; }
.alert { padding: 14px 18px; border-radius: 12px; border: 1px solid transparent; display: flex; align-items: center; gap: 10px; }
.alert::before { font-size: 20px; }
.alert.critical { border-color:#fecaca; background:#fff1f2; color:#b91c1c; }
.alert.critical::before { content: '🚨'; }
.alert.warn { border-color:#fde68a; background:#fffbeb; color: #b45309; }
.alert.warn::before { content: '⚠️'; }
.alert.ok { border-color:#bbf7d0; background:#f0fdf4; color: #15803d; }
.alert.ok::before { content: '✅'; }
@media (prefers-color-scheme: dark) {
  .alert.critical { background:#2d1516; border-color:#7f1d1d; color:#fca5a5; }
  .alert.warn { background:#2d230d; border-color:#7c5800; color:#fde047; }
  .alert.ok { background:#112a1a; border-color:#14532d; color:#86efac; }
}
.grade { display:inline-block; padding: 8px 14px; border-radius:999px; border:1px solid rgba(255,255,255,.5); font-weight:700; background: rgba(255,255,255,.2); backdrop-filter: blur(5px); }
.footer-note { color: var(--muted); font-size: 13px; text-align:center; padding: 24px; background: var(--bg); }
.badge { display:inline-block; padding:4px 10px; border-radius: 8px; background: var(--badge-bg); color: var(--badge-text); border:1px solid var(--badge-border); font-size:13px; font-weight: 500;}
.report-section p { margin: 0 0 12px 0; color: var(--text-weak); }
.report-section li { margin-bottom: 8px; color: var(--text-weak); }
.report-section a { color: var(--accent); text-decoration: none; font-weight: 500; }
.report-section a:hover { text-decoration: underline; }
.report-section strong { font-weight: 700; color: var(--text); }

/* 🔥 FIXED: 코드 블록 및 긴 텍스트 줄바꿈 처리 */
.report-section pre, .translation-content pre {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    background: var(--badge-bg);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border);
}

.report-section code, .translation-content code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    font-family: monospace;
    font-size: 0.9em;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--badge-bg);
}

/* 🔥 FIXED: 테이블 반응형 처리 (베트남어 지원 강화) */
.report-section table, .translation-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.report-section table { table-layout: fixed; }
.translation-content table { table-layout: auto; }
.translation-content .table-wrapper { 
    overflow-x: auto; 
    margin: 20px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.report-section th, .report-section td, .translation-content th, .translation-content td {
    border: 1px solid var(--border);
    padding: 12px 16px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    vertical-align: top;
}
.report-section th {
    background-color: var(--badge-bg);
    font-weight: 600;
}
.translation-content th {
    background: var(--primary);
    color: white;
    font-weight: 600;
    border-bottom: 2px solid var(--primary-dark);
}
.translation-content td {
    border-bottom: 1px solid var(--border);
}
.translation-content tr:nth-child(even) {
    background: var(--bg-light);
}
.translation-content tr:hover {
    background: var(--bg-hover);
}

/* 🔥 FIXED: 긴 단어 강제 줄바꿈으로 레이아웃 깨짐 방지 */
.report-section *, .translation-content * {
    word-break: break-word;
    overflow-wrap: break-word;
}


/* 번역 결과 전용 스타일 추가 */
    .translation-content {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 20px var(--shadow);
        line-height: 1.7;
        /* 베트남어 특수 문자 지원을 위한 폰트 설정 */
        font-family: 'Noto Sans', 'Noto Sans KR', 'Noto Sans JP', 'Noto Sans SC', sans-serif;
    }
.translation-content h1, .translation-content h2, .translation-content h3 {
    color: var(--primary);
    margin-top: 24px;
    margin-bottom: 16px;
}
.translation-content h1 {
    font-size: 24px;
    font-weight: 700;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 8px;
}
.translation-content h2 {
    font-size: 20px;
    font-weight: 600;
}
.translation-content h3 {
    font-size: 18px;
    font-weight: 500;
}
.translation-content p {
    margin-bottom: 12px;
    color: var(--text-weak);
}
.translation-content ul, .translation-content ol {
    margin: 16px 0;
    padding-left: 24px;
}
.translation-content li {
    margin-bottom: 8px;
    color: var(--text-weak);
}
.translation-content strong {
    color: var(--text);
    font-weight: 600;
}
.translation-content blockquote {
    border-left: 4px solid var(--primary);
    padding-left: 16px;
    margin: 16px 0;
    font-style: italic;
    color: var(--muted);
}
"""

EMBED_CSS = DEFAULT_EMBED_CSS

# 🔥 FIXED: 향상된 마크다운 -> HTML 변환 함수 (베트남어 및 테이블 지원 강화)
def md_to_html(md_text: str) -> str:
    """마크다운 텍스트를 HTML로 변환합니다. 베트남어 및 테이블 처리를 강화했습니다."""
    if not md_text:
        return ""
    
    if MARKDOWN_AVAILABLE:
        try:
            # markdown2 라이브러리 우선 사용 (테이블 지원 강화)
            import markdown2
            return markdown2.markdown(
                md_text, 
                extras=[
                    "fenced-code-blocks", 
                    "tables", 
                    "break-on-newline", 
                    "spoiler",
                    "strike",
                    "target-blank-links",
                    "cuddled-lists",
                    "footnotes"
                ]
            )
        except:
            try:
                # markdown 라이브러리 대안 사용
                import markdown
                return markdown.markdown(
                    md_text,
                    extensions=['codehilite', 'tables', 'fenced_code', 'nl2br', 'attr_list']
                )
            except:
                pass
    
    # 폴백: 기본 마크다운 파싱 (베트남어 및 테이블 지원 강화)
    # 전각 기호 정규화: ｜(U+FF5C), －(U+FF0D) 등을 ASCII로 변환해 테이블/수평선 인식 개선
    html = (
        md_text
        .replace('｜', '|')
        .replace('￨', '|')
        .replace('－', '-')
        .replace('﹣', '-')
        .replace('—', '-')
        .replace('–', '-')
    )
    
    # 베트남어 특수 문자 보존
    html = preserve_vietnamese_chars(html)
    
    # 코드 블록 처리 (``` 구문)
    html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # 인라인 코드 처리
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # 헤딩 변환 (개선된 패턴)
    html = re.sub(r'^###\s*(.*)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^##\s*(.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^#\s*(.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # 강조 텍스트 처리
    html = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', html)  # 볼드+이탤릭
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)  # 볼드
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)  # 이탤릭
    
    # 취소선
    html = re.sub(r'~~(.*?)~~', r'<del>\1</del>', html)
    
    # 링크 처리
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)
    
    # 테이블 처리 (강화된 버전)
    html = process_markdown_tables(html)
    
    # 리스트 처리 (개선된 버전)
    lines = html.split('\n')
    in_ul = False
    in_ol = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 순서 있는 리스트
        if re.match(r'^\d+\.\s+', stripped):
            if not in_ol:
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                result_lines.append('<ol>')
                in_ol = True
            content = re.sub(r'^\d+\.\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        # 순서 없는 리스트
        elif re.match(r'^[-*+]\s+', stripped):
            if not in_ul:
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append('<ul>')
                in_ul = True
            content = re.sub(r'^[-*+]\s+', '', stripped)
            result_lines.append(f'<li>{content}</li>')
        else:
            # 리스트 종료
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(line)
    
    # 남은 리스트 태그 정리
    if in_ul:
        result_lines.append('</ul>')
    if in_ol:
        result_lines.append('</ol>')
    
    html = '\n'.join(result_lines)
    
    # 블록쿼트 처리
    html = re.sub(r'^>\s*(.*)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # 수평선 처리
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)
    html = re.sub(r'^\*\*\*+$', r'<hr>', html, flags=re.MULTILINE)
    
    # 단락 처리 (개선된 버전)
    paragraphs = re.split(r'\n\s*\n', html)
    processed_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # HTML 태그로 시작하는 경우 단락 태그 추가하지 않음
        if re.match(r'^<(?:h[1-6]|ul|ol|li|blockquote|pre|hr|div|table)', para, re.IGNORECASE):
            processed_paragraphs.append(para)
        else:
            # 일반 텍스트는 p 태그로 감싸기
            para = para.replace('\n', '<br>')
            processed_paragraphs.append(f'<p>{para}</p>')
    
    html = '\n\n'.join(processed_paragraphs)
    
    return html

def preprocess_markdown_for_translation(text: str) -> str:
    """
    번역된 텍스트의 마크다운을 전처리하여 테이블 깨짐 현상을 방지합니다.
    """
    if not text:
        return text
    
    # 테이블 구조 보존을 위한 전처리
    lines = text.split('\n')
    processed_lines = []
    in_table = False
    table_buffer = []
    
    for line in lines:
        stripped = line.strip()
        
        # 테이블 시작 감지 (파이프 | 포함)
        if '|' in stripped and not stripped.startswith('http'):
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
        # 테이블 구분선 감지 (--- 또는 ===)
        elif in_table and (stripped.startswith('|') and ('---' in stripped or '===' in stripped)):
            table_buffer.append(line)
        # 테이블 종료 감지 (빈 줄 또는 파이프가 없는 줄)
        elif in_table and (not stripped or '|' not in stripped):
            # 테이블 버퍼 처리
            if table_buffer:
                processed_lines.extend(process_table_markdown(table_buffer))
                table_buffer = []
            in_table = False
            processed_lines.append(line)
        # 테이블 내부 라인
        elif in_table:
            table_buffer.append(line)
        # 일반 텍스트
        else:
            processed_lines.append(line)
    
    # 마지막 테이블 처리
    if table_buffer:
        processed_lines.extend(process_table_markdown(table_buffer))
    
    return '\n'.join(processed_lines)

def process_table_markdown(table_lines: list) -> list:
    """
    테이블 마크다운을 처리하여 깨짐 현상을 방지합니다.
    """
    if not table_lines:
        return []
    
    processed_lines = []
    
    for i, line in enumerate(table_lines):
        if '|' in line:
            # 테이블 셀 내용 정리
            cells = line.split('|')
            cleaned_cells = []
            
            for cell in cells:
                # 셀 내용 정리 (공백, 특수문자 처리)
                cleaned_cell = cell.strip()
                if cleaned_cell:
                    # 베트남어 특수 문자 보존
                    cleaned_cell = preserve_vietnamese_chars(cleaned_cell)
                    cleaned_cells.append(cleaned_cell)
                else:
                    cleaned_cells.append(' ')
            
            # 테이블 라인 재구성
            processed_line = '|' + '|'.join(cleaned_cells) + '|'
            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)
    
    return processed_lines

def process_markdown_tables(html: str) -> str:
    """
    마크다운 테이블을 HTML 테이블로 변환합니다.
    """
    lines = html.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 테이블 시작 감지 (파이프 | 포함)
        if '|' in line and not line.startswith('http'):
            table_lines = []
            header_line = line
            
            # 헤더 라인 추가
            table_lines.append(header_line)
            i += 1
            
            # 구분선 확인
            if i < len(lines) and '|' in lines[i] and ('---' in lines[i] or '===' in lines[i]):
                separator_line = lines[i]
                table_lines.append(separator_line)
                i += 1
            
            # 테이블 본문 수집
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            # 테이블을 HTML로 변환
            html_table = convert_table_to_html(table_lines)
            result_lines.append(html_table)
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines)

def convert_table_to_html(table_lines: list) -> str:
    """
    마크다운 테이블 라인을 HTML 테이블로 변환합니다.
    """
    if not table_lines:
        return ""
    
    html_parts = ['<div class="table-wrapper">', '<table>']
    
    for i, line in enumerate(table_lines):
        if '|' in line:
            # 구분선 라인은 건너뛰기
            if '---' in line or '===' in line:
                continue
            
            # 테이블 셀 분리 및 정리
            cells = [cell.strip() for cell in line.split('|')]
            
            # 빈 셀 처리
            if cells and not cells[0]:  # 첫 번째 빈 셀 제거
                cells = cells[1:]
            if cells and not cells[-1]:  # 마지막 빈 셀 제거
                cells = cells[:-1]
            
            if cells:
                if i == 0:  # 헤더 라인
                    html_parts.append('<thead>')
                    html_parts.append('<tr>')
                    for cell in cells:
                        html_parts.append(f'<th>{cell}</th>')
                    html_parts.append('</tr>')
                    html_parts.append('</thead>')
                    html_parts.append('<tbody>')
                else:  # 본문 라인
                    html_parts.append('<tr>')
                    for cell in cells:
                        html_parts.append(f'<td>{cell}</td>')
                    html_parts.append('</tr>')
    
    html_parts.append('</tbody>')
    html_parts.append('</table>')
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)

def preserve_vietnamese_chars(text: str) -> str:
    """
    베트남어 특수 문자를 보존합니다.
    """
    # 베트남어 특수 문자 매핑
    vietnamese_chars = {
        'à': 'à', 'á': 'á', 'ạ': 'ạ', 'ả': 'ả', 'ã': 'ã',
        'â': 'â', 'ầ': 'ầ', 'ấ': 'ấ', 'ậ': 'ậ', 'ẩ': 'ẩ', 'ẫ': 'ẫ',
        'ă': 'ă', 'ằ': 'ằ', 'ắ': 'ắ', 'ặ': 'ặ', 'ẳ': 'ẳ', 'ẵ': 'ẵ',
        'è': 'è', 'é': 'é', 'ẹ': 'ẹ', 'ẻ': 'ẻ', 'ẽ': 'ẽ',
        'ê': 'ê', 'ề': 'ề', 'ế': 'ế', 'ệ': 'ệ', 'ể': 'ể', 'ễ': 'ễ',
        'ì': 'ì', 'í': 'í', 'ị': 'ị', 'ỉ': 'ỉ', 'ĩ': 'ĩ',
        'ò': 'ò', 'ó': 'ó', 'ọ': 'ọ', 'ỏ': 'ỏ', 'õ': 'õ',
        'ô': 'ô', 'ồ': 'ồ', 'ố': 'ố', 'ộ': 'ộ', 'ổ': 'ổ', 'ỗ': 'ỗ',
        'ơ': 'ơ', 'ờ': 'ờ', 'ớ': 'ớ', 'ợ': 'ợ', 'ở': 'ở', 'ỡ': 'ỡ',
        'ù': 'ù', 'ú': 'ú', 'ụ': 'ụ', 'ủ': 'ủ', 'ũ': 'ũ',
        'ư': 'ư', 'ừ': 'ừ', 'ứ': 'ứ', 'ự': 'ự', 'ử': 'ử', 'ữ': 'ữ',
        'ỳ': 'ỳ', 'ý': 'ý', 'ỵ': 'ỵ', 'ỷ': 'ỷ', 'ỹ': 'ỹ',
        'đ': 'đ'
    }
    
    for original, preserved in vietnamese_chars.items():
        text = text.replace(original, preserved)
    
    return text

def create_translated_html(translated_text: str, title: str = "번역된 내용") -> str:
    """번역된 텍스트를 예쁜 HTML로 변환 (이모지 및 특수문자 처리 개선 + 베트남어 지원)"""
    # 특수문자 및 이모지 처리 개선
    processed_text = translated_text
    
    # 우크라이나어 번역 결과에서 "0000" 같은 특수 패턴 처리
    if "0000" in processed_text and any(char in processed_text for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
        # 우크라이나어 텍스트로 인식하여 특수 처리
        processed_text = processed_text.replace("0000", "Результат перекладу")  # 실제 우크라이나어 텍스트로 대체
    
    # 베트남어 특수 문자 처리 (마크다운 인식 개선)
    if any(char in processed_text for char in "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"):
        # 베트남어 특수 문자를 안전하게 처리
        processed_text = processed_text.replace("à", "à").replace("á", "á").replace("ạ", "ạ")
        processed_text = processed_text.replace("ả", "ả").replace("ã", "ã").replace("â", "â")
        processed_text = processed_text.replace("ầ", "ầ").replace("ấ", "ấ").replace("ậ", "ậ")
        processed_text = processed_text.replace("ẩ", "ẩ").replace("ẫ", "ẫ").replace("ă", "ă")
        processed_text = processed_text.replace("ằ", "ằ").replace("ắ", "ắ").replace("ặ", "ặ")
        processed_text = processed_text.replace("ẳ", "ẳ").replace("ẵ", "ẵ").replace("è", "è")
        processed_text = processed_text.replace("é", "é").replace("ẹ", "ẹ").replace("ẻ", "ẻ")
        processed_text = processed_text.replace("ẽ", "ẽ").replace("ê", "ê").replace("ề", "ề")
        processed_text = processed_text.replace("ế", "ế").replace("ệ", "ệ").replace("ể", "ể")
        processed_text = processed_text.replace("ễ", "ễ").replace("ì", "ì").replace("í", "í")
        processed_text = processed_text.replace("ị", "ị").replace("ỉ", "ỉ").replace("ĩ", "ĩ")
        processed_text = processed_text.replace("ò", "ò").replace("ó", "ó").replace("ọ", "ọ")
        processed_text = processed_text.replace("ỏ", "ỏ").replace("õ", "õ").replace("ô", "ô")
        processed_text = processed_text.replace("ồ", "ồ").replace("ố", "ố").replace("ộ", "ộ")
        processed_text = processed_text.replace("ổ", "ổ").replace("ỗ", "ỗ").replace("ơ", "ơ")
        processed_text = processed_text.replace("ờ", "ờ").replace("ớ", "ớ").replace("ợ", "ợ")
        processed_text = processed_text.replace("ở", "ở").replace("ỡ", "ỡ").replace("ù", "ù")
        processed_text = processed_text.replace("ú", "ú").replace("ụ", "ụ").replace("ủ", "ủ")
        processed_text = processed_text.replace("ũ", "ũ").replace("ư", "ư").replace("ừ", "ừ")
        processed_text = processed_text.replace("ứ", "ứ").replace("ự", "ự").replace("ử", "ử")
        processed_text = processed_text.replace("ữ", "ữ").replace("ỳ", "ỳ").replace("ý", "ý")
        processed_text = processed_text.replace("ỵ", "ỵ").replace("ỷ", "ỷ").replace("ỹ", "ỹ")
        processed_text = processed_text.replace("đ", "đ")
    
    # 마크다운 변환 전 텍스트 전처리 (테이블 깨짐 방지)
    processed_text = preprocess_markdown_for_translation(processed_text)
    
    html_content = md_to_html(processed_text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"""
    <html>
      <head>
        {EMBED_HEAD}
        <style>{EMBED_CSS}</style>
      </head>
      <body>
        <div class="report-wrap">
          <div class="translation-content">
            <div style="border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;">
              <h1 style="margin: 0; color: var(--primary);">🌐 {title}</h1>
              <p style="margin: 8px 0 0 0; color: var(--muted); font-size: 14px;">📅 {timestamp}</p>
            </div>
            <div>{html_content}</div>
          </div>
        </div>
      </body>
    </html>
    """

def extract_landlord_name_robustly(contract_text: str) -> str:
    """🔥 3단계에 걸쳐 임대인 이름을 집요하게 추출하는 함수 (Gradio용 수정)"""
    llm = ChatUpstage()

    # --- 1단계: 이름만 정확히 추출 시도 ---
    prompt_step1 = ChatPromptTemplate.from_template(
        "다음 계약서 텍스트에서 '임대인' 또는 '집주인'의 이름만 정확하게 추출해줘. "
        "다른 말은 모두 제외하고 이름만 말해줘. (예: 홍길동). "
        "만약 이름이 없으면 '없음'이라고 말해줘. 텍스트: {contract}"
    )
    chain_step1 = prompt_step1 | llm | StrOutputParser()
    name_step1 = chain_step1.invoke({"contract": contract_text}).strip()

    # 1단계 검증: 2~5글자의 한글 이름인지 확인
    if re.fullmatch(r'[가-힣]{2,5}', name_step1.replace(" ", "")):
        print(f"  [임대인 검사] 1단계 성공: '{name_step1}' 추출")
        return name_step1.replace(" ", "")

    # --- 2단계: 실패 시, 문장 단위로 추출 후 파이썬으로 이름 찾기 ---
    print("  [임대인 검사] 1단계 실패, 2단계 시도 중...")
    prompt_step2 = ChatPromptTemplate.from_template(
        "다음 계약서 텍스트에서 '임대인' 또는 '집주인'의 이름이 포함된 라인 또는 문장 전체를 그대로 알려줘. "
        "텍스트: {contract}"
    )
    chain_step2 = prompt_step2 | llm | StrOutputParser()
    sentence = chain_step2.invoke({"contract": contract_text})
    
    # 2단계 검증: 문장에서 2~5글자 한글 패턴 찾기
    match = re.search(r'[가-힣]{2,5}', sentence)
    if match:
        name_step2 = match.group(0)
        print(f"  [임대인 검사] 2단계 성공: '{name_step2}' 추출")
        return name_step2

    # --- 3단계: AI 추출 실패 알림 ---
    # Gradio 환경에서는 CLI처럼 사용자 입력(input)을 받을 수 없으므로,
    # 자동 추출에 실패했음을 알리고 종료합니다.
    print("  [임대인 검사] 2단계 실패. 자동 이름 추출에 실패했습니다.")
    return "이름 자동 추출 실패"

# ### MODIFIED FUNCTION ###: get_multilingual_font에 lang_code를 전달하도록 수정
def create_clean_report_image(report_text: str, report_type: str = "report", lang_code: str = 'KO') -> Image.Image:
    """깔끔한 텍스트 기반 리포트 이미지 생성 (다국어 지원 + 이모지 지원)"""
    width = 1200
    margin = 50
    line_height = 28
    
    # 다국어 폰트 설정 (언어 코드 전달)
    title_font = get_multilingual_font(28, bold=True, lang_code=lang_code)
    heading_font = get_multilingual_font(20, bold=True, lang_code=lang_code) 
    text_font = get_multilingual_font(16, bold=False, lang_code=lang_code)
    small_font = get_multilingual_font(14, bold=False, lang_code=lang_code)
    
    # 이모지 폰트 설정
    emoji_font = get_emoji_font(16)
    
    # 폰트 로드 실패 시 안전장치
    if not title_font or not heading_font or not text_font or not small_font:
        print("⚠️ 폰트 로드 실패 - 텍스트 렌더링을 건너뜁니다.")
        # 기본 이미지 생성
        img = Image.new('RGB', (width, 600), '#ffffff')
        draw = ImageDraw.Draw(img)
        error_font = ImageFont.load_default()
        draw.text((margin, margin), "A required font could not be loaded.\nCannot render the report image.", fill='#dc2626', font=error_font)
        return img
    
    # 텍스트 전처리 및 높이 계산
    lines = []
    current_y = margin + 60
    
    # 제목 추가 (이모지 포함)
    if "translation" in report_type.lower() or "번역" in report_type:
        if lang_code == 'EN':
            title = "🌐 Translation Result"
        elif lang_code == 'JA':
            title = "🌐 翻訳結果"
        elif lang_code == 'ZH':
            title = "🌐 翻译结果"
        # 우크라이나어, 베트남어 제목 추가
        elif lang_code == 'UK':
            title = "🌐 Результат перекладу"
        elif lang_code == 'VI':
            title = "🌐 Kết quả dịch"
        else:
            title = "🌐 번역 결과"
    elif "analysis" in report_type or "분석" in report_type:
        title = "AI 부동산 계약서 분석 리포트"
    else:
        title = "AI 상담 답변"
    
    lines.append(('title', title, current_y))
    current_y += 50
    
    # 날짜 추가 (이모지 제거하여 tofu 방지)
    now = datetime.now()
    if lang_code == 'KO':
        date_str = f"생성일시: {now.strftime('%Y')}년 {now.strftime('%m')}월 {now.strftime('%d')}일 {now.strftime('%H')}시 {now.strftime('%M')}분"
    elif lang_code == 'EN':
        date_str = f"Generated: {now.strftime('%Y-%m-%d %H:%M')}"
    elif lang_code == 'JA':
        date_str = f"生成日時: {now.strftime('%Y')}年{now.strftime('%m')}月{now.strftime('%d')}日 {now.strftime('%H')}時{now.strftime('%M')}分"
    elif lang_code == 'ZH':
        date_str = f"生成时间: {now.strftime('%Y')}年{now.strftime('%m')}月{now.strftime('%d')}日 {now.strftime('%H')}时{now.strftime('%M')}分"
    elif lang_code == 'UK':
        date_str = f"Створено: {now.strftime('%Y-%m-%d %H:%M')}"
    elif lang_code == 'VI':
        date_str = f"Được tạo: {now.strftime('%Y-%m-%d %H:%M')}"
    else: # 혹시 모를 예외 처리
        date_str = f"Generated: {now.strftime('%Y-%m-%d %H:%M')}"

    lines.append(('date', date_str, current_y))
    current_y += 40
    
    # 구분선
    lines.append(('divider', '', current_y))
    current_y += 30
    
    # 본문 처리
    for line in report_text.split('\n'):
        line = line.strip()
        if not line:
            current_y += 15
            continue
            
        # 헤딩 처리 (# 제거)
        if line.startswith('# '):
            text = line[2:].strip()
            lines.append(('h1', text, current_y))
            current_y += 45
        elif line.startswith('## '):
            text = line[3:].strip()
            lines.append(('h2', text, current_y))
            current_y += 35
        elif line.startswith('### '):
            text = line[4:].strip()
            lines.append(('h3', text, current_y))
            current_y += 30
        # 리스트 처리
        elif line.startswith('- '):
            text = line[2:].strip()
            # 긴 텍스트는 자동 줄바꿈 (언어별 너비 조정)
            if lang_code in ['UK', 'ZH', 'JA']:
                # 키릴, 중국어, 일본어는 더 좁은 너비로 줄바꿈
                wrapped = textwrap.fill(text, width=50)
            else:
                # 한국어, 영어 등은 기존 너비
                wrapped = textwrap.fill(text, width=70)
            for wrapped_line in wrapped.split('\n'):
                lines.append(('bullet', wrapped_line, current_y))
                current_y += line_height
        # 볼드 처리 (** 제거)
        elif line.startswith('**') and line.endswith('**'):
            text = line[2:-2].strip()
            lines.append(('bold', text, current_y))
            current_y += line_height
        # 구분선
        elif '---' in line:
            lines.append(('divider', '', current_y))
            current_y += 20
        # 일반 텍스트
        else:
            # 긴 줄 자동 줄바꿈 (언어별 너비 조정)
            if lang_code in ['UK', 'ZH', 'JA']:
                # 키릴, 중국어, 일본어는 더 좁은 너비로 줄바꿈
                wrapped = textwrap.fill(line, width=55)
            else:
                # 한국어, 영어 등은 기존 너비
                wrapped = textwrap.fill(line, width=75)
            for wrapped_line in wrapped.split('\n'):
                lines.append(('text', wrapped_line, current_y))
                current_y += line_height
    
    # 푸터 공간
    current_y += 30
    footer_text = "본 분석은 참고용이며 법적 효력이 없습니다. 중요한 결정 전 반드시 전문가와 상담하시기 바랍니다."
    # lang_code 값 체계(KO/EN/JA/ZH/UK/VI)에 맞춰 현지화 (이모지 제거하여 tofu 방지)
    if lang_code == 'EN':
        footer_text = "This analysis is for reference only and has no legal effect. Please consult with experts before making important decisions."
    elif lang_code == 'JA':
        footer_text = "この分析は参考用であり、法的効力はありません。重要な決定の前に必ず専門家にご相談ください。"
    elif lang_code == 'ZH':
        footer_text = "本分析仅供参考，不具有法律效力。在做出重要决定之前，请务必咨询专家。"
    elif lang_code == 'UK':
        footer_text = "Цей аналіз призначений лише для ознайомлення та не має юридичної сили. Проконсультуйтеся з експертами перед прийняттям важливих рішень."
    elif lang_code == 'VI':
        footer_text = "Phân tích này chỉ mang tính tham khảo và không có hiệu lực pháp lý. Vui lòng tham khảo ý kiến chuyên gia trước khi đưa ra quyết định quan trọng."

    lines.append(('footer', footer_text, current_y))
    current_y += 50
    
    # 최종 이미지 크기
    total_height = current_y + margin
    
    # 이미지 생성
    img = Image.new('RGB', (width, total_height), '#ffffff')
    draw = ImageDraw.Draw(img)
    
    # 헤더 배경
    header_height = 120
    draw.rectangle([0, 0, width, header_height], fill='#10b981')
    
    # 메인 컨텐츠 배경 (흰색 카드)
    draw.rectangle([margin//2, header_height, width-margin//2, total_height-margin//2],
                   fill='#ffffff', outline='#e5e7eb', width=2)
    
    # 텍스트 렌더링
    for line_type, text, y in lines:
        try:
            if line_type == 'title':
                # 제목 중앙 정렬 (이모지 포함)
                draw_text_with_emoji(draw, text, (width//2, 30), title_font, emoji_font, 'center', '#ffffff')
                
            elif line_type == 'date':
                # 날짜 우측 정렬
                bbox = draw.textbbox((0, 0), text, font=small_font)
                x = width - margin - (bbox[2] - bbox[0])
                draw.text((x, y), text, fill='#6b7280', font=small_font)
                
            elif line_type == 'divider':
                # 구분선
                draw.line([margin, y, width-margin, y], fill='#e5e7eb', width=2)
                
            elif line_type == 'h1':
                h1_font = get_multilingual_font(22, bold=True, lang_code=lang_code) or title_font
                draw_text_with_emoji(draw, text, (margin, y), h1_font, emoji_font, 'left', '#10b981')
                # 헤딩 밑줄
                draw.line([margin, y+32, margin+300, y+32], fill='#10b981', width=3)
                
            elif line_type == 'h2':
                draw_text_with_emoji(draw, text, (margin, y), heading_font, emoji_font, 'left', '#047857')
                
            elif line_type == 'h3':
                h3_font = get_multilingual_font(18, bold=True, lang_code=lang_code) or heading_font
                draw_text_with_emoji(draw, text, (margin, y), h3_font, emoji_font, 'left', '#1f2937')
                
            elif line_type == 'bullet':
                # 불릿 포인트
                draw.text((margin, y), "•", fill='#10b981', font=text_font)
                draw_text_with_emoji(draw, text, (margin + 20, y), text_font, emoji_font, 'left', '#374151')
                
            elif line_type == 'bold':
                bold_font = get_multilingual_font(16, bold=True, lang_code=lang_code) or text_font
                draw_text_with_emoji(draw, text, (margin, y), bold_font, emoji_font, 'left', '#dc2626')
                
            elif line_type == 'text':
                draw_text_with_emoji(draw, text, (margin, y), text_font, emoji_font, 'left', '#374151')
                
            elif line_type == 'footer':
                # 푸터 텍스트 중앙 정렬 및 자동 줄바꿈
                wrapped_footer = textwrap.wrap(text, width=100)
                footer_y = y
                for line in wrapped_footer:
                    bbox = draw.textbbox((0, 0), line, font=small_font)
                    x = (width - (bbox[2] - bbox[0])) // 2
                    draw.text((x, footer_y), line, fill='#6b7280', font=small_font)
                    footer_y += 20 # 줄 간격

        except Exception as e:
            # 개별 텍스트 렌더링 실패 시 건너뜀
            print(f"⚠️ 텍스트 렌더링 오류: {e}")
            continue
    
    return img


def render_report_html(file_name: str, rule_analysis: dict, ai_analysis: dict, title="🏠 AI 부동산 계약서 종합 분석 리포트") -> str:
    score = rule_analysis.get("safety_score", -1)
    if score >= 80:
        grade_text = f"매우 안전 ({score}점)"
    elif score >= 50:
        grade_text = f"보통 ({score}점)"
    elif score >= 0:
        grade_text = f"위험 ({score}점)"
    else:
        grade_text = "점수 계산 오류"

    alerts = rule_analysis.get("alerts", [])
    alerts_html = []
    for a in alerts:
        cls = "alert"
        # 키워드 기반으로 alert 종류 자동 판별
        if any(keyword in a for keyword in ["치명", "🚨", "반드시"]):
            cls += " critical"
        elif any(keyword in a for keyword in ["위험", "경고", "⚠️", "주의"]):
            cls += " warn"
        else: # "권장", "✅", "💡" 등
            cls += " ok"
        
        # ::before 아이콘을 사용하므로, 텍스트에서 이모티콘은 제거
        clean_alert_text = re.sub(r'^[🚨⚠️✅💡🕵️]+', '', a).strip()
        alerts_html.append(f'<div class="{cls}">{md_to_html(clean_alert_text)}</div>')
        
    alerts_html = "\n".join(alerts_html) if alerts_html else '<div class="alert">표시할 알림이 없습니다.</div>'

    ai_block = md_to_html(ai_analysis.get("analysis", ""))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""
    <html>
      <head>
        {EMBED_HEAD}
        <style>{EMBED_CSS}</style>
      </head>
      <body>
        <div class="report-wrap">
          <article class="report-card">
            <header class="report-header">
              <h1>{title}</h1>
              <div class="meta">
                <span class="badge">📄 {file_name}</span> 
                <span class="badge">📅 {timestamp}</span>
                <span class="badge">{grade_text}</span>
            </header>

            <section class="report-section">
              <h2>계약서 안전도 검사</h2>
              <div class="alerts">{alerts_html}</div>
            </section>

            <section class="report-section">
              <h2>AI 심층 분석</h2>
              <div>{ai_block}</div>
            </section>

            <footer class="footer-note">
              본 분석은 참고용이며 법적 효력이 없습니다. 중요한 결정 전 반드시 전문가와 상담하시기 바랍니다.
            </footer>
          </article>
        </div>
      </body>
    </html>
    """
    return html

def split_text_for_analysis(text: str, max_tokens: int = 3500) -> list:
    """
    긴 텍스트를 토큰 제한에 맞게 분할합니다.
    대략적인 토큰 계산: 1 토큰 ≈ 4 문자 (한국어 기준)
    """
    if not text:
        return []
    
    # 대략적인 토큰 수 계산 (한국어 기준)
    estimated_tokens = len(text) // 4
    
    if estimated_tokens <= max_tokens:
        return [text]
    
    # 문단 단위로 분할
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 현재 청크에 문단을 추가했을 때의 토큰 수 계산
        test_chunk = current_chunk + paragraph + "\n\n"
        test_tokens = len(test_chunk) // 4
        
        if test_tokens <= max_tokens:
            current_chunk = test_chunk
        else:
            # 현재 청크가 있으면 저장
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # 새 청크 시작
            current_chunk = paragraph + "\n\n"
    
    # 마지막 청크 추가
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def perform_ai_analysis(contract_text: str) -> dict:
    """RAG를 사용하여 계약서를 심층 분석합니다. (토큰 제한 자동 처리)"""
    # RAG 검색기(RETRIEVER)가 준비되었는지 확인
    if not RETRIEVER:
        return {"analysis": "⚠️ AI 분석 엔진(RAG)이 초기화되지 않았습니다. 프로그램을 다시 시작하거나 설정을 확인해주세요."}
    
    try:
        # 1) Groundedness Check 객체 생성
        groundedness_checker = UpstageGroundednessCheck()

        # 2) 프롬프트 정의 (간소화된 버전)
        prompt = ChatPromptTemplate.from_template(
            """당신은 한국 부동산 법률 전문가입니다. 주어진 [참고 자료]를 바탕으로 다음 [계약서]를 분석하고, 임차인에게 불리하거나 누락된 조항이 없는지 상세히 설명해주세요. 답변은 마크다운 형식으로 명확하게 정리해주세요.

[참고 자료]
{context}

[계약서]
{contract}

[분석 요청]
1. **임차인에게 불리한 조항**: 독소 조항이나 일반적으로 임차인에게 불리하게 작용할 수 있는 내용을 짚어주세요.
2. **누락된 중요 조항**: 임차인 보호를 위해 참고 자료에 근거하여 반드시 포함되어야 하지만 빠져 있는 조항이 있는지 확인해주세요.
3. **개선 방안 및 대안 제시**: 발견된 문제점에 대해 구체적으로 어떻게 수정하거나 추가하면 좋을지 대안을 제시해주세요.
4. **종합적인 법률 자문**: 계약 전반에 대한 종합적인 의견과 추가적으로 확인해야 할 사항을 알려주세요.
"""
        )

        # 3) context 텍스트화 유틸
        def docs_to_text(docs):
            try:
                return "\n\n---\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
            except Exception:
                return str(docs)

        # 4) 답변 생성 체인 (출력에 근거 인용 유도)
        chain = (
            {
                "context": RETRIEVER | RunnableLambda(docs_to_text),
                "contract": RunnablePassthrough()
            }
            | prompt
            | ChatUpstage(model="solar-pro2", reasoning_effort="high")
            | StrOutputParser()
        )

        # 5) 토큰 제한 확인 및 텍스트 분할
        estimated_tokens = len(contract_text) // 4
        print(f"📊 계약서 토큰 수: 약 {estimated_tokens} 토큰")
        
        # RAG 검색 결과의 토큰 수도 고려하여 더 낮은 임계값 사용
        if estimated_tokens > 2000:  # RAG context를 고려하여 2000으로 낮춤
            print(f"⚠️ 토큰 수 초과 감지: 약 {estimated_tokens} 토큰 (제한: 4000)")
            print("📝 텍스트를 자동으로 분할하여 분석을 진행합니다...")
            
            # 텍스트를 청크로 분할
            text_chunks = split_text_for_analysis(contract_text, max_tokens=2000)
            print(f"📋 총 {len(text_chunks)}개 청크로 분할 완료")
            
            # 각 청크별로 분석 수행
            all_analyses = []
            for i, chunk in enumerate(text_chunks, 1):
                print(f"🔍 청크 {i}/{len(text_chunks)} 분석 중...")
                try:
                    # RAG 체인 대신 단순 분석 사용
                    simple_prompt = ChatPromptTemplate.from_template(
                        """한국 부동산 법률 전문가로서 다음 [계약서]를 분석해주세요.

[계약서]
{contract}

다음 사항들을 중점적으로, 임차인의 입장에서 이해하기 쉽게 마크다운 형식으로 항목을 나누어 분석해주세요:
1. **임차인에게 불리한 조항**: 독소 조항이나 일반적으로 임차인에게 불리하게 작용할 수 있는 내용을 짚어주세요.
2. **누락된 중요 조항**: 임차인 보호를 위해 반드시 포함되어야 하지만 빠져 있는 조항이 있는지 확인해주세요.
3. **개선 방안 및 대안 제시**: 발견된 문제점에 대해 구체적으로 어떻게 수정하거나 추가하면 좋을지 대안을 제시해주세요.
4. **종합적인 법률 자문**: 계약 전반에 대한 종합적인 의견과 추가적으로 확인해야 할 사항을 알려주세요.
"""
                    )
                    simple_chain = simple_prompt | ChatUpstage(model="solar-pro2", reasoning_effort="high") | StrOutputParser()
                    chunk_result = simple_chain.invoke({"contract": chunk})
                    all_analyses.append(f"## 청크 {i} 분석 결과\n\n{chunk_result}")
                except Exception as chunk_error:
                    print(f"⚠️ 청크 {i} 분석 실패: {chunk_error}")
                    all_analyses.append(f"## 청크 {i} 분석 실패\n\n오류: {chunk_error}")
            
            # 모든 분석 결과 통합
            analysis_result = "\n\n---\n\n".join(all_analyses)
            print("✅ 모든 청크 분석 완료 및 통합")
            
            # Groundedness Check는 전체 텍스트에 대해 수행
            try:
                groundedness_result = groundedness_checker.invoke({"text": contract_text})
            except Exception as ge:
                print(f"⚠️ Groundedness Check 실패: {ge}")
                groundedness_result = None
        else:
            # 일반적인 분석 수행
            print(f"✅ 토큰 수 확인: 약 {estimated_tokens} 토큰 (제한 내)")
            
            try:
                # RAG 답변과 Groundedness Check 병행 체인
                rag_chain_with_check = RunnablePassthrough.assign(
                    context=itemgetter("contract") | RunnableLambda(build_grounded_context_for_contract),
                    answer=itemgetter("contract") | chain
                ).assign(
                    groundedness=groundedness_checker
                )

                # 실행 및 결과 추출
                result_dict = rag_chain_with_check.invoke({"contract": contract_text})
                analysis_result = result_dict.get("answer", "")
                groundedness_result = result_dict.get("groundedness", None)
            except Exception as e:
                print(f"⚠️ RAG 분석 실패, 단순 분석으로 전환: {e}")
                # RAG 실패 시 단순 분석으로 전환
                simple_prompt = ChatPromptTemplate.from_template(
                    """한국 부동산 법률 전문가로서 다음 [계약서]를 분석해주세요.

[계약서]
{contract}

다음 사항들을 중점적으로, 임차인의 입장에서 이해하기 쉽게 마크다운 형식으로 항목을 나누어 분석해주세요:
1. **임차인에게 불리한 조항**: 독소 조항이나 일반적으로 임차인에게 불리하게 작용할 수 있는 내용을 짚어주세요.
2. **누락된 중요 조항**: 임차인 보호를 위해 반드시 포함되어야 하지만 빠져 있는 조항이 있는지 확인해주세요.
3. **개선 방안 및 대안 제시**: 발견된 문제점에 대해 구체적으로 어떻게 수정하거나 추가하면 좋을지 대안을 제시해주세요.
4. **종합적인 법률 자문**: 계약 전반에 대한 종합적인 의견과 추가적으로 확인해야 할 사항을 알려주세요.
"""
                )
                simple_chain = simple_prompt | ChatUpstage(model="solar-pro2", reasoning_effort="high") | StrOutputParser()
                analysis_result = simple_chain.invoke({"contract": contract_text})
                groundedness_result = None

        print("\n" + "="*50)
        print("🕵️  [계약서 분석] Groundedness Check 결과 (터미널 전용)")
        if isinstance(groundedness_result, dict):
            score = groundedness_result.get("binary_score") or groundedness_result.get("score") or groundedness_result.get("result")
            reason = groundedness_result.get("reason") or groundedness_result.get("explanation")
        else:
            # 객체나 문자열 등 다양한 형태 방어적 처리
            score = getattr(groundedness_result, "binary_score", str(groundedness_result))
            reason = getattr(groundedness_result, "reason", "")
        print(f" - 사실 기반 점수: {score} ({'근거 있음' if str(score).lower().strip() == 'grounded' else '근거 없음'})")
        if reason:
            print(f" - 이유: {reason}")
        print("="*50 + "\n")

        return {"analysis": analysis_result}
    except Exception as e:
        print(f"❌ Groundedness Check 또는 AI 분석 중 오류: {e}")
        return {"analysis": f"❌ AI 분석 중 오류 발생: {e}"}

def deepl_translate_text(text, target_lang):
    if not DEEPL_API_KEY:
        lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
        return f"[{lang_names.get(target_lang, target_lang)} 번역 기능]\n\nDeepL API 키가 설정되지 않아 실제 번역은 불가능합니다.\n\n원본 텍스트:\n{text}..."
    try:
        headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
        data = {"text": [text], "target_lang": target_lang}
        response = requests.post(DEEPL_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    except Exception as e:
        return f"번역 오류: {e}\n\n원본 텍스트:\n{text}..."

def split_text_for_tts(text, max_bytes=4500):
    if len(text.encode('utf-8')) <= max_bytes:
        return [text]
    chunks, current_chunk = [], ""
    sentences = re.split(r'(?<=[.!?다])\s+', text)
    for sentence in sentences:
        test_chunk = current_chunk + sentence + " "
        if len(test_chunk.encode('utf-8')) <= max_bytes:
            current_chunk = test_chunk
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
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

def generate_report(file_name, rule_analysis, ai_analysis):
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

def extract_clean_text_from_html(html_content: str) -> str:
    """HTML에서 순수 텍스트만 추출 (CSS, 태그 제거)"""
    # CSS 스타일 블록 완전 제거
    text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
    # script 태그도 제거
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    # head 태그 전체 제거
    text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.DOTALL)
    
    # 블록 태그를 개행으로 변환
    text = re.sub(r'</?(h1|h2|h3|h4|h5|h6|p|div|section|article|header|footer|li)>', '\n', text)
    text = re.sub(r'</?(ul|ol)>', '\n\n', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # 나머지 모든 HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # HTML 엔티티 디코딩
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    
    # 연속된 공백과 개행 정리
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 앞뒤 공백 제거 및 빈 줄 정리
    lines = [line.strip() for line in text.split('\n')]
    clean_lines = []
    for line in lines:
        if line or (clean_lines and clean_lines[-1]):  # 연속된 빈 줄 방지
            clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()

def convert_emoji_to_text(text: str) -> str:
    """이모지를 한글 텍스트로 변환 (PNG 생성용)"""
    emoji_map = {
        '🏠': '[집]', '📋': '[문서]', '🔍': '[검색]', '📊': '[차트]', 
        '💬': '[채팅]', '🤖': '[AI]', '📄': '[파일]', '📅': '[날짜]',
        '🚨': '[경고]', '⚠️': '[주의]', '✅': '[확인]', '💡': '[아이디어]',
        '🕵️': '[탐정]', '🌐': '[지구]', '🎧': '[헤드폰]', '📸': '[카메라]',
        '🗑️': '[휴지통]', '📤': '[업로드]', '🏢': '[빌딩]', '📝': '[메모]',
        '🧠': '[뇌]', '👍': '[좋아요]', '❌': '[X]', '⭐': '[별]',
        '📎': '[클립]', '🔗': '[링크]', '📌': '[핀]', '🧾': '[영수증]',
        '📘': '[파란책]', '📙': '[주황책]', '📗': '[초록책]', '📕': '[빨간책]',
        '🔥': '[불]', '✨': '[반짝임]', '📈': '[상승차트]', '📉': '[하락차트]',
        '🐢': '[거북이]', '🏡': '[주택]', '💰': '[돈]', '📋': '[체크리스트]',
        '🔐': '[자물쇠]', '📜': '[계약서]', '⚖️': '[저울]', '🏛️': '[법원]'
    }
    
    for emoji, text_replacement in emoji_map.items():
        text = text.replace(emoji, text_replacement)
    
    # 남은 이모지들을 일반적인 패턴으로 제거하거나 변환
    text = re.sub(r'[\U0001F600-\U0001F64F]', '[이모지]', text)  # 감정 이모지
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '[기호]', text)    # 기호 이모지
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '[교통]', text)    # 교통 이모지
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '[국기]', text)    # 국가 이모지
    
    return text

def detect_language_code(text: str, translate_lang: str) -> str:
    """텍스트와 번역 언어를 기반으로 적절한 언어 코드 반환"""
    if translate_lang == "원본":
        # 한글이 많으면 KO, 영어가 많으면 EN 등
        if len(re.findall(r'[가-힣]', text)) > len(text) * 0.3:
            return 'KO'
        elif len(re.findall(r'[a-zA-Z]', text)) > len(text) * 0.5:
            return 'EN'
        # 일본어, 중국어 감지 로직 추가
        elif len(re.findall(r'[\u3040-\u30ff]', text)) > len(text) * 0.3: # Hiragana/Katakana
            return 'JA'
        elif len(re.findall(r'[\u4e00-\u9fff]', text)) > len(text) * 0.3: # CJK Unified Ideographs
            return 'ZH'
        else:
            return 'KO'  # 기본값
    else:
        return translate_lang

def html_to_png_downloadable(html_content: str, filename_prefix="report_html", lang_code_override: str | None = None):
    """HTML을 PNG로 저장 - 순수 텍스트만 추출하여 깔끔하게 저장 (이모지 지원)"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # HTML에서 순수 텍스트만 추출
    clean_text = extract_clean_text_from_html(html_content)
    
    # 언어 감지 (또는 호출부에서 override)
    if lang_code_override:
        lang_code = lang_code_override
    else:
        lang_code = 'KO'
        # 번역 결과에 포함된 언어명을 기반으로 언어 코드 설정
        if any(keyword in clean_text for keyword in ['Translation Result', 'English', '영어']):
            lang_code = 'EN'
        elif any(keyword in clean_text for keyword in ['翻訳結果', '日本語', '일본어']):
            lang_code = 'JA'
        elif any(keyword in clean_text for keyword in ['翻译结果', '中文', '중국어']):
            lang_code = 'ZH'
        elif any(keyword in clean_text for keyword in ['Результат перекладу', '우크라이나어']) or any(char in clean_text for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
            lang_code = 'UK'
        elif any(keyword in clean_text for keyword in ['Kết quả dịch', '베트남어']):
            lang_code = 'VI'
    
    # PIL로 깔끔한 이미지 생성 (이모지 포함)
    img = create_clean_report_image(clean_text, filename_prefix, lang_code)
    out_path = Path(tempfile.gettempdir()) / f"{filename_prefix}_{ts}.png"
    img.save(out_path, format='PNG', quality=95, optimize=True)
    return str(out_path)

def create_report_image(report_text, title="AI 계약서 분석 리포트", lang="ko"):
    """기존 PIL 이미지 생성 함수 - 사용하지 않음 (하위 호환용으로만 유지)"""
    return create_clean_report_image(report_text, "legacy_report")

def wrap_chat_html(answer_html_or_md: str, title="🤖 AI 답변"):
    content = md_to_html(answer_html_or_md)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <html>
      <head>
        {EMBED_HEAD}
        <style>{EMBED_CSS}</style>
      </head>
      <body>
        <div class="report-wrap">
          <article class="report-card">
            <header class="report-header" style="background: linear-gradient(135deg, #667eea, #764ba2);">
              <h1>{title}</h1>
              <div class="meta">📅 생성일시: {timestamp}</div>
            </header>
            <section class="report-section">
              <div>{content}</div>
            </section>
          </article>
        </div>
      </body>
    </html>
    """
    return html

# Gradio용 functions
def analyze_contract(file, progress=gr.Progress(track_tqdm=True)):
    if file is None:
        return "❌ 파일을 업로드해주세요.", "", "", ""
    try:
        progress(0.1, desc="[🐢🐢🐢🐢🪄🪄🪄🪄🪄..")
        text, status = extract_text_from_file(file.name)
        if not text:
            return f"❌ 텍스트 추출 실패: {status}", "", "", ""

        progress(0.4, desc="🐢🐢🐢🐢🪄🪄🪄🪄🪄...")
        rule_analysis = perform_rule_based_analysis(text) # <<< 임대인 조회가 포함된 함수 호출

        progress(0.7, desc="🐢🐢🐢🐢🪄🪄🪄🪄🪄.")
        ai_analysis = perform_ai_analysis(text)

        progress(0.9, desc="[🐢🐢🐢🐢🪄🪄🪄🪄🪄...")
        md_report = generate_report(os.path.basename(file.name), rule_analysis, ai_analysis)
        html_report = render_report_html(os.path.basename(file.name), rule_analysis, ai_analysis)

        return html_report, text, md_report, html_report
    except Exception as e:
        return f"❌ 분석 중 오류 발생: {str(e)}", "", "", ""

def chat_with_ai(message, history):
    """RAG와 Groundedness Check를 사용하여 법률 상담 채팅을 진행합니다."""
    if not message.strip():
        return history, ""
    
    # RAG 검색기(RETRIEVER)가 준비되었는지 확인
    if not RETRIEVER:
        err_msg = "⚠️ AI 상담 엔진(RAG)이 초기화되지 않았습니다. 프로그램을 다시 시작하거나 설정을 확인해주세요."
        history.append((message, err_msg))
        return history, ""

    try:
        # 1) Groundedness Check 객체 생성
        groundedness_checker = UpstageGroundednessCheck()

        # 2) 프롬프트 정의
        prompt = ChatPromptTemplate.from_template(
            """당신은 한국 부동산 법률 전문가입니다. 주어진 [참고 자료]를 바탕으로 사용자의 [질문]에 대해 친절하고 상세하게 답변해주세요. 답변은 마크다운 형식으로 명확하게 정리해주세요. 법적 효력이 없음을 명시하고 전문가 상담을 권유하는 내용을 포함해주세요. 답변은 곧바로 핵심 내용부터 시작하고, RAG/참고 자료를 언급하는 서문이나 메타 문구(예: '주어진 [참고 자료]와 관련 법률을 바탕으로 답변드립니다')는 절대 포함하지 마세요. 각 핵심 주장마다 관련 [참고 자료]나 조문/문구를 한두 문장으로 간략히 인용하고 따옴표로 표시하세요.

[참고 자료]
{context}

[질문]
{question}
"""
        )

        # 3) 답변 생성 체인 (출력에 근거 인용 유도)
        chain = (
            {
                "context": RETRIEVER | RunnableLambda(docs_to_text),
                "question": RunnablePassthrough()
            }
            | prompt
            | ChatUpstage(model="solar-pro2", reasoning_effort="high")
            | StrOutputParser()
        )

        # 4) RAG 답변과 Groundedness Check 병행
        rag_chain_with_check = RunnablePassthrough.assign(
            context=itemgetter("question") | RunnableLambda(build_grounded_context_for_question),
            answer=itemgetter("question") | chain
        ).assign(
            groundedness=groundedness_checker
        )

        # 5) 실행 및 터미널 출력
        result_dict = rag_chain_with_check.invoke({"question": message})
        response = result_dict.get("answer", "")
        groundedness_result = result_dict.get("groundedness", None)

        print("\n" + "="*50)
        print("💬 [실시간 상담] Groundedness Check 결과 (터미널 전용)")
        if isinstance(groundedness_result, dict):
            score = groundedness_result.get("binary_score") or groundedness_result.get("score") or groundedness_result.get("result")
            reason = groundedness_result.get("reason") or groundedness_result.get("explanation")
        else:
            score = getattr(groundedness_result, "binary_score", str(groundedness_result))
            reason = getattr(groundedness_result, "reason", "")
        print(f" - 사실 기반 점수: {score} ({'근거 있음' if str(score).lower().strip() == 'grounded' else '근거 없음'})")
        if reason:
            print(f" - 이유: {reason}")
        print("="*50 + "\n")

        history.append((message, response))
        return history, response
    except Exception as e:
        err_msg = f"❌ 답변 생성 중 오류: {e}"
        history.append((message, err_msg))
        return history, err_msg

# Gradio 인터페이스 
def create_interface():
    with gr.Blocks(
        title="AI 부동산 법률 비서",
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="green")
    ) as interface:
        # 메인 헤더 - 아름다운 디자인
        with gr.Row():
            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
                text-align: center;
                position: relative;
                overflow: hidden;
            ">
                <!-- 배경 장식 요소 -->
                <div style="
                    position: absolute;
                    top: -50px;
                    right: -50px;
                    width: 100px;
                    height: 100px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 50%;
                "></div>
                <div style="
                    position: absolute;
                    bottom: -30px;
                    left: -30px;
                    width: 60px;
                    height: 60px;
                    background: rgba(255, 255, 255, 0.08);
                    border-radius: 50%;
                "></div>
                
                <!-- 메인 제목 -->
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 1rem;
                    margin-bottom: 1rem;
                ">
                    <div style="
                        font-size: 3rem;
                        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
                    "></div>
                    <div>
                        <h1 style="
                            color: white;
                            margin: 0;
                            font-size: 4.5rem;
                            font-weight: 700;
                            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                            letter-spacing: -0.02em;
                        ">Shellter</h1>
                    </div>
                    <div style="
                        font-size: 3rem;
                        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
                    "></div>
                </div>
                
                <!-- 서브 타이틀 -->
                <p style="
                    color: rgba(255, 255, 255, 0.95);
                    margin: 0;
                    font-size: 1.2rem;
                    font-weight: 400;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                    line-height: 1.5;
                ">🐢 부동산 계약서의 숨은 위험을 찾아내고, 당신의 소중한 자산을 지켜드립니다.</p>
                
                <!-- 기능 하이라이트 -->
                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 1.5rem;
                    margin-top: 1.5rem;
                    flex-wrap: wrap;
                ">
                </div>
            </div>
            """)

        # 메인 컨텐츠 
        with gr.Row():
            # 왼쪽: 파일 분석
            with gr.Column(scale=6):
                gr.Markdown("## 📋 계약서 분석")
                file_input = gr.File(
                    label="📁 계약서 파일을 업로드하세요",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx", ".hwp", ".txt"],
                    type="filepath"
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 초기화", variant="secondary")
                
                with gr.Accordion("🌐 분석 결과 부가기능", open=False):
                    with gr.Row():
                        analysis_translate_lang = gr.Dropdown(
                            choices=[
                                ("원본", "원본"),
                                ("영어 🇺🇸", "EN"),
                                ("일본어 🇯🇵", "JA"),
                                ("중국어 🇨🇳", "ZH"),
                                ("우크라이나어 🇺🇦", "UK"),
                                ("베트남어 🇻🇳", "VI")
                            ],
                            label="언어 선택",
                            value="원본"
                        )
                        analysis_speech_lang = gr.Dropdown(
                            choices=[
                                ("한국어 🇰🇷", "KO"),
                                ("영어 🇺🇸", "EN"),
                                ("일본어 🇯🇵", "JA"),
                                ("중국어 🇨🇳", "ZH"),
                                ("우크라이나어 🇺🇦", "UK"),
                                ("베트남어 🇻🇳", "VI")
                            ],
                            label="음성 언어",
                            value="KO"
                        )
                    with gr.Row():
                        analysis_translate_btn = gr.Button("🌎 번역하기", variant="secondary")
                        analysis_speech_btn = gr.Button("🎧 음성 생성", variant="secondary")
                        analysis_image_btn = gr.Button("📁 PNG 저장", variant="secondary")
                        analysis_translate_png_btn = gr.Button("🗂️ 번역 PNG", variant="secondary")

                    # 번역 결과를 HTML로 표시
                    analysis_translation_output = gr.HTML(label="번역된 분석 결과", visible=True)
                    with gr.Row():
                        analysis_audio_output = gr.Audio(label="분석 결과 음성", type="filepath")
                        analysis_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                    with gr.Row():
                        analysis_image_download = gr.File(label="📁 생성된 리포트 PNG", visible=True)
                        analysis_translate_image_download = gr.File(label="🗂️ 번역 리포트 PNG", visible=True)

            # 오른쪽: 채팅 및 보고서
            with gr.Column(scale=6):
                gr.Markdown("## 🕵️ AI 분석 & 상담")
                with gr.Tabs() as tabs:
                    with gr.TabItem("📊 분석 보고서", id=0):
                        analysis_output_html = gr.HTML(
                           value="<div style='text-align: center; padding: 40px; color: #6b7280; border: 2px dashed #e5e7eb; border-radius: 12px;'><p>📤 파일을 업로드하고 <b>[🔍 분석 시작]</b> 버튼을 클릭하세요.</p></div>"
                        )
                    
                    with gr.TabItem("💬 실시간 상담", id=1):
                        chatbot = gr.Chatbot(
                            value=[(None, "안녕하세요! 부동산 관련 질문이 있으시면 언제든 물어보세요.")],
                            height=400, show_label=False, container=True, show_copy_button=True,
                            bubble_full_width=False,
                        )
                        msg_input = gr.Textbox(placeholder="부동산 관련 질문을 입력하세요...", show_label=False, container=False)
                        send_btn = gr.Button("📤 전송", variant="primary")
                        gr.Examples(
                            [
                                "전세 계약 시 주의사항은 뭔가요?",
                                "전세집에 하자(곰팡이, 누수 등)가 생기면 어떻게 하나요?",
                                "집주인이 바뀌면 계약서를 다시 써야 하나요?",
                                "전세 만기인데 보증금을 못 준다 하면 어떻게 해야 하나요?",
                                "이 집이 경매에 넘어갔다고 하는데 어떻게 하나요?",
                                "전세보증보험, 지금이라도 가입하면 보호받을 수 있나요?"
                            ],
                            inputs=msg_input, label="💡 질문 예시"
                        )
                        with gr.Accordion("🌐 채팅 답변 부가기능", open=False):
                            with gr.Row():
                                chat_translate_lang = gr.Dropdown(
                                    choices=[
                                        ("원본", "원본"),
                                        ("영어 🇺🇸", "EN"),
                                        ("일본어 🇯🇵", "JA"),
                                        ("중국어 🇨🇳", "ZH"),
                                        ("우크라이나어 🇺🇦", "UK"),
                                        ("베트남어 🇻🇳", "VI")
                                    ],
                                    label="번역 언어",
                                    value="원본"
                                )
                                chat_speech_lang = gr.Dropdown(
                                    choices=[
                                        ("한국어 🇰🇷", "KO"),
                                        ("영어 🇺🇸", "EN"),
                                        ("일본어 🇯🇵", "JA"),
                                        ("중국어 🇨🇳", "ZH"),
                                        ("우크라이나어 🇺🇦", "UK"),
                                        ("베트남어 🇻🇳", "VI")
                                    ],
                                    label="음성 언어",
                                    value="KO"
                                )
                            with gr.Row():
                                chat_translate_btn = gr.Button("🌎 번역", variant="secondary")
                                chat_speech_btn = gr.Button("🎧 음성", variant="secondary")
                                chat_image_btn = gr.Button("📁 PNG 저장", variant="secondary")
                                chat_translate_png_btn = gr.Button("🗂️  번역 PNG", variant="secondary")
                            # 채팅 번역 결과도 HTML로 표시
                            chat_translation_output = gr.HTML(label="번역된 답변", visible=True)
                            chat_audio_output = gr.Audio(label="답변 음성", type="filepath")
                            chat_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                            with gr.Row():
                                chat_image_download = gr.File(label="📁 답변 PNG", visible=True)
                                chat_translate_image_download = gr.File(label="🗂️  번역 답변 PNG", visible=True)
                        

                    


        # 상태 관리
        extracted_text = gr.State("")
        analysis_report_md = gr.State("")
        analysis_report_html_state = gr.State("")
        last_chat_response = gr.State("")
        analysis_translated_text = gr.State("")
        chat_translated_text = gr.State("")

        # 번역 함수 (HTML 포함)
        def translate_analysis_with_html(report_md, lang):
            if not report_md.strip():
                return "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 분석 결과가 없습니다.</div>", ""
            if lang == "원본":
                return create_translated_html(report_md, "원본 분석 결과"), report_md
            translated = deepl_translate_text(report_md, lang)
            lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            title = f"{lang_names.get(lang, lang)} 번역 결과"
            return create_translated_html(translated, title), translated

        def translate_chat_with_html(last_resp, lang):
            if not last_resp.strip():
                return "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 답변이 없습니다.</div>", ""
            if lang == "원본":
                return create_translated_html(last_resp, "원본 답변"), last_resp
            translated = deepl_translate_text(last_resp, lang)
            lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            title = f"{lang_names.get(lang, lang)} 번역 답변"
            return create_translated_html(translated, title), translated

        # 번역 PNG 저장 함수들
        def save_analysis_translation_png(translated_text, translate_lang):
            if not translated_text.strip():
                return None
            # 번역 텍스트를 HTML로 감싸고, HTML→텍스트 정제→PNG 파이프라인 재사용
            lang_title = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            html = create_translated_html(translated_text, f"{lang_title.get(translate_lang, translate_lang)} 번역 결과")
            # 파일명에 언어 코드 포함
            code = translate_lang if translate_lang in {"EN","JA","ZH","UK","VI"} else "ORIG"
            # HTML→텍스트 정제 내부에서 이모지 한글 변환(convert_emoji_to_text) 수행됨
            return html_to_png_downloadable(html, filename_prefix=f"analysis_translation_{code}", lang_code_override=(translate_lang if translate_lang != "원본" else "KO"))

        def save_chat_translation_png(translated_text, translate_lang):
            if not translated_text.strip():
                return None
            # 번역 텍스트를 HTML로 감싸고, HTML→텍스트 정제→PNG 파이프라인 재사용
            lang_title = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            html = create_translated_html(translated_text, f"{lang_title.get(translate_lang, translate_lang)} 번역 답변")
            # 파일명에 언어 코드 포함
            code = translate_lang if translate_lang in {"EN","JA","ZH","UK","VI"} else "ORIG"
            # HTML→텍스트 정제 내부에서 이모지 한글 변환(convert_emoji_to_text) 수행됨
            return html_to_png_downloadable(html, filename_prefix=f"chat_translation_{code}", lang_code_override=(translate_lang if translate_lang != "원본" else "KO"))

        # 이벤트 핸들러
        def clear_all():
            empty_html = "<div style='display:flex; justify-content:center; align-items:center; height:400px; border: 2px dashed #e5e7eb; border-radius: 20px;'><p style='color:#6b7280;'>📤 파일을 업로드하고 <b>[🔍 분석 시작]</b> 버튼을 클릭하세요.</p></div>"
            empty_translation = "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 내용이 없습니다.</div>"
            return (
                None, empty_html, empty_translation, None, "", 
                [(None, "안녕하세요! 부동산 관련 질문이 있으시면 언제든 물어보세요.")], 
                None, "", "", empty_translation, "", None, None, None, None, None, "", "", gr.update(selected=0)
            )

        def analyze_and_store_report(file, progress=gr.Progress(track_tqdm=True)):
            html_report, text, md_report, html_pretty = analyze_contract(file, progress)
            return html_report, text, md_report, html_pretty, gr.update(selected=0)

        def store_chat_response(message, history):
            new_history, last_resp = chat_with_ai(message, history) # chat_with_ai는 이제 RAG 기반
            # 마지막 답변만 last_chat_response 상태에 저장
            if new_history and len(new_history) > 0:
                last_resp_text = new_history[-1][1]
            else:
                last_resp_text = ""
            return new_history, "", last_resp_text

        def generate_analysis_speech(report_md, lang, translate_lang):
            if not report_md.strip():
                return None, "분석 결과가 없습니다."
            
            speech_text_to_use = report_md

            # 번역 옵션이 '원본'이 아닐 경우 번역을 먼저 시도
            if translate_lang != "원본":
                lang_code_map = {"EN": "EN", "JA": "JA", "ZH": "ZH", "UK": "UK", "VI": "VI"}
                if lang in lang_code_map:
                    translated_output = deepl_translate_text(report_md, lang)
                    if "번역 오류" not in translated_output:
                        speech_text_to_use = translated_output
            
            # 언어 코드를 직접 사용 (이미 올바른 형식)
            return google_text_to_speech(speech_text_to_use, lang)

        def save_analysis_png(report_html):
            if not report_html:
                return None
            return html_to_png_downloadable(report_html, filename_prefix="analysis_report")

        def generate_chat_speech(last_resp, lang, translate_lang):
            if not last_resp.strip():
                return None, "채팅 답변이 없습니다."

            speech_text_to_use = last_resp

            # 번역 옵션이 '원본'이 아닐 경우 번역을 먼저 시도
            if translate_lang != "원본":
                lang_code_map = {"EN": "EN", "JA": "JA", "ZH": "ZH", "UK": "UK", "VI": "VI"}
                if lang in lang_code_map:
                    translated_output = deepl_translate_text(last_resp, lang)
                    if "번역 오류" not in translated_output:
                        speech_text_to_use = translated_output

            # 언어 코드를 직접 사용 (이미 올바른 형식)
            return google_text_to_speech(speech_text_to_use, lang)

        def save_chat_png(last_resp):
            if not last_resp.strip():
                return None
            html = wrap_chat_html(last_resp, title="🤖 AI 답변")
            return html_to_png_downloadable(html, filename_prefix="chat_response")
        
        # 바인딩
        analyze_btn.click(
            fn=analyze_and_store_report,
            inputs=[file_input],
            outputs=[analysis_output_html, extracted_text, analysis_report_md, analysis_report_html_state, tabs]
        )
        clear_btn.click(
            fn=clear_all,
            outputs=[
                file_input, analysis_output_html, analysis_translation_output, analysis_audio_output,
                analysis_speech_status, chatbot, chat_audio_output, chat_speech_status,
                msg_input, chat_translation_output, extracted_text, analysis_report_md,
                analysis_image_download, chat_image_download, last_chat_response, 
                analysis_translate_image_download, chat_translate_image_download,
                analysis_translated_text, chat_translated_text, tabs
            ]
        )
        analysis_translate_btn.click(
            fn=translate_analysis_with_html, 
            inputs=[analysis_report_md, analysis_translate_lang], 
            outputs=[analysis_translation_output, analysis_translated_text]
        )
        analysis_speech_btn.click(
            fn=generate_analysis_speech, 
            inputs=[analysis_report_md, analysis_speech_lang, analysis_translate_lang], 
            outputs=[analysis_audio_output, analysis_speech_status]
        )
        analysis_image_btn.click(
            fn=save_analysis_png, 
            inputs=[analysis_report_html_state], 
            outputs=[analysis_image_download]
        )
        analysis_translate_png_btn.click(
            fn=save_analysis_translation_png,
            inputs=[analysis_translated_text, analysis_translate_lang],
            outputs=[analysis_translate_image_download]
        )
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
        chat_translate_btn.click(
            fn=translate_chat_with_html, 
            inputs=[last_chat_response, chat_translate_lang], 
            outputs=[chat_translation_output, chat_translated_text]
        )
        chat_speech_btn.click(
            fn=generate_chat_speech, 
            inputs=[last_chat_response, chat_speech_lang, chat_translate_lang], 
            outputs=[chat_audio_output, chat_speech_status]
        )
        chat_image_btn.click(
            fn=save_chat_png, 
            inputs=[last_chat_response], 
            outputs=[chat_image_download]
        )
        chat_translate_png_btn.click(
            fn=save_chat_translation_png,
            inputs=[chat_translated_text, chat_translate_lang],
            outputs=[chat_translate_image_download]
        )

    return interface

def main():
    print("🐢🐢🐢🐢 AI 부동산 법률 비서를 시작합니다...")
    
    # 1. (백그라운드 작업) 다국어 폰트 다운로드 및 설정
    setup_fonts()
    
    # 2. (백그라운드 작업) AI의 지식 베이스(Vector DB) 구축
    build_ai_brain_if_needed()

    # 3. (백그라운드 작업) RAG 검색기(Retriever) 초기화
    initialize_retriever()
    
    try:
        # 4. Gradio 인터페이스 생성 및 실행
        app = create_interface()
        print("✅ 인터페이스 생성 완료. 웹서버를 시작합니다.")
        app.launch(server_name="0.0.0.0", server_port=7860, share=True, favicon_path="./Image/logo.png")
    except Exception as e:
        print(f"❌ 서버 시작 중 오류 발생: {e}")

if __name__ == "__main__":
    main()