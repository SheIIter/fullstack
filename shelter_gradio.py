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

# 선택 의존성 (HTML -> PNG 변환용)
HTML2IMAGE_AVAILABLE = False
try:
    from html2image import Html2Image  
    HTML2IMAGE_AVAILABLE = True
except Exception:
    HTML2IMAGE_AVAILABLE = False

# 선택 의존성 (Markdown -> HTML 변환)
MARKDOWN_AVAILABLE = False
try:
    import markdown2  # pip install markdown2
    MARKDOWN_AVAILABLE = True
except Exception:
    MARKDOWN_AVAILABLE = False


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

# API 엔드포인트
TTS_API_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}" if GOOGLE_API_KEY else None
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# Upstage Chat API 설정
try:
    from langchain_upstage import ChatUpstage, UpstageDocumentParseLoader
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    UPSTAGE_AVAILABLE = True
except ImportError:
    UPSTAGE_AVAILABLE = False

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


# 공용: 캡처용 <head> 스니펫 (웹폰트+메타+다크모드 변수)
EMBED_HEAD = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- 가독성과 한글 안정성을 위한 Noto Sans KR -->
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
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
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    line-height: 1.7;
  }
</style>
"""

# CSS utils
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

/* 번역 결과 전용 스타일 추가 */
.translation-content {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px var(--shadow);
    line-height: 1.7;
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
.translation-content code {
    background: var(--badge-bg);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.9em;
}
"""

# style.css 파일 로드 제거 - Gradio 테마와의 충돌 방지
EMBED_CSS = DEFAULT_EMBED_CSS


def md_to_html(md_text: str) -> str:
    if not md_text:
        return ""
    if MARKDOWN_AVAILABLE:
        return markdown2.markdown(md_text, extras=["fenced-code-blocks", "tables", "break-on-newline", "spoiler"])
    
    # 폴백 
    html = md_text
    
    # 헤딩 변환
    html = re.sub(r'^###\s*(.*)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^##\s*(.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^#\s*(.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # 볼드 텍스트
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # 리스트 변환
    html = re.sub(r'^- (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # 연속된 li 태그들을 ul로 감싸기
    html = re.sub(r'(<li>.*?</li>)(?:\n(<li>.*?</li>))+', lambda m: '<ul>' + m.group(0).replace('\n', '') + '</ul>', html, flags=re.DOTALL)
    
    # 줄바꿈 처리
    html = html.replace('\n\n', '</p><p>')
    html = html.replace('\n', '<br>')
    
    # 문단 태그로 감싸기
    if not html.startswith('<'):
        html = '<p>' + html + '</p>'
    
    return html

def create_translated_html(translated_text: str, title: str = "번역된 내용") -> str:
    """번역된 텍스트를 예쁜 HTML로 변환"""
    html_content = md_to_html(translated_text)
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


#  PNG 저장 시 텍스트 깨짐 방지를 위한 폰트 로더
def get_font(size=16, bold=False):
    """
    애플리케이션에 번들된 폰트를 우선 로드합니다.
    이를 통해 어떤 환경에서도 한글이 깨지지 않도록 보장합니다.
    """
    font_name = "NotoSansKR-Bold.ttf" if bold else "NotoSansKR-Regular.ttf"
    font_path = Path(__file__).parent / font_name
    
    try:
        # 1. 번들된 폰트 사용 시도 (가장 안정적)
        return ImageFont.truetype(str(font_path), size)
    except IOError:
        print(f"경고: 번들 폰트 '{font_path}'를 찾을 수 없습니다. 시스템 폰트를 탐색합니다.")
        # 2. Windows 시스템 폰트
        try:
            return ImageFont.truetype("malgun.ttf", size)
        except IOError:
            # 3. macOS 시스템 폰트
            try:
                return ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", size)
            except IOError:
                # 4. 최후의 수단
                return ImageFont.load_default()


# HTML 보고서 생성
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
              </div>
              <div style="margin-top:16px;"><span class="grade">{grade_text}</span></div>
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


# 기존 함수들을 여기에 포함 (길어서 생략하지만 실제로는 모두 포함되어야 함)
def extract_text_from_file(file_path: str) -> tuple[str, str]:
    if not file_path or not os.path.exists(file_path):
        return "", "파일을 찾을 수 없습니다"
    file_extension = Path(file_path).suffix.lower()
    if file_extension in ['.txt', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), "성공"
        except:
            return "", "텍스트 파일 읽기 실패"
    try:
        if UPSTAGE_AVAILABLE:
            pages = UpstageDocumentParseLoader(file_path, ocr="force").load()
            extracted_text = "\n\n".join([p.page_content for p in pages])
            if not extracted_text.strip():
                return "", "텍스트 추출 실패"
            return extracted_text, "성공"
        else:
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

def perform_rule_based_analysis(contract_text: str) -> dict:
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
        return {"alerts": alerts, "safety_score": safety_score}
    except Exception as e:
        alerts.append(f"⚠️ 분석 중 오류 발생: {str(e)}")
        safety_score = -1
        return {"alerts": alerts, "safety_score": safety_score}

def perform_ai_analysis(contract_text: str) -> dict:
    try:
        if UPSTAGE_AVAILABLE and UPSTAGE_API_KEY:
            prompt = ChatPromptTemplate.from_template(
                """한국 부동산 법률 전문가로서 다음 [계약서]를 분석해주세요.

[계약서]
{contract}

다음 사항들을 중점적으로, 임차인의 입장에서 이해하기 쉽게 마크다운 형식으로 항목을 나누어 분석해주세요:
1.  **임차인에게 불리한 조항**: 독소 조항이나 일반적으로 임차인에게 불리하게 작용할 수 있는 내용을 짚어주세요.
2.  **누락된 중요 조항**: 임차인 보호를 위해 반드시 포함되어야 하지만 빠져 있는 조항이 있는지 확인해주세요.
3.  **개선 방안 및 대안 제시**: 발견된 문제점에 대해 구체적으로 어떻게 수정하거나 추가하면 좋을지 대안을 제시해주세요.
4.  **종합적인 법률 자문**: 계약 전반에 대한 종합적인 의견과 추가적으로 확인해야 할 사항(등기부등본 확인 등)을 알려주세요.
"""
            )
            chain = prompt | ChatUpstage(model="solar-pro") | StrOutputParser()
            analysis_result = chain.invoke({"contract": contract_text})
            return {"analysis": analysis_result}
        else:
            analysis_result = f"""
### 📝 계약서 간단 분석 결과

**이 분석은 Upstage API 키가 없어 샘플로 제공되는 것입니다.**

-   **계약서 개요**:
    -   계약서 길이: {len(contract_text):,}자
    -   주요 키워드: '임대차', '보증금' 등 계약의 기본 요소가 포함되어 있는지 확인합니다.

-   **⚠️ 주의가 필요한 항목 (샘플)**:
    1.  **보증금 반환 조항**: "계약 만료 시 즉시 반환한다"와 같은 명확한 문구가 있는지 확인해야 합니다.
    2.  **수선 의무**: 주요 시설(보일러, 수도 등) 고장에 대한 수리 책임이 누구에게 있는지 명시되어야 합니다.
    3.  **특약사항 검토**: 임차인에게 일방적으로 불리한 특약(예: 과도한 원상복구 의무)이 있는지 꼼꼼히 봐야 합니다.

-   **💡 권장사항**:
    -   등기부등본을 발급받아 계약서상 임대인과 실제 소유주가 일치하는지 확인하세요.
    -   계약 전 해당 주소의 전입세대열람을 통해 선순위 임차인이 있는지 확인하세요.
"""
            return {"analysis": analysis_result}
    except Exception as e:
        return {"analysis": f"분석 중 오류 발생: {str(e)}"}

def deepl_translate_text(text, target_lang):
    if not DEEPL_API_KEY:
        lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
        return f"[{lang_names.get(target_lang, target_lang)} 번역 기능]\n\nDeepL API 키가 설정되지 않아 실제 번역은 불가능합니다.\n\n원본 텍스트:\n{text[:500]}..."
    try:
        headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
        data = {"text": [text], "target_lang": target_lang}
        response = requests.post(DEEPL_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    except Exception as e:
        return f"번역 오류: {e}\n\n원본 텍스트:\n{text[:500]}..."

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

def google_text_to_speech(text, lang_code="KO"):
    if not GOOGLE_API_KEY:
        return None, "Google API 키가 설정되지 않아 음성 생성이 불가능합니다."
    text = re.sub(r"[^\w\s가-힣.,!?]", "", text) # 구두점 일부 유지
    text_chunks = split_text_for_tts(text)
    voice_map = {
        "KO": {"languageCode": "ko-KR", "name": "ko-KR-Wavenet-A"},
        "EN": {"languageCode": "en-US", "name": "en-US-Wavenet-F"},
        "JA": {"languageCode": "ja-JP", "name": "ja-JP-Wavenet-A"},
        "ZH": {"languageCode": "cmn-CN", "name": "cmn-CN-Wavenet-A"},
    }
    if lang_code not in voice_map:
        return None, f"지원하지 않는 언어: {lang_code}"
    try:
        first_chunk = text_chunks[0]
        request_body = {"input": {"text": first_chunk}, "voice": voice_map[lang_code],
                        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 0.9, "pitch": -2}}
        response = requests.post(TTS_API_URL, data=json.dumps(request_body), timeout=30)
        if response.status_code == 200:
            audio_content = base64.b64decode(response.json()['audioContent'])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_content)
                msg = "음성 생성 완료!" if len(text_chunks) == 1 else f"음성 생성 완료!"
                return tmp_file.name, msg
        else:
            return None, f"TTS 오류: {response.text}"
    except Exception as e:
        return None, f"TTS 오류: {e}"

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

# 텍스트만 추출하는 유틸리티 함수
def extract_clean_text_from_html(html_content: str) -> str:
    """HTML에서 순수 텍스트만 추출 (CSS, 태그 제거)"""
    # CSS 스타일 블록 완전 제거
    text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
    # script 태그도 제거
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    # head 태그 전체 제거
    text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.DOTALL)
    
    # 블록 태그를 개행으로 변환
    text = re.sub(r'</?(h1|h2|h3|h4|h5|h6|p|div|section|article|header|footer)>', '\n', text)
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

# 이모지를 텍스트로 변환하는 함수
def convert_emoji_to_text(text: str) -> str:
    """이모지를 한글 텍스트로 변환"""
    emoji_map = {
        '🏠': '[집]', '📋': '[문서]', '🔍': '[검색]', '📊': '[차트]', 
        '💬': '[채팅]', '🤖': '[AI]', '📄': '[파일]', '📅': '[날짜]',
        '🚨': '[경고]', '⚠️': '[주의]', '✅': '[확인]', '💡': '[아이디어]',
        '🕵️': '[탐정]', '🌍': '[지구]', '🔊': '[스피커]', '📸': '[카메라]',
        '🗑️': '[휴지통]', '📤': '[업로드]', '🏢': '[빌딩]', '📝': '[메모]',
        '🧠': '[뇌]', '👍': '[좋아요]', '❌': '[X]', '⭐': '[별]'
    }
    
    for emoji, text_replacement in emoji_map.items():
        text = text.replace(emoji, text_replacement)
    
    # 남은 이모지들을 일반적인 패턴으로 제거하거나 변환
    text = re.sub(r'[\U0001F600-\U0001F64F]', '[이모지]', text)  # 감정 이모지
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '[기호]', text)    # 기호 이모지
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '[교통]', text)    # 교통 이모지
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '[국기]', text)    # 국가 이모지
    
    return text

# HTML 보고서를 PNG로 저장하는 함수들 (개선된 버전)
def html_to_png_downloadable(html_content: str, filename_prefix="report_html"):
    """HTML을 PNG로 저장 - 순수 텍스트만 추출하여 깔끔하게 저장"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # HTML에서 순수 텍스트만 추출
    clean_text = extract_clean_text_from_html(html_content)
    
    # 이모지를 텍스트로 변환
    clean_text = convert_emoji_to_text(clean_text)
    
    # PIL로 깔끔한 이미지 생성
    img = create_clean_report_image(clean_text, filename_prefix)
    out_path = Path(tempfile.gettempdir()) / f"{filename_prefix}_{ts}.png"
    img.save(out_path, format='PNG', quality=95, optimize=True)
    return str(out_path)

def create_clean_report_image(report_text: str, report_type: str = "report") -> Image.Image:
    """깔끔한 텍스트 기반 리포트 이미지 생성 (이모지 없음, CSS 없음)"""
    width = 1200
    margin = 50
    line_height = 28
    
    # 폰트 설정
    title_font = get_font(28, bold=True)
    heading_font = get_font(20, bold=True) 
    text_font = get_font(16, bold=False)
    small_font = get_font(14, bold=False)
    
    # 텍스트 전처리 및 높이 계산
    lines = []
    current_y = margin + 60
    
    # 제목 추가
    if "analysis" in report_type or "분석" in report_type:
        title = "AI 부동산 계약서 분석 리포트"
    else:
        title = "AI 상담 답변"
    
    lines.append(('title', title, current_y))
    current_y += 50
    
    # 날짜 추가
    date_str = f"생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}"
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
            # 긴 텍스트는 자동 줄바꿈
            wrapped = textwrap.fill(text, width=80)
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
            # 긴 줄 자동 줄바꿈
            wrapped = textwrap.fill(line, width=90)
            for wrapped_line in wrapped.split('\n'):
                lines.append(('text', wrapped_line, current_y))
                current_y += line_height
    
    # 푸터 공간
    current_y += 30
    lines.append(('footer', '본 분석은 참고용이며 법적 효력이 없습니다. 중요한 결정 전 반드시 전문가와 상담하시기 바랍니다.', current_y))
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
        if line_type == 'title':
            # 제목 중앙 정렬
            bbox = draw.textbbox((0, 0), text, font=title_font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text((x, 30), text, fill='#ffffff', font=title_font)
            
        elif line_type == 'date':
            # 날짜 우측 정렬
            bbox = draw.textbbox((0, 0), text, font=small_font)
            x = width - margin - (bbox[2] - bbox[0])
            draw.text((x, y), text, fill='#6b7280', font=small_font)
            
        elif line_type == 'divider':
            # 구분선
            draw.line([margin, y, width-margin, y], fill='#e5e7eb', width=2)
            
        elif line_type == 'h1':
            draw.text((margin, y), text, fill='#10b981', font=get_font(22, bold=True))
            # 헤딩 밑줄
            draw.line([margin, y+32, margin+300, y+32], fill='#10b981', width=3)
            
        elif line_type == 'h2':
            draw.text((margin, y), text, fill='#047857', font=heading_font)
            
        elif line_type == 'h3':
            draw.text((margin, y), text, fill='#1f2937', font=get_font(18, bold=True))
            
        elif line_type == 'bullet':
            # 불릿 포인트
            draw.text((margin, y), "•", fill='#10b981', font=text_font)
            draw.text((margin + 20, y), text, fill='#374151', font=text_font)
            
        elif line_type == 'bold':
            draw.text((margin, y), text, fill='#dc2626', font=get_font(16, bold=True))
            
        elif line_type == 'text':
            draw.text((margin, y), text, fill='#374151', font=text_font)
            
        elif line_type == 'footer':
            # putter 텍스트 중앙 정렬
            bbox = draw.textbbox((0, 0), text, font=small_font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), text, fill='#6b7280', font=small_font)
    
    return img

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
        progress(0.1, desc="[1/4] 파일에서 텍스트 추출 중...")
        text, status = extract_text_from_file(file.name)
        if not text:
            return f"❌ 텍스트 추출 실패: {status}", "", "", ""
        
        progress(0.4, desc="[2/4] 규칙 기반 안전도 분석 중...")
        rule_analysis = perform_rule_based_analysis(text)
        
        progress(0.7, desc="[3/4] AI 심층 분석 진행 중...")
        ai_analysis = perform_ai_analysis(text)

        progress(0.9, desc="[4/4] 최종 보고서 생성 중...")
        md_report = generate_report(os.path.basename(file.name), rule_analysis, ai_analysis)
        html_report = render_report_html(os.path.basename(file.name), rule_analysis, ai_analysis)
        
        return html_report, text, md_report, html_report
    except Exception as e:
        return f"❌ 분석 중 오류 발생: {str(e)}", "", "", ""

def chat_with_ai(message, history):
    if not message.strip():
        return history, ""
    try:
        if UPSTAGE_AVAILABLE and UPSTAGE_API_KEY:
            prompt = ChatPromptTemplate.from_template(
                """당신은 한국 부동산 법률 전문가입니다. 사용자의 질문에 친절하고 정확하게 답변해주세요.

사용자 질문: {question}

답변 시 다음 사항을 고려해주세요:
- 한국 부동산 관련 법률과 실무에 기반한 정확한 정보 제공
- 이해하기 쉬운 용어와 구체적인 예시 포함
- 필요시 주의사항과 권장사항도 함께 안내
- 법적 효력이 없음을 명시하고 전문가 상담 권유""")
            chain = prompt | ChatUpstage(model="solar-pro") | StrOutputParser()
            response = chain.invoke({"question": message})
        else:
            responses = {
                "보증금": "보증금 관련 질문이시군요! 보증금은 계약 종료 시 반드시 반환되어야 하며, 지연 시 이자도 지급해야 합니다.",
                "전세": "전세 계약에서는 확정일자를 받고, 전입신고를 하여 대항력을 확보하는 것이 중요합니다.",
                "월세": "월세 계약에서는 임대료 인상 한도(5%)와 계약갱신청구권을 확인하세요.",
                "계약서": "계약서에는 당사자 정보, 목적물 정보, 임대조건, 특약사항이 명확히 기재되어야 합니다.",
                "임대인": "임대인의 신원을 확인하고, 등기부등본을 통해 실제 소유자인지 확인하는 것이 중요합니다."
            }
            response = "안녕하세요! 부동산 관련 질문에 답변드리겠습니다.\n\n"
            found = False
            for k, ans in responses.items():
                if k in message:
                    response += f"**{k} 관련 답변:**\n{ans}\n\n"
                    found = True
            if not found:
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
        return history, response
    except Exception as e:
        err_msg = f"❌ 답변 생성 중 오류: {str(e)}"
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
                        ">SHELLTER</h1>
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
                
                with gr.Accordion("🌍 분석 결과 부가기능", open=False):
                    with gr.Row():
                        analysis_translate_lang = gr.Dropdown(choices=["원본", "EN", "JA", "ZH", "UK", "VI"], label="언어 선택", value="원본")
                        analysis_speech_lang = gr.Dropdown(choices=["한국어", "영어", "일본어", "중국어"], label="음성 언어", value="한국어")
                    with gr.Row():
                        analysis_translate_btn = gr.Button("🌍 번역하기", variant="secondary")
                        analysis_speech_btn = gr.Button("🔊 음성 생성", variant="secondary")
                        analysis_image_btn = gr.Button("📸 PNG 저장", variant="secondary")

                    # 번역 결과를 HTML로 표시
                    analysis_translation_output = gr.HTML(label="번역된 분석 결과", visible=True)
                    with gr.Row():
                        analysis_audio_output = gr.Audio(label="분석 결과 음성", type="filepath")
                        analysis_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                    analysis_image_download = gr.File(label="📸 생성된 리포트 PNG", visible=True)

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
                            ["전세 계약 시 주의사항은?", "보증금 반환을 위한 조건은?", "월세 계약과 전세 계약의 차이점은?"],
                            inputs=msg_input, label="💡 질문 예시"
                        )
                        with gr.Accordion("🌍 채팅 답변 부가기능", open=False):
                            with gr.Row():
                                chat_translate_lang = gr.Dropdown(choices=["원본", "EN", "JA", "ZH", "UK", "VI"], label="번역 언어", value="원본")
                                chat_speech_lang = gr.Dropdown(choices=["한국어", "영어", "일본어", "중국어"], label="음성 언어", value="한국어")
                            with gr.Row():
                                chat_translate_btn = gr.Button("🌍 번역", variant="secondary")
                                chat_speech_btn = gr.Button("🔊 음성", variant="secondary")
                                chat_image_btn = gr.Button("📸 PNG 저장", variant="secondary")
                            # 채팅 번역 결과도 HTML로 표시
                            chat_translation_output = gr.HTML(label="번역된 답변", visible=True)
                            chat_audio_output = gr.Audio(label="답변 음성", type="filepath")
                            chat_speech_status = gr.Textbox(label="음성 상태", interactive=False)
                            chat_image_download = gr.File(label="📸 답변 PNG", visible=True)

        # 상태 관리
        extracted_text = gr.State("")
        analysis_report_md = gr.State("")
        analysis_report_html_state = gr.State("")
        last_chat_response = gr.State("")

        # 번역 함수 (HTML 포함)
        def translate_analysis_with_html(report_md, lang):
            if not report_md.strip():
                return "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 분석 결과가 없습니다.</div>"
            if lang == "원본":
                return create_translated_html(report_md, "원본 분석 결과")
            translated = deepl_translate_text(report_md, lang)
            lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            title = f"{lang_names.get(lang, lang)} 번역 결과"
            return create_translated_html(translated, title)

        def translate_chat_with_html(last_resp, lang):
            if not last_resp.strip():
                return "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 답변이 없습니다.</div>"
            if lang == "원본":
                return create_translated_html(last_resp, "원본 답변")
            translated = deepl_translate_text(last_resp, lang)
            lang_names = {"EN": "영어", "JA": "일본어", "ZH": "중국어", "UK": "우크라이나어", "VI": "베트남어"}
            title = f"{lang_names.get(lang, lang)} 번역 답변"
            return create_translated_html(translated, title)

        # 이벤트 핸들러
        def clear_all():
            empty_html = "<div style='display:flex; justify-content:center; align-items:center; height:400px; border: 2px dashed #e5e7eb; border-radius: 20px;'><p style='color:#6b7280;'>📤 파일을 업로드하고 <b>[🔍 분석 시작]</b> 버튼을 클릭하세요.</p></div>"
            empty_translation = "<div style='padding: 20px; text-align: center; color: #6b7280;'>번역할 내용이 없습니다.</div>"
            return (
                None, empty_html, empty_translation, None, "", 
                [(None, "안녕하세요! 부동산 관련 질문이 있으시면 언제든 물어보세요.")], 
                None, "", "", empty_translation, "", None, None, None, gr.update(selected=0)
            )

        def analyze_and_store_report(file, progress=gr.Progress(track_tqdm=True)):
            html_report, text, md_report, html_pretty = analyze_contract(file, progress)
            return html_report, text, md_report, html_pretty, gr.update(selected=0)

        def store_chat_response(message, history):
            new_history, last_resp = chat_with_ai(message, history)
            return new_history, "", last_resp
        
        def generate_analysis_speech(report_md, lang, translate_lang):
            if not report_md.strip():
                return None, "분석 결과가 없습니다."
            speech_text = report_md
            if translate_lang != "원본":
                translated_output = deepl_translate_text(report_md, translate_lang)
                if "번역 오류" not in translated_output:
                    speech_text = translated_output
            return google_text_to_speech(speech_text, {"한국어": "KO", "영어": "EN", "일본어": "JA", "중국어": "ZH"}.get(lang, "KO"))

        def save_analysis_png(report_html):
            if not report_html:
                return None
            return html_to_png_downloadable(report_html, filename_prefix="analysis_report")

        def generate_chat_speech(last_resp, lang, translate_lang):
            if not last_resp.strip():
                return None, "채팅 답변이 없습니다."
            speech_text = last_resp
            if translate_lang != "원본":
                 translated_output = deepl_translate_text(last_resp, translate_lang)
                 if "번역 오류" not in translated_output:
                     speech_text = translated_output
            return google_text_to_speech(speech_text, {"한국어": "KO", "영어": "EN", "일본어": "JA", "중국어": "ZH"}.get(lang, "KO"))

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
                analysis_image_download, chat_image_download, last_chat_response, tabs
            ]
        )

        analysis_translate_btn.click(
            fn=translate_analysis_with_html, 
            inputs=[analysis_report_md, analysis_translate_lang], 
            outputs=[analysis_translation_output]
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

        # 채팅 전송 액션
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
            outputs=[chat_translation_output]
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

    return interface


# 메인함수
def main():
    print("🐢🐢🐢🐢 AI 부동산 법률 비서를 시작합니다...")
    try:
        app = create_interface()
        print("✅ 인터페이스 생성 완료")
        # 'share'를 True로 바꾸면 외부에서 접속 가능한 공개 URL이 생성됩니다.
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        print(f"❌ 서버 시작 중 오류 발생: {e}")

if __name__ == "__main__":
    main()