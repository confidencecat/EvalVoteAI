# 코드 블록/JSON 추출 유틸
def _strip_code_fence(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

def extract_json_array(text: str):
    import json
    if not text or not text.strip():
        raise ValueError('empty')
    text = _strip_code_fence(text)
    l = text.find('[')
    r = text.rfind(']')
    if l != -1 and r != -1 and r > l:
        candidate = text[l:r + 1]
        return json.loads(candidate)
    l = text.find('{')
    r = text.rfind('}')
    if l != -1 and r != -1 and r > l:
        candidate = '[' + text[l:r + 1] + ']'
        return json.loads(candidate)
    raise ValueError('no json found')

# API 키 자동 로딩
def load_api_keys():
    import os
    sys_keys, sub_keys = [], []
    i = 1
    while True:
        k = os.getenv(f'SYS_{i}')
        if not k:
            break
        sys_keys.append(k)
        i += 1
    i = 1
    while True:
        k = os.getenv(f'SUB_{i}')
        if not k:
            break
        sub_keys.append(k)
        i += 1
    if not sub_keys:
        i = 1
        while True:
            k = os.getenv(f'API_{i}')
            if not k:
                break
            sub_keys.append(k)
            i += 1
    if not sys_keys and sub_keys:
        sys_keys = sub_keys[:1]
    return sys_keys, sub_keys

# AI 호출 함수
def call_ai(prompt='테스트', system='지침', history=None, fine=None, api_key=None):
    import google.generativeai as genai
    from config import GEMINI_MODEL, history_str
    if api_key is None:
        return ''
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    if fine:
        ex = ''.join([f"user: {q}\nassistant: {a}\n" for q, a in fine])
        combined = f"{ex}user: {prompt}"
    else:
        his = history_str(history if history is not None else [])
        combined = f"{his}user: {prompt}"
    resp = model.start_chat(history=[]).send_message(combined)
    txt = resp._result.candidates[0].content.parts[0].text.strip()
    return txt[9:].strip() if txt.lower().startswith('assistant:') else txt

def call_ai_with_memory(prompt, system, mem_file, api_key, memory_limit=10):
    from config import last_n_history, save_memory, call_ai
    history = last_n_history(mem_file, memory_limit)
    save_memory({'role': 'user', 'content': prompt}, mem_file)
    reply = call_ai(prompt, system, history=history, fine=None, api_key=api_key)
    save_memory({'role': 'assistant', 'content': reply}, mem_file)
    return reply
# 메모리 관리 함수
def save_memory(entry, memory_file):
    mem = load_json(memory_file)
    mem.append(entry)
    save_json(memory_file, mem)

def history_str(buf):
    s = ''
    for msg in buf:
        s += f"{msg['role']}: {msg['content']}\n"
    return s

def last_n_history(memory_file, n=10):
    mem = load_json(memory_file)
    return mem[-n:] if mem else []

# 시스템 메시지/프롬프트
SYS_SYSTEM = '''
당신은 페르소나 '집도'용 SYS AI입니다.
반드시 **쉼표(,)로만 구분된 키워드 나열 한 줄**로 출력하십시오.
출력 예시는 "키워드1, 키워드2, 키워드3" 형태입니다. 다른 설명/문장/코드는 절대 금지합니다.

입력으로 제공되는 [소그룹 특징], [공동 요소], [다양성 설명], [페르소나 수]를 바탕으로
"페르소나 수 * 4" 개수만큼 서로 중복되지 않도록, 서로 다른 특성을 반영하는 키워드를 생성하십시오.
키워드는 간결하되 의미가 분명해야 합니다.
'''

SYS_FINE = [
    [
        """[소그룹 특징] 한국 중산층 20~40대 직장인
[공동 요소] 대한민국 거주
[다양성 설명] 직업이 다양하다. 가족 구성이 다양하고 여가 활동도 다양하다.
[페르소나 수] 3""",
        "공기업_사무직, 프리랜서_디자이너, 중소기업_기획, 1인가구, 유자녀_맞벌이, 반려동물, 독서모임, 주말_등산, 가치소비, 저축_우선, 주식_초보, 부업_관심"
    ],
    [
        """[소그룹 특징] 미국 대학 STEM 전공생
[공동 요소] 기숙사 생활
[다양성 설명] 국적이 다양하고 장학금도 다양하다. 국적/장학금/연구스타일/동아리활동이 다양하다.
[페르소나 수] 2""",
        "국제학생, 장학금_전액, 랩미팅_주도, 해커톤_참여, 오픈소스_기여, 장비_매니아, 창업동아리, 튜터링"
    ],
]

SUB_SYSTEM = '''
당신은 페르소나 생성 전용 SUB AI입니다.
반드시 아래 4개 키만을 가지는 **순수 JSON 배열**로만 출력하십시오. 다른 텍스트, 주석, 설명은 금지합니다.
이름은 영문, 공백/특수문자 없이 생성하십시오.

필수 키:
1. "name": (영문, 공백·특수문자 금지)
2. "mind": 가치관·신념 (한 문단, 구체적으로)
3. "action": 행동 패턴·습관 (길고 구체적으로)
4. "character": 출생 배경·가족·교육·직업·중요 경험 (시간 순, 매우 구체적으로)

출력 예시:
[
  {"name":"","mind":"","action":"","character":""}
]
'''

SYS1_DIALOG_SYSTEM = '''
당신은 "페르소나 집단 설계 컨설턴트(SYS_1)"입니다. 사용자의 도메인/목표/제약을 대화로 파악하고,
소그룹 분할 기준, 필수 공동 요소, 다양성 축, 각 소그룹별 페르소나 수에 대해 질의응답을 통해 구체화합니다.
- 친절히 질문하고, 중간 요약을 제공합니다.
- 코드/JSON을 출력하지 말고 사람 친화적 문장으로만 답하세요.
- 사용자가 'quit'을 입력할 때까지 대화를 이어갑니다.
- 본 프로그램의 입력 규격(소그룹 이름, 소그룹 정보, 필수 공동 요소, 페르소나 수, 다양성)을 이해하고 질문을 설계합니다.
'''

SYS2_SUMMARIZER_SYSTEM = '''
당신은 "소그룹 설계 요약기(SYS_2)"입니다. 사용자와 SYS_1의 대화 로그를 입력받아
이 프로그램의 "소그룹 입력 폼"에 맞는 내용을 생성합니다. 반드시 **순수 JSON 배열**만 출력합니다.
각 원소는 다음 키를 포함해야 합니다:
- name: 소그룹 이름(간결)
- info: 소그룹 정보(예: 특성, 맥락, 배경, 목표 등 한 문단)
- common: 소그룹의 필수 공동 요소(한 문장)
- count: 페르소나 수(양의 정수)
- diversity: 다양성(서술형, 내부 차이의 축을 명확히)
예시 형식:
[
  {"name":"","info":"","common":"","count":3,"diversity":""}
]
다른 텍스트나 주석 없이 JSON 배열만 출력하세요.
'''

# 프롬프트 빌더 함수
def build_sub_prompt(subgroup, pick3):
    return (
        "아래 정보를 바탕으로 1명의 페르소나를 생성하십시오.\n"
        f"[소그룹 특징] {subgroup['info']}\n"
        f"[공동 요소] {subgroup['common']}\n"
        f"[다양성 설명] {subgroup['diversity']}\n"
        f"[핵심 키워드 3개] {', '.join(pick3)}\n"
        "반드시 지정된 JSON 배열 형식으로만 출력하십시오."
    )

def build_vote_prompt(user_q):
    return (
        f"{user_q}\n\n"
        "출력 형식(정확히 두 줄):\n"
        "1) 한 문장 이유\n"
        "2) 숫자(1~5) 한 글자만 (1=매우 긍정 ~ 5=매우 부정)"
    )
# config.py
# 공통 상수, 설정, 유틸 함수 등을 이 파일에서 관리합니다.

import os
import json
import re
from matplotlib import font_manager, rcParams
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모델명
GEMINI_MODEL = 'gemini-2.0-flash'

# 한글 폰트 설정 함수
def set_korean_font():
    try:
        candidates = [
            'Malgun Gothic',        # Windows
            'AppleGothic',          # macOS
            'NanumGothic',          # Linux/Windows/macOS
            'Noto Sans CJK KR',
            'Noto Sans KR',
            'Batang', 'Gulim'
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                rcParams['font.family'] = [name]
                break
        else:
            rcParams['font.family'] = ['DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
    except Exception:
        rcParams['font.family'] = ['DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

# JSON 로드/저장 함수
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# API 키 자동 로딩 함수
def load_api_keys():
    sys_keys, sub_keys = [], []
    i = 1
    while True:
        k = os.getenv(f'SYS_{i}')
        if not k:
            break
        sys_keys.append(k)
        i += 1
    i = 1
    while True:
        k = os.getenv(f'SUB_{i}')
        if not k:
            break
        sub_keys.append(k)
        i += 1
    if not sub_keys:
        i = 1
        while True:
            k = os.getenv(f'API_{i}')
            if not k:
                break
            sub_keys.append(k)
            i += 1
    if not sys_keys and sub_keys:
        sys_keys = sub_keys[:1]
    return sys_keys, sub_keys

# 폴더명 정제 함수
def sanitize_folder(name, max_len=60):
    s = re.sub(r'[^\w\s-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:max_len] if len(s) > max_len else s
