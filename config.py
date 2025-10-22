import os
import json
import re
from matplotlib import font_manager, rcParams
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_MODEL = 'gemini-2.5-flash'

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

def extract_json_array(text: str):
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

def load_api_keys():
    """SYS_KEY와 SUB_KEY 각각 하나씩만 로드합니다."""
    sys_key = os.getenv('SYS_KEY')
    sub_key = os.getenv('SUB_KEY')
    
    # 하위 호환성을 위해 기존 형식도 체크
    if not sys_key:
        sys_key = os.getenv('SYS_1')
    if not sub_key:
        sub_key = os.getenv('SUB_1')
    
    # SUB_KEY가 없으면 API_KEY로 대체
    if not sub_key:
        sub_key = os.getenv('API_KEY') or os.getenv('API_1')
    
    # SYS_KEY가 없으면 SUB_KEY를 사용
    if not sys_key and sub_key:
        sys_key = sub_key
    
    return sys_key, sub_key

def call_ai(prompt='테스트', system='지침', history=None, fine=None, api_key=None):
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
    history = last_n_history(mem_file, memory_limit)
    save_memory({'role': 'user', 'content': prompt}, mem_file)
    reply = call_ai(prompt, system, history=history, fine=None, api_key=api_key)
    save_memory({'role': 'assistant', 'content': reply}, mem_file)
    return reply

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

def set_korean_font():
    try:
        candidates = [
            'Malgun Gothic', # Windows
            'AppleGothic', # macOS
            'NanumGothic', # Linux/Windows/macOS
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

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    print(f"[DEBUG save_json] 파일 저장 중: {path}")
    print(f"[DEBUG save_json] 데이터 타입: {type(data)}, 길이: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[DEBUG save_json] 저장 완료: {path}")
    except Exception as e:
        print(f"[ERROR save_json] 저장 실패: {e}")

def sanitize_folder(name, max_len=60):
    s = re.sub(r'[^\w\s-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:max_len] if len(s) > max_len else s
