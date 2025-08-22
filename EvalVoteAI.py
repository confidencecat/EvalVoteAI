# EvalVoteAI.py
# v1.1: 그래프에 격자(표) 표시, 값 라벨 추가, 한글 폰트 자동 설정 포함
# 나머지 로직은 v1과 동일 (비동기 구조, JSON 안전 파싱, SYS/SUB 분리, 소그룹 순차·내부 병렬 등)

import os
import json
import re
import random
import asyncio
from collections import Counter

from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ---------------------------------
# 기본 설정/유틸
# ---------------------------------

load_dotenv()
GEMINI_MODEL = 'gemini-2.0-flash'


def _set_korean_font():
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


_set_korean_font()


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


def sanitize_folder(name, max_len=60):
    s = re.sub(r'[^\w\s-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:max_len] if len(s) > max_len else s


# === 모델 출력에서 JSON만 안전 추출 ===

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


# ---------------------------------
# API 키 자동 로딩 (SYS_n, SUB_n, (없으면) API_n)
# ---------------------------------

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


# ---------------------------------
# AI 호출 (few-shot / history 지원)
# ---------------------------------
import google.generativeai as genai

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


async def aio_call_ai(prompt, system, *, history=None, fine=None, api_key=None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: call_ai(prompt, system, history, fine, api_key))


def call_ai_with_memory(prompt, system, mem_file, api_key, memory_limit=10):
    history = last_n_history(mem_file, memory_limit)
    save_memory({'role': 'user', 'content': prompt}, mem_file)
    reply = call_ai(prompt, system, history=history, fine=None, api_key=api_key)
    save_memory({'role': 'assistant', 'content': reply}, mem_file)
    return reply


async def aio_call_ai_with_memory(prompt, system, mem_file, api_key, memory_limit=10):
    history = last_n_history(mem_file, memory_limit)
    save_memory({'role': 'user', 'content': prompt}, mem_file)
    reply = await aio_call_ai(prompt, system, history=history, fine=None, api_key=api_key)
    save_memory({'role': 'assistant', 'content': reply}, mem_file)
    return reply


# ---------------------------------
# SYS/SUB 시스템 메시지 & few-shot
# ---------------------------------

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
[다양성 설명] 재정/직업/가족구성/여가활동에서 분화
[페르소나 수] 3""",
        "공기업_사무직, 프리랜서_디자이너, 중소기업_기획, 1인가구, 유자녀_맞벌이, 반려동물, 독서모임, 주말_등산, 가치소비, 저축_우선, 주식_초보, 부업_관심"
    ],
    [
        """[소그룹 특징] 미국 대학 STEM 전공생
[공동 요소] 기숙사 생활
[다양성 설명] 국적/장학금/연구스타일/동아리활동 다양
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


# ---------------------------------
# SYS 키워드 생성 (소그룹 단위)
# ---------------------------------

async def sys_generate_keywords(sys_key, subgroup):
    target = subgroup['count'] * 4
    prompt = (
        f"[소그룹 특징] {subgroup['info']}\n"
        f"[공동 요소] {subgroup['common']}\n"
        f"[다양성 설명] {subgroup['diversity']}\n"
        f"[페르소나 수] {subgroup['count']}\n"
        f"요구 개수: {target}\n"
        f"지시: 쉼표로만 구분된 키워드를 정확히 {target}개 생성"
    )
    txt = await aio_call_ai(prompt, SYS_SYSTEM, fine=SYS_FINE, api_key=sys_key)

    kws_raw = [w.strip() for w in txt.split(',') if w.strip()]
    seen, uniq = set(), []
    for w in kws_raw:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    kws = uniq
    if len(kws) >= target:
        return kws[:target]
    if len(kws) == 0:
        kws = [f"키워드_{i+1}" for i in range(target)]
    else:
        while len(kws) < target:
            kws.append(random.choice(uniq))
    return kws[:target]


# ---------------------------------
# SUB 페르소나 생성
# ---------------------------------

def build_sub_prompt(subgroup, pick3):
    return (
        "아래 정보를 바탕으로 1명의 페르소나를 생성하십시오.\n"
        f"[소그룹 특징] {subgroup['info']}\n"
        f"[공동 요소] {subgroup['common']}\n"
        f"[다양성 설명] {subgroup['diversity']}\n"
        f"[핵심 키워드 3개] {', '.join(pick3)}\n"
        "반드시 지정된 JSON 배열 형식으로만 출력하십시오."
    )


async def create_one_persona(subgroup, pick3, sub_key, name_counts, group_dir):
    txt = await aio_call_ai(build_sub_prompt(subgroup, pick3), SUB_SYSTEM, api_key=sub_key)
    try:
        data = extract_json_array(txt)
        p = data[0]
    except Exception:
        p = {
            "name": "Persona",
            "mind": "다양한 관점과 합리적 사고를 중시합니다.",
            "action": "정보를 수집하고 근거를 바탕으로 판단합니다.",
            "character": "대한민국 고등학생으로서 평범한 가정에서 성장했고, 스스로 목표를 세우며 학업과 활동을 병행해 왔습니다."
        }

    base = (p.get('name') or 'Persona')
    num = name_counts.get(base, 0) + 1
    name_counts[base] = num
    idx = f"{base}_{num:03d}"

    mem_dir = os.path.join(group_dir, 'MEMORY')
    os.makedirs(mem_dir, exist_ok=True)
    mem_path = os.path.join(mem_dir, f"{idx}.json")

    p['name'] = idx
    p['file'] = mem_path
    p['keywords'] = pick3
    p['subgroup'] = subgroup['name']
    p['system'] = (
        f"너는 아래 정보를 따르는 페르소나이다.\n"
        f"가치관 : {p['mind']}\n"
        f"행동 : {p['action']}\n"
        f"특징 : {p['character']}"
    )
    return p


async def create_personas_for_subgroup(subgroup, keywords, sub_keys, name_counts, group_dir):
    total = subgroup['count']
    all_p = []
    K = max(1, len(sub_keys))

    for start in range(0, total, K):
        batch = []
        end = min(start + K, total)
        for i in range(start, end):
            pick3 = random.sample(keywords, 3) if len(keywords) >= 3 else keywords[:]
            key = sub_keys[(i - start) % K]
            batch.append(asyncio.create_task(create_one_persona(subgroup, pick3, key, name_counts, group_dir)))
        res = await asyncio.gather(*batch)
        all_p.extend(res)

    print(f"<[ {subgroup['name']} ]  제작 완료>")
    return all_p


# ---------------------------------
# 질문/응답 (비동기, 이유 → 숫자)
# ---------------------------------

def build_vote_prompt(user_q):
    return (
        f"{user_q}\n\n"
        "출력 형식(정확히 두 줄):\n"
        "1) 한 문장 이유\n"
        "2) 숫자(1~5) 한 글자만 (1=매우 긍정 ~ 5=매우 부정)"
    )


async def ask_one_persona(p, sub_key, qtext):
    reply = await aio_call_ai_with_memory(build_vote_prompt(qtext), p['system'], p['file'], sub_key, memory_limit=10)
    parts = reply.strip().split('\n', 1)
    reason = parts[0].strip() if parts else ''
    num = parts[1].strip() if len(parts) > 1 else ''
    return {'name': p['name'], 'reason': reason, 'number': num}


async def ask_many(personas, sub_keys, qtext):
    K = max(1, len(sub_keys))
    out = []
    for start in range(0, len(personas), K):
        tasks = []
        end = min(start + K, len(personas))
        for i in range(start, end):
            key = sub_keys[(i - start) % K]
            tasks.append(asyncio.create_task(ask_one_persona(personas[i], key, qtext)))
        out.extend(await asyncio.gather(*tasks))
    return out


def save_vote_outputs(group_dir, qtext, results):
    qslug = sanitize_folder(qtext, max_len=80)
    res_dir = os.path.join(group_dir, 'RESULTS', qslug)
    os.makedirs(res_dir, exist_ok=True)

    # JSON 저장
    save_json(os.path.join(res_dir, 'results.json'), results)

    # Markdown & 그래프 저장
    md_path = os.path.join(res_dir, f"{qslug}.md")
    nums = []
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {qtext}\n\n")
        f.write(f"## Persona Number : {len(results):03d}\n\n")
        for r in results:
            n = r['number']
            try:
                # 정규식으로 숫자만 추출 (예: "2) 4" -> ['2', '4'])
                nums_in_str = re.findall(r'\d', n)
                if nums_in_str:
                    # 마지막 숫자를 정수로 변환
                    n_int = int(nums_in_str[-1])
                else:
                    n_int = None
            except (ValueError, IndexError):
                n_int = None
            if n_int in [1, 2, 3, 4, 5]:
                nums.append(n_int)
            f.write(f"{r['name']} : {r['reason']}\n    : {n}\n\n---\n\n")

    counts = Counter(nums)
    values = [counts.get(i, 0) for i in range(1, 6)]

    # ---- 그래프 생성 ----
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(1, 6), values, color='skyblue')
        
        ax.set_xlabel('응답 (1=매우 긍정 ~ 5=매우 부정)')
        ax.set_ylabel('빈도')
        ax.set_title(qtext, wrap=True)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

        # 막대 위 값 라벨
        for rect in bars:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, f"{qslug}_bar.png"))
        plt.close(fig)
        print(f"'{qslug}'에 대한 그래프 저장 완료.")
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")


# ---------------------------------
# 플로우 (집단 생성 / 질문)
# ---------------------------------

async def ask_flow():
    _, sub_keys = load_api_keys()
    if not sub_keys:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_1.. 또는 API_1.. 형식으로 등록해 주세요.")
        return

    group_name = input("불러올 집단 이름: ").strip()
    group_dir = os.path.join('.', sanitize_folder(group_name))
    personas = load_json(os.path.join(group_dir, 'PERSONA.json'))
    if not personas:
        print("해당 집단의 PERSONA.json을 찾지 못했습니다.")
        return

    print(f"현재 페르소나 수: {len(personas)}")
    multi = input("질문 여러 번 하기? [y/n]: ").strip().lower() in ['y', 'yes', 't', 'true']

    if multi:
        n = int(input("질문 수: ").strip())
        qs = [input(f"{i+1}번째 질문: ").strip() for i in range(n)]
        for q in qs:
            res = await ask_many(personas, sub_keys, q)
            save_vote_outputs(group_dir, q, res)
        print("-시스템 종료-")
    else:
        while True:
            q = input("> ").strip()
            if q.lower() == 'exit':
                return
            res = await ask_many(personas, sub_keys, q)
            save_vote_outputs(group_dir, q, res)
            print("-응답 저장 완료-")


async def create_group_flow():
    sys_keys, sub_keys = load_api_keys()
    if not sub_keys:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_1.. 또는 API_1.. 형식으로 등록해 주세요.")
        return

    mode = input("1) 페르소나 집단 제작  2) 질문하기  [1/2]: ").strip()
    if mode == '2':
        await ask_flow()
        return

    group_name = input("페르소나 집단의 이름: ").strip()
    group_dir = os.path.join('.', sanitize_folder(group_name))
    os.makedirs(group_dir, exist_ok=True)

    g = int(input("소그룹 개수: ").strip())
    subgroups = []
    for i in range(g):
        print(f"\n-- 소그룹 {i+1} 입력 --")
        name = input("소그룹 이름: ").strip()
        info = input("소그룹 정보(예: 중산층에 해당하는 인물들이며 ~): ").strip()
        common = input("소그룹의 필수 공동 요소(예: 대한민국에 거주한다.): ").strip()
        count = int(input("소그룹의 페르소나 수: ").strip())
        diversity = input("소그룹의 다양성(서술형): ").strip()
        subgroups.append({
            'name': name,
            'info': info,
            'common': common,
            'count': count,
            'diversity': diversity
        })

    all_personas = []
    name_counts = {}

    for idx, sg in enumerate(subgroups):
        if sys_keys:
            sys_key = sys_keys[idx % len(sys_keys)]
        else:
            sys_key = sub_keys[0]
        kws = await sys_generate_keywords(sys_key, sg)
        personas = await create_personas_for_subgroup(sg, kws, sub_keys, name_counts, group_dir)
        all_personas.extend(personas)

    save_json(os.path.join(group_dir, 'PERSONA.json'), all_personas)
    print(f"\n[완료] 모든 소그룹 제작이 종료되었습니다. ({len(all_personas)}명)")


# ---------------------------------
# 엔트리포인트
# ---------------------------------

async def main():
    await create_group_flow()


if __name__ == '__main__':
    asyncio.run(main())
