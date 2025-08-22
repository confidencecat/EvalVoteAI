
import os
import json
import re

import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import matplotlib.pyplot as plt

# config.py에서 공통 설정/함수 import
from config import (
    GEMINI_MODEL,
    set_korean_font,
    load_json,
    save_json,
    load_api_keys,
    sanitize_folder
)


# 메모리 관리, 시스템 메시지, 프롬프트 등 config.py에서 import
from config import (
    save_memory,
    history_str,
    last_n_history,
    SYS_SYSTEM,
    SYS_FINE,
    SUB_SYSTEM,
    SYS1_DIALOG_SYSTEM,
    SYS2_SUMMARIZER_SYSTEM,
    build_sub_prompt,
    build_vote_prompt
)
# 한글 폰트 설정
set_korean_font()




# config.py에서 유틸/상수/함수 import
from config import (
    _strip_code_fence,
    extract_json_array,
    load_api_keys,
    call_ai,
    call_ai_with_memory,
    SYS_SYSTEM,
    SYS_FINE,
    SUB_SYSTEM,
    SYS1_DIALOG_SYSTEM,
    SYS2_SUMMARIZER_SYSTEM
)

# ---------------------------------
# SYS 키워드 생성 (소그룹 단위)
# ---------------------------------

def sys_generate_keywords(sys_key, subgroup):
    target = subgroup['count'] * 4
    prompt = (
        f"[소그룹 특징] {subgroup['info']}\n"
        f"[공동 요소] {subgroup['common']}\n"
        f"[다양성 설명] {subgroup['diversity']}\n"
        f"[페르소나 수] {subgroup['count']}\n"
        f"요구 개수: {target}\n"
        f"지시: 쉼표로만 구분된 키워드를 정확히 {target}개 생성"
    )
    txt = call_ai(prompt, SYS_SYSTEM, fine=SYS_FINE, api_key=sys_key)

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
# SUB 페르소나 생성 (동기 함수) + ThreadPoolExecutor로 동시 처리
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


def create_one_persona(subgroup, pick3, sub_key, name_counts, group_dir):
    txt = call_ai(build_sub_prompt(subgroup, pick3), SUB_SYSTEM, api_key=sub_key)
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


def create_personas_for_subgroup(subgroup, keywords, sub_keys, name_counts, group_dir):
    total = subgroup['count']
    all_p = []
    K = max(1, len(sub_keys))

    # SUB API 수만큼 동시 실행 → 배치 반복 (ThreadPoolExecutor)
    for start in range(0, total, K):
        end = min(start + K, total)
        with ThreadPoolExecutor(max_workers=K) as ex:
            futures = []
            for i in range(start, end):
                pick3 = random.sample(keywords, 3) if len(keywords) >= 3 else keywords[:]
                key = sub_keys[(i - start) % K]
                futures.append(ex.submit(create_one_persona, subgroup, pick3, key, name_counts, group_dir))
            for fut in as_completed(futures):
                all_p.append(fut.result())

    print(f"<[ {subgroup['name']} ]  제작 완료>")
    return all_p


# ---------------------------------
# 질문/응답 (이유 → 숫자) - ThreadPoolExecutor로 동시 처리
# ---------------------------------

def build_vote_prompt(user_q):
    return (
        f"{user_q}\n\n"
        "출력 형식(정확히 두 줄):\n"
        "1) 한 문장 이유\n"
        "2) 숫자(1~5) 한 글자만 (1=매우 긍정/찬성 ~ 5=매우 부정/반대)"
    )


def ask_one_persona(p, sub_key, qtext):
    reply = call_ai_with_memory(build_vote_prompt(qtext), p['system'], p['file'], sub_key, memory_limit=10)
    parts = reply.strip().split('\n', 1)
    reason = parts[0].strip() if parts else ''
    num = parts[1].strip() if len(parts) > 1 else ''
    return {'name': p['name'], 'reason': reason, 'number': num}


def ask_many(personas, sub_keys, qtext):
    K = max(1, len(sub_keys))
    out = []
    for start in range(0, len(personas), K):
        end = min(start + K, len(personas))
        with ThreadPoolExecutor(max_workers=K) as ex:
            futures = []
            for i in range(start, end):
                key = sub_keys[(i - start) % K]
                futures.append(ex.submit(ask_one_persona, personas[i], key, qtext))
            for fut in as_completed(futures):
                out.append(fut.result())
    return out


# ---------------------------------
# 결과 저장 (JSON/MD/PNG)
# ---------------------------------

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
# [NEW] 기존 집단 수정 플로우 (소그룹 추가/삭제)
# ---------------------------------

def modify_group_flow():
    """NEW: 기존 페르소나 집단에 소그룹을 추가하거나 삭제합니다.
    - 삭제 시 PERSONA.json의 해당 소그룹 소속 페르소나와 각 MEMORY/*.json을 모두 제거합니다.
    - 추가 시 기존 생성 로직과 동일하게 키워드 생성 → 페르소나 생성 → PERSONA.json 갱신을 수행합니다.
    """
    sys_keys, sub_keys = load_api_keys()
    if not sub_keys:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_1.. 또는 API_1.. 형식으로 등록해 주세요.")
        return

    group_name = input("수정할 페르소나 집단 이름: ").strip()
    group_dir = os.path.join('.', sanitize_folder(group_name))
    os.makedirs(group_dir, exist_ok=True)

    persona_path = os.path.join(group_dir, 'PERSONA.json')
    personas = load_json(persona_path)

    while True:
        sel = input("1) 소그룹 추가  2) 소그룹 삭제  (엔터=종료): ").strip()
        if sel == '1':
            # 추가 입력 받기 (기존 생성 폼과 동일)
            name = input("소그룹 이름: ").strip()
            info = input("소그룹 정보(예: 중산층에 해당하는 인물들이며 ~): ").strip()
            common = input("소그룹의 필수 공동 요소: ").strip()
            count = int(input("소그룹의 페르소나 수: ").strip())
            diversity = input("소그룹의 다양성(서술형): ").strip()
            subgroup = {
                'name': name,
                'info': info,
                'common': common,
                'count': count,
                'diversity': diversity,
            }
            sys_key = (sys_keys[len(personas) % len(sys_keys)]) if sys_keys else sub_keys[0]
            kws = sys_generate_keywords(sys_key, subgroup)

            name_counts = {}
            # 기존 이름 카운트 복원
            for p in personas:
                base = p['name'].rsplit('_', 1)[0]
                n = int(p['name'].rsplit('_', 1)[1]) if '_' in p['name'] else 0
                name_counts[base] = max(name_counts.get(base, 0), n)

            new_ps = create_personas_for_subgroup(subgroup, kws, sub_keys, name_counts, group_dir)
            personas.extend(new_ps)
            save_json(persona_path, personas)
            print(f"[추가 완료] '{name}' 소그룹에서 {len(new_ps)}명 생성 및 저장")

        elif sel == '2':
            target = input("삭제할 소그룹 이름(여러 개는 콤마 구분): ").strip()
            if not target:
                continue
            targets = {t.strip() for t in target.split(',') if t.strip()}
            keep = []
            removed = []
            for p in personas:
                if p.get('subgroup') in targets:
                    # MEMORY 파일 삭제 시도
                    mem_path = p.get('file')
                    if mem_path and os.path.isfile(mem_path):
                        try:
                            os.remove(mem_path)
                        except Exception:
                            pass
                    removed.append(p)
                else:
                    keep.append(p)
            personas = keep
            save_json(persona_path, personas)
            # 빈 MEMORY 폴더 정리
            mem_dir = os.path.join(group_dir, 'MEMORY')
            try:
                if os.path.isdir(mem_dir) and not os.listdir(mem_dir):
                    shutil.rmtree(mem_dir)
            except Exception:
                pass
            print(f"[삭제 완료] 소그룹 {sorted(list(targets))} 에 속한 {len(removed)}명 삭제")

        elif sel == '':
            break
        else:
            print('잘못된 입력입니다.')


# ---------------------------------
# [NEW] SYS_1 대화 모드 → SYS_2 요약 → 자동 소그룹 생성
# ---------------------------------

def auto_generate_group_flow():
    """NEW: 4) 자동 생성 플로우
    - SYS_1과 사용자 대화(quit 입력까지). 대화 로그를 group_dir/AUTO/dialog.jsonl로 저장.
    - SYS_2가 로그를 읽어 소그룹 설계 JSON을 산출.
    - 산출 결과를 그대로 사용해 소그룹별 키워드 생성 및 페르소나 생성 자동 실행.
    """
    sys_keys, sub_keys = load_api_keys()
    if not sub_keys:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_1.. 또는 API_1.. 형식으로 등록해 주세요.")
        return

    if not sys_keys:
        print("SYS 키(SYS_1, SYS_2)가 필요합니다. .env에 SYS_1, SYS_2를 등록하세요.")
        return

    sys1_key = sys_keys[0]
    sys2_key = sys_keys[1] if len(sys_keys) >= 2 else sys_keys[0]

    group_name = input("자동 생성할 페르소나 집단 이름: ").strip()
    group_dir = os.path.join('.', sanitize_folder(group_name))
    os.makedirs(group_dir, exist_ok=True)

    auto_dir = os.path.join(group_dir, 'AUTO')
    os.makedirs(auto_dir, exist_ok=True)
    dialog_path = os.path.join(auto_dir, 'dialog.jsonl')

    # --- SYS_1 대화 모드 ---
    print("[SYS_1 대화 모드 시작] 필요 정보를 말씀해 주세요. (종료: quit)")
    genai.configure(api_key=sys1_key)
    model1 = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYS1_DIALOG_SYSTEM)
    chat1 = model1.start_chat(history=[])

    dialog = []
    # 첫 안내 메시지 생성
    first = chat1.send_message("사용자 요구 파악을 위한 첫 질문을 해주세요.")
    first_txt = first._result.candidates[0].content.parts[0].text.strip()
    print(f"SYS_1: {first_txt}")
    dialog.append({'role': 'assistant', 'content': first_txt})

    while True:
        user_in = input("YOU: ")
        if user_in.strip().lower() == 'quit':
            break
        dialog.append({'role': 'user', 'content': user_in})
        resp = chat1.send_message(user_in)
        txt = resp._result.candidates[0].content.parts[0].text.strip()
        print(f"SYS_1: {txt}")
        dialog.append({'role': 'assistant', 'content': txt})

    # 대화 로그 저장
    with open(dialog_path, 'w', encoding='utf-8') as f:
        for turn in dialog:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    print(f"[저장] 대화 로그 → {dialog_path}")

    # --- SYS_2 요약/설계 산출 ---
    transcript = history_str(dialog)
    print("[SYS_2] 대화 내용을 바탕으로 소그룹 설계를 생성합니다...")
    summary_txt = call_ai(transcript, SYS2_SUMMARIZER_SYSTEM, api_key=sys2_key)
    try:
        subgroups = extract_json_array(summary_txt)
    except Exception:
        print("[경고] SYS_2 출력 파싱 실패. 빈 소그룹 목록으로 진행합니다.")
        subgroups = []

    # --- 생성 실행 ---
    persona_path = os.path.join(group_dir, 'PERSONA.json')
    personas = load_json(persona_path)
    name_counts = {}
    for p in personas:
        base = p['name'].rsplit('_', 1)[0]
        n = int(p['name'].rsplit('_', 1)[1]) if '_' in p['name'] else 0
        name_counts[base] = max(name_counts.get(base, 0), n)

    for idx, sg in enumerate(subgroups):
        # 형식 정규화 및 기본값 보정
        subgroup = {
            'name': str(sg.get('name', f'Subgroup_{idx+1}')),
            'info': str(sg.get('info', '')),
            'common': str(sg.get('common', '')),
            'count': int(sg.get('count', 0) or 0),
            'diversity': str(sg.get('diversity', '')),
        }
        if subgroup['count'] <= 0:
            print(f"[스킵] '{subgroup['name']}' count가 0이어서 건너뜁니다.")
            continue
        sys_key = (sys_keys[idx % len(sys_keys)]) if sys_keys else sub_keys[0]
        kws = sys_generate_keywords(sys_key, subgroup)
        new_ps = create_personas_for_subgroup(subgroup, kws, sub_keys, name_counts, group_dir)
        personas.extend(new_ps)

    save_json(persona_path, personas)
    print(f"[완료] 자동 생성 종료. 현재 총 페르소나 수: {len(personas)}명")


# ---------------------------------
# 플로우 (집단 생성 / 질문 / 수정 / 자동생성) - 동기 함수들
# ---------------------------------

def ask_flow():
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
            res = ask_many(personas, sub_keys, q)
            save_vote_outputs(group_dir, q, res)
        print("-시스템 종료-")
    else:
        while True:
            q = input("> ").strip()
            if q.lower() == 'exit':
                return
            res = ask_many(personas, sub_keys, q)
            save_vote_outputs(group_dir, q, res)
            print("-응답 저장 완료-")


def create_group_flow():
    sys_keys, sub_keys = load_api_keys()
    if not sub_keys:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_1.. 또는 API_1.. 형식으로 등록해 주세요.")
        return

    # NEW: 메뉴 확장
    mode = input("1) 페르소나 집단 제작  2) 질문하기  3) 기존 집단 수정  4) 집단 자동 생성  [1/2/3/4]: ").strip()
    if mode == '2':
        ask_flow()
        return
    if mode == '3':  # NEW
        modify_group_flow()
        return
    if mode == '4':  # NEW
        auto_generate_group_flow()
        return

    # (기존) 1) 집단 제작
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

    # 소그룹은 순차 처리, 각 소그룹 내부는 ThreadPoolExecutor로 동시 생성
    for idx, sg in enumerate(subgroups):
        sys_key = (sys_keys[idx % len(sys_keys)]) if sys_keys else sub_keys[0]
        kws = sys_generate_keywords(sys_key, sg)
        personas = create_personas_for_subgroup(sg, kws, sub_keys, name_counts, group_dir)
        all_personas.extend(personas)

    save_json(os.path.join(group_dir, 'PERSONA.json'), all_personas)
    print(f"\n[완료] 모든 소그룹 제작이 종료되었습니다. ({len(all_personas)}명)")


# ---------------------------------
# 엔트리포인트 (동기)
# ---------------------------------

def main():
    create_group_flow()

if __name__ == '__main__':
    main()
