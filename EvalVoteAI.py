
import os
import json
import re
import sys

import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import matplotlib.pyplot as plt

from config import (
    GEMINI_MODEL,
    set_korean_font,
    load_json,
    save_json,
    load_api_keys,
    sanitize_folder,
    save_memory,
    history_str,
    last_n_history,
    SYS_SYSTEM,
    SYS_FINE,
    SUB_SYSTEM,
    build_sub_prompt,
    build_vote_prompt,
    _strip_code_fence,
    extract_json_array,
    call_ai,
    call_ai_with_memory
)

set_korean_font()

keyword_num = 4
keyword_choices_num = 3


def sys_generate_keywords(sys_key, subgroup):
    global keyword_num
    target = subgroup['count'] * keyword_num
    print(f"[DEBUG] 키워드 생성 시작 - 목표: {target}개")
    prompt = (
        f"[소그룹 특징] {subgroup['info']}\n"
        f"[공동 요소] {subgroup['common']}\n"
        f"[다양성 설명] {subgroup['diversity']}\n"
        f"[페르소나 수] {subgroup['count']}\n"
        f"요구 개수: {target}\n"
        f"지시: 쉼표로만 구분된 키워드를 정확히 {target}개 생성"
    )
    txt = call_ai(prompt, SYS_SYSTEM, fine=SYS_FINE, api_key=sys_key)
    print(f"[DEBUG] AI 키워드 응답 받음 - 길이: {len(txt)}")

    kws_raw = [w.strip() for w in txt.split(',') if w.strip()]
    seen, uniq = set(), []
    for w in kws_raw:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    kws = uniq
    print(f"[DEBUG] 중복 제거 후 키워드 수: {len(kws)}")
    
    if len(kws) >= target:
        print(f"[DEBUG] 키워드 충분 - {target}개 선택")
        return kws[:target]
    if len(kws) == 0:
        print(f"[WARNING] 키워드 생성 실패 - 기본 키워드 사용")
        kws = [f"키워드_{i+1}" for i in range(target)]
    else:
        print(f"[DEBUG] 키워드 부족 - 랜덤 복제하여 {target}개 맞춤")
        while len(kws) < target:
            kws.append(random.choice(uniq))
    return kws[:target]


def create_one_persona(subgroup, pick_num, sub_key, name_counts, group_dir):
    print(f"[DEBUG] 페르소나 생성 시작 - 키워드: {pick_num}")
    txt = call_ai(build_sub_prompt(subgroup, pick_num), SUB_SYSTEM, api_key=sub_key)
    print(f"[DEBUG] AI 응답 받음 - 길이: {len(txt)}")
    try:
        data = extract_json_array(txt)
        p = data[0]
        print(f"[DEBUG] JSON 파싱 성공 - 이름: {p.get('name', 'Unknown')}")
    except Exception as e:
        print(f"[DEBUG] JSON 파싱 실패: {e}")
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

    if not os.path.exists(mem_path):
        save_json(mem_path, [])
        print(f"[DEBUG] 메모리 파일 생성: {mem_path}")

    p['name'] = idx
    p['file'] = mem_path
    p['keywords'] = pick_num
    p['subgroup'] = subgroup['name']
    p['system'] = (
        f"너는 아래 정보를 따르는 페르소나이다.\n"
        f"가치관 : {p['mind']}\n"
        f"행동 : {p['action']}\n"
        f"특징 : {p['character']}"
    )
    print(f"[DEBUG] 페르소나 생성 완료: {idx}")
    return p


def create_personas_for_subgroup(subgroup, keywords, sub_key, name_counts, group_dir):
    """단일 API 키를 사용하여 병렬로 페르소나를 생성합니다."""
    global keyword_choices_num
    total = subgroup['count']
    all_p = []
    
    # 병렬 처리할 배치 크기 설정 (동시에 처리할 작업 수)
    batch_size = 10  # API 제한을 고려하여 5개씩 병렬 처리
    
    print(f"[DEBUG] 소그룹 '{subgroup['name']}' - 총 {total}명 생성, 배치 크기: {batch_size}")
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"[DEBUG] 배치 처리 중: {start+1}~{end}번째 페르소나")
        
        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            futures = []
            for i in range(start, end):
                pick_num = random.sample(keywords, keyword_choices_num) if len(keywords) >= keyword_choices_num else keywords[:]
                print(f"[DEBUG] {i+1}번째 페르소나 작업 제출 - 키워드: {pick_num}")
                futures.append(ex.submit(create_one_persona, subgroup, pick_num, sub_key, name_counts, group_dir))
            
            for idx, fut in enumerate(as_completed(futures)):
                try:
                    result = fut.result()
                    all_p.append(result)
                    print(f"[DEBUG] {start+idx+1}번째 페르소나 완료: {result['name']}")
                except Exception as e:
                    print(f"[ERROR] 페르소나 생성 실패: {e}")

    print(f"<[ {subgroup['name']} ]  제작 완료 - 총 {len(all_p)}명>")
    return all_p


def ask_one_persona(p, sub_key, qtext):
    print(f"[DEBUG] 페르소나 '{p['name']}'에게 질문 중...")
    reply = call_ai_with_memory(build_vote_prompt(qtext), p['system'], p['file'], sub_key, memory_limit=10)
    print(f"[DEBUG] 페르소나 '{p['name']}' 응답 완료")
    parts = reply.strip().split('\n', 1)
    reason = parts[0].strip() if parts else ''
    num = parts[1].strip() if len(parts) > 1 else ''
    return {'name': p['name'], 'reason': reason, 'number': num}


def ask_many(personas, sub_key, qtext):
    """단일 API 키를 사용하여 병렬로 페르소나에게 질문합니다."""
    out = []
    batch_size = 20
    
    print(f"[DEBUG] 총 {len(personas)}명의 페르소나에게 질문, 배치 크기: {batch_size}")
    
    for start in range(0, len(personas), batch_size):
        end = min(start + batch_size, len(personas))
        print(f"[DEBUG] 배치 처리 중: {start+1}~{end}번째 페르소나")
        
        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            futures = []
            for i in range(start, end):
                futures.append(ex.submit(ask_one_persona, personas[i], sub_key, qtext))
            
            for idx, fut in enumerate(as_completed(futures)):
                try:
                    result = fut.result()
                    out.append(result)
                    print(f"[DEBUG] {start+idx+1}번째 응답 완료: {result['name']}")
                except Exception as e:
                    print(f"[ERROR] 페르소나 질문 실패: {e}")
    
    print(f"[DEBUG] 전체 응답 완료 - 총 {len(out)}개")
    return out



def save_vote_outputs(group_dir, qtext, results):
    qslug = sanitize_folder(qtext, max_len=80)
    res_dir = os.path.join(group_dir, 'RESULTS', qslug)
    os.makedirs(res_dir, exist_ok=True)

    save_json(os.path.join(res_dir, 'results.json'), results)

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

def modify_group_flow():
    """NEW: 기존 페르소나 집단에 소그룹을 추가하거나 삭제합니다.
    - 삭제 시 PERSONA.json의 해당 소그룹 소속 페르소나와 각 MEMORY/*.json을 모두 제거합니다.
    - 추가 시 기존 생성 로직과 동일하게 키워드 생성 → 페르소나 생성 → PERSONA.json 갱신을 수행합니다.
    """
    sys_key, sub_key = load_api_keys()
    if not sub_key:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_KEY를 등록해 주세요.")
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
            kws = sys_generate_keywords(sys_key, subgroup)

            name_counts = {}
            # 기존 이름 카운트 복원
            for p in personas:
                base = p['name'].rsplit('_', 1)[0]
                n = int(p['name'].rsplit('_', 1)[1]) if '_' in p['name'] else 0
                name_counts[base] = max(name_counts.get(base, 0), n)

            new_ps = create_personas_for_subgroup(subgroup, kws, sub_key, name_counts, group_dir)
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


def ask_flow():
    _, sub_key = load_api_keys()
    if not sub_key:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_KEY를 등록해 주세요.")
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
            res = ask_many(personas, sub_key, q)
            save_vote_outputs(group_dir, q, res)
        print("-시스템 종료-")
    else:
        while True:
            q = input("> ").strip()
            if q.lower() == 'exit':
                return
            res = ask_many(personas, sub_key, q)
            save_vote_outputs(group_dir, q, res)
            print("-응답 저장 완료-")


def create_group_flow():
    sys_key, sub_key = load_api_keys()
    if not sub_key:
        print("SUB(API) 키를 찾을 수 없습니다. .env에 SUB_KEY를 등록해 주세요.")
        return

    mode = input("1) 페르소나 집단 제작  2) 질문하기  3) 기존 집단 수정  [1/2/3]: ").strip()
    if mode == '2':
        ask_flow()
        return
    if mode == '3':
        modify_group_flow()
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

    # 소그룹은 순차 처리, 각 소그룹 내부는 ThreadPoolExecutor로 동시 생성
    for idx, sg in enumerate(subgroups):
        print(f"\n[DEBUG] 소그룹 {idx+1}/{len(subgroups)} 처리 시작: {sg['name']}")
        kws = sys_generate_keywords(sys_key, sg)
        print(f"[DEBUG] 키워드 생성 완료: {len(kws)}개")
        personas = create_personas_for_subgroup(sg, kws, sub_key, name_counts, group_dir)
        print(f"[DEBUG] 소그룹 페르소나 생성 완료: {len(personas)}명")
        all_personas.extend(personas)
        print(f"[DEBUG] 누적 페르소나 수: {len(all_personas)}명")

    persona_file = os.path.join(group_dir, 'PERSONA.json')
    print(f"\n[DEBUG] PERSONA.json 저장 시작: {persona_file}")
    print(f"[DEBUG] 저장할 페르소나 수: {len(all_personas)}명")
    
    # 페르소나 데이터 샘플 출력
    if all_personas:
        print(f"[DEBUG] 첫 번째 페르소나 샘플: name={all_personas[0].get('name')}, subgroup={all_personas[0].get('subgroup')}")
    
    save_json(persona_file, all_personas)
    
    # 저장 확인
    if os.path.exists(persona_file):
        saved_data = load_json(persona_file)
        print(f"[DEBUG] PERSONA.json 저장 확인 완료: {len(saved_data)}명 저장됨")
    else:
        print(f"[ERROR] PERSONA.json 파일이 생성되지 않았습니다!")
    
    print(f"\n[완료] 모든 소그룹 제작이 종료되었습니다. ({len(all_personas)}명)")


def main():
    create_group_flow()

if __name__ == '__main__':
    main()
