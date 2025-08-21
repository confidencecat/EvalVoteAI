import os
from dotenv import load_dotenv
import json
import sys
import concurrent.futures
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import matplotlib.pyplot as plt
import re
from collections import Counter

load_dotenv()
API_KEYS = []
i = 1
while True:
    key = os.getenv(f'API_{i}')
    if not key:
        break
    API_KEYS.append(key)
    i += 1


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_json(path, data):
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
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


def AI(prompt, system, memory_file=None, api_key=None, retries=3):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system)
    history = []
    if memory_file:
        history = load_json(memory_file)
    his_str = history_str(history)
    combined = f"{his_str}user: {prompt}"
    if memory_file:
        save_memory({'role': 'user', 'content': prompt}, memory_file)

    attempt = 0
    while True:
        try:
            resp = model.start_chat(history=[]).send_message(combined)
            text = resp._result.candidates[0].content.parts[0].text.strip()
            reply = text[11:].strip() if text.lower().startswith('assistant: ') else text
            if memory_file:
                save_memory({'role': 'assistant', 'content': reply}, memory_file)
            return reply
        except ResourceExhausted:
            attempt += 1
            if attempt > retries:
                print(f"[WARN] API quota exhausted. Skipping prompt: {prompt[:30]}...")
                return ''
            wait = 2 ** attempt
            print(f"[INFO] Quota exhausted, retrying in {wait}s...")
            time.sleep(wait)

CREATION_SYSTEM = '''
당신은 페르소나 생성 전용 AI입니다.
사용자가 입력한 '성향'을 바탕으로, 다음 4개 키만을 가진 순수 JSON 배열을 출력해야 합니다.
추가 설명이나 주석은 포함하지 마십시오.
'성향'을 따르되 극단적으로 따르지 않으며, 이름을 포함한 모든 정보를 매우 다양하게 생성하십시오.
'교정된 성향'은 '성향'안에서 다양한 선택지를 주기 때문에 이를 따르는 것이 좋을 수 있습니다.

필수 키:
1. "name": 다양한 이름 영어 이름을 사용해라. 이름에 띄어쓰기나 특수 문자를 절대 넣으면 안된다.
2. "mind": 가치관·신념 (한 문단, 구체적으로)
3. "action": 행동 패턴·습관 (구체적, 이것도 길고 구체적으로)
4. "character": 출생 배경·가족·교육·직업·중요 경험 (시간 순, 매우 구체적이면 인생 전체를 통틀어 길게 설명)

출력 예시:
[
  {
    "name": "",
    "mind": "",
    "action": "",
    "character": ""
  },
  ...
]
'''

VARIOUS_SYSTEM = '''
당신은 페르소나 다양성 생성 AI입니다.
입력 받은 문자열에 등장하지 않았지만 주어진 '성향'을 따르는 새로운 특성(다양)을 생성하십시오.
'''

def generate_personas(count, api_count, trait):
    keys = API_KEYS[:api_count]
    # print(keys)
    rounds, rem = divmod(count, api_count)
    generated = []

    existing = load_json('PERSONAS.json')
    various_input = ''
    for p in existing:
        various_input += f"[{p['name']}: {p.get('mind','')}]\n"
    various_input += f"\n위 내요은 현재까지 제작한 페르소나들의 정보이다. 이와 다른 페르소나를 생성하도록 새로운 '성향'을 작성해라.\n [기존 성향 : {trait}]\n기존 성향을 따르면서 다양성이 유지되도록 간단하게 작성해라."

    def call_api(api_key):
        various = AI(various_input, VARIOUS_SYSTEM, None, api_key)
        prompt = f"1명의 페르소나를 생성해주세요. 성향: {trait}, 교정된 성향: {various}"
        print(f"교정된 성향 : {various}")
        resp = AI(prompt, CREATION_SYSTEM, None, api_key)
        try:
            persona = json.loads(resp)[0]
        except Exception:
            start, end = resp.find('{'), resp.rfind('}')+1
            persona = json.loads(resp[start:end])
        print(f"{len(generated):03d} is generated")
        print('=======================================')
        return persona

    print()
    for _ in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=api_count) as executor:
            futures = [executor.submit(call_api, k) for k in keys]
            for fut in concurrent.futures.as_completed(futures):
                p = fut.result()
                generated.append(p)

    for k in keys[:rem]:
        p = call_api(k)
        generated.append(p)

    if len(generated) < count:
        print(f"[WARN] 생성된 페르소나: {len(generated)}/{count}")
    return generated[:count]


def process_group(group, key):
    out = []
    for p in group:
        res = AI(question, p['system'], p['file'], key)
        out.append((p['name'], res))
    return out


def save_results(question, results):
    folder = re.sub(r'[^\w\s]', '', question)
    folder = re.sub(r'\s+', '_', folder)

    MAX_LEN = 50
    if len(folder) > MAX_LEN:
        folder = folder[-MAX_LEN:]

    os.makedirs(folder, exist_ok=True)

    md_file = os.path.join(folder, f"{folder}.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# {question}\n\n")
        f.write(f"## Persona Number : {len(results):03d}\n\n")
        numbers = []
        for name, res in results:
            parts = res.split('\n', 1)
            num = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else ''
            numbers.append(int(num))
            f.write(f"{name} : {num}\n    : {reason}\n\n---\n\n")

    counts = Counter(numbers)
    values = [counts.get(i, 0) for i in range(1, 6)]
    plt.figure()
    plt.bar(range(1, 6), values)
    plt.xlabel('Response')
    plt.ylabel('Number of times')
    graph_path = os.path.join(folder, f"{folder}_그래프.png")
    plt.savefig(graph_path)
    plt.close()


if __name__ == '__main__':
    personas = load_json('PERSONAS.json')
    existing = {p['name'].split('_')[0]: 0 for p in personas}

    auto = input('페르소나 자동 생성? [y/n]: ')
    if auto.strip().lower() in ['y','yes','t','true']:
        while True:
            count   = int(input('생성할 페르소나 수: '))
            if count == 0:
                break

            api_cnt = min(int(input('use API count: ')), len(API_KEYS))
            trait   = input('페르소나 성향: ')
            raw = generate_personas(count, api_cnt, trait)

            name_counts = existing.copy()
            for p in raw:
                base = p['name']
                num  = name_counts.get(base, 0) + 1
                name_counts[base] = num
                idx = f"{base}_{num:03d}"
                p['name'] = idx
                p['file'] = f"./memory/{idx}_MEM.json"
                p['system'] = (
                    f"너는 아래 정보를 따르는 페르소나이다.\n"
                    f"가치관 : {p['mind']}\n"
                    f"행동 : {p['action']}\n"
                    f"특징 : {p['character']}"
                )
                personas.append(p)
            save_json('PERSONAS.json', personas)
    else:
        n = int(input('페르소나 수: '))
        for _ in range(n):
            name = input('name: ')
            mind = input('mind: ')
            action = input('action: ')
            character = input('character: ')
            file_path = f"./memory/{name}_MEM.json"
            system_msg = (
                f"너는 아래 정보를 따르는 페르소나이다.\n"
                f"가치관 : {mind}\n"
                f"행동 : {action}\n"
                f"특징 : {character}"
            )
            personas.append({
                'name': name,
                'mind': mind,
                'action': action,
                'character': character,
                'file': file_path,
                'system': system_msg
            })
        save_json('PERSONAS.json', personas)

    if not personas:
        sys.exit()

    api_cnt = min(len(API_KEYS), len(personas))
    keys = API_KEYS[:api_cnt]
    total = len(personas)
    per, rem = divmod(total, api_cnt)
    chunks = []
    start = 0
    for sz in [per + (1 if i < rem else 0) for i in range(api_cnt)]:
        chunks.append(personas[start:start+sz])
        start += sz
    
    print(f"현재 페르소나 수 {len(personas)}\n")

    question_many = input("질문 여러번 하기[y/n] ")

    if question_many.strip().lower() in ['y','yes','t','true']:
        questions_n = int(input("질문 수 : "))
        questions = []
        for i in range(questions_n):
            q = input(f"{i+1}번째 질문 : ")
            questions.append(q)
        respond_tf = input("응답을 출력하시겠습니까? n권장\n[y / n]  ").strip().lower() in ['y','yes','t','true']

        for q in questions:                
            question = f"[\n{q}\n]\n위 질문에 숫자를 출력하고 다음 줄에 이유를 한 문장으로 출력\n[1 ~ 5] 순서대로 [매우 긍정 ~ 매우 부정]이다."

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=api_cnt) as ex:
                futures = [ex.submit(process_group, grp, k) for grp, k in zip(chunks, keys)]
                for fut in concurrent.futures.as_completed(futures):
                    for name, res in fut.result():
                        if respond_tf:
                            print(f"==========|{name}_Persona|==========")
                            print(f"||{res}||")
                            print('===================================')
                        results.append((name, res))
            print('-응답 생성 완료-')
            save_results(q, results)
        print('-시스템 종료-')
        sys.exit()
    else:
        while True:
            q = input('> ')
            if q == 'exit':
                sys.exit()
            question = f"[\n{q}\n]\n위 질문에 숫자를 출력하고 다음 줄에 이유를 한 문장으로 출력\n[1 ~ 5] 순서대로 [매우 긍정 ~ 매우 부정]이다."
            
            respond_tf = input("응답을 출력하시겠습니까? n권장\n[y / n]  ").strip().lower() in ['y','yes','t','true']

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=api_cnt) as ex:
                futures = [ex.submit(process_group, grp, k) for grp, k in zip(chunks, keys)]
                for fut in concurrent.futures.as_completed(futures):
                    for name, res in fut.result():
                        if respond_tf:
                            print(f"==========|{name}_Persona|==========")
                            print(f"||{res}||")
                            print('===================================')
                        results.append((name, res))

            save_results(q, results)
