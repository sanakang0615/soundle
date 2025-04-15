import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time
from typing import Dict, Any

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_model_response(content: str) -> Dict[str, Any]:
    """
    모델의 응답이 코드 블록(```json ... ```) 형태일 경우 이를 제거하고 JSON 파싱을 수행합니다.
    """
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON 파싱 오류:", e)
        print("수신된 내용:", content)
        return {"mnemonic_keyword": None, "verbal_cue": None, "error": "JSON 파싱 오류"}

def generate_mnemonic(word, meaning):
    system_prompt = """당신은 언어 학습을 도와주는 어시스턴트입니다. 영어 단어를 기억하기 쉽게 하는 니모닉을 만들어주세요.
    각 단어에 대해 다음을 제공해주세요:
    1. mnemonic_keyword: 영어 단어와 발음이 비슷한 한글 단어나 표현
    2. verbal_cue: mnemonic_keyword와 단어의 의미를 연결하는 설명
    
    응답은 다음 JSON 형식으로 반환해주세요:
    {
        "mnemonic_keyword": "한글로 된 니모닉 키워드",
        "verbal_cue": "한글로 된 설명"
    }
    """
    
    user_prompt = f"단어: {word}\n의미: {meaning}\n\n이 단어에 대한 니모닉을 만들어주세요."
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    result = parse_model_response(content)
    return result

def main():
    # CSV 파일 읽기
    df = pd.read_csv('dict/kss.csv')
    
    # 결과를 저장할 리스트
    results = []
    
    # 각 단어에 대해 니모닉 생성
    for _, row in tqdm(df.iterrows(), total=len(df)):
        word = row['word']
        meaning = row['meaning']
        
        result = generate_mnemonic(word, meaning)
        if result:
            results.append({
                'word': word,
                'meaning': meaning,
                'mnemonic_keyword': result['mnemonic_keyword'],
                'verbal_cue': result['verbal_cue']
            })
        
        # API 호출 간 딜레이 추가 (선택사항)
        time.sleep(1)
    
    # 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)
    
    # 원본 DataFrame에 결과 추가
    df = pd.merge(df, result_df, on=['word', 'meaning'], how='left')
    
    # 결과 저장
    df.to_csv('dict/kss_with_naive_mnemonics.csv', index=False, encoding='utf-8-sig')
    print("Results saved to dict/kss_with_naive_mnemonics.csv")

if __name__ == "__main__":
    main() 