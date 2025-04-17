import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

df = pd.read_csv('dict/kss_with_naive_mnemonics.csv')

# 새로운 컬럼을 저장할 리스트
pronunciations = []

# 각 단어에 대해 웹 스크래핑 수행
for word in tqdm(df['word'], desc="단어 발음 수집 중"):
    url = f'http://aha-dic.com/View.asp?word={word}'
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # phoneticKor 클래스를 가진 span 태그 찾기
        phonetic_spans = soup.find_all('span', class_='phoneticKor')
        
        # 모든 발음을 합치기
        pronunciation = ''
        for span in phonetic_spans:
            # 원본 텍스트 가져오기
            text = span.get_text()
            # 대괄호 제거
            text = text.replace('[', '').replace(']', '')
            
            # accent 클래스를 가진 span 태그 찾기
            accent_spans = span.find_all('span', class_='accent')
            
            # accent 태그가 있는 경우 처리
            if accent_spans:
                result_text = ''
                current_pos = 0
                
                for accent_span in accent_spans:
                    accent_text = accent_span.get_text()
                    accent_pos = text.find(accent_text, current_pos)
                    
                    if accent_pos != -1:
                        # accent 앞부분 추가
                        result_text += text[current_pos:accent_pos]
                        # accent 부분을 <>로 감싸기
                        result_text += f'<{accent_text}>'
                        current_pos = accent_pos + len(accent_text)
                
                # 남은 부분 추가
                result_text += text[current_pos:]
                pronunciation += result_text
            else:
                pronunciation += text
        
        if pronunciation:
            print(f"단어: {word} -> 발음: {pronunciation}")
        else:
            print(f"단어: {word} -> 발음 정보 없음")
            
        pronunciations.append(pronunciation)
        
        # 서버에 부하를 주지 않기 위해 잠시 대기
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing word {word}: {e}")
        pronunciations.append('')

# 새로운 컬럼 추가
df['pronunciation'] = pronunciations

# 새로운 CSV 파일로 저장
df.to_csv('dict/kss_data.csv', index=False, encoding='utf-8-sig')


