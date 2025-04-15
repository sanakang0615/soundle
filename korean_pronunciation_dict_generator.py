import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('dict/final/ko-dict-ipa.csv')

# 결과를 저장할 리스트
result_data = []

# 각 행에 대해 처리
for _, row in df.iterrows():
    word = row['word']
    # NaN 값 처리
    if pd.isna(row['pronunciation']):
        pronunciations = ['']
    else:
        pronunciations = str(row['pronunciation']).split('/')
    
    # 각 발음에 대해 처리
    for pron in pronunciations:
        # ː 제거
        pron = pron.replace('ː', '')
        
        # 결과 리스트에 추가
        result_data.append({
            'word': word,
            'pronunciation': pron
        })

# DataFrame으로 변환
result_df = pd.DataFrame(result_data)

# 중복 제거
result_df = result_df.drop_duplicates()

# 새로운 CSV 파일로 저장
result_df.to_csv('dict/final/ko-dict-pronunciation.csv', index=False, encoding='utf-8-sig')

print(f"총 {len(result_df)}개의 고유한 발음이 저장되었습니다.")