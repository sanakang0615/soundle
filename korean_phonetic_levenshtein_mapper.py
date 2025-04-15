import pandas as pd
from korean_phonetic_levenshtein import KoreanPhoneticSimilarity
from tqdm import tqdm

# CSV 파일 읽기
kss_df = pd.read_csv('dict/final/kss-dict-ipa-pronunciation.csv')
ko_df = pd.read_csv('dict/final/ko-dict-pronunciation.csv')

# 발음 유사도 계산기 초기화
kps = KoreanPhoneticSimilarity()

# 결과를 저장할 리스트
result_data = []

# 각 KSS 단어에 대해 처리
for _, kss_row in tqdm(kss_df.iterrows(), total=len(kss_df), desc="단어 매핑 중"):
    word = kss_row['word']
    # NaN 값 처리
    if pd.isna(kss_row['pronunciation']):
        pronunciation = ''
    else:
        pronunciation = str(kss_row['pronunciation'])
    
    # 유사도 계산 결과를 저장할 리스트
    similarities = []
    
    # 각 KO 단어와의 유사도 계산
    for _, ko_row in ko_df.iterrows():
        # NaN 값 처리
        if pd.isna(ko_row['pronunciation']):
            ko_pronunciation = ''
        else:
            ko_pronunciation = str(ko_row['pronunciation'])
            
        similarity = kps.weighted_levenshtein(pronunciation, ko_pronunciation)
        similarities.append((ko_row['word'], similarity))
    
    # 유사도 기준으로 정렬하고 상위 5개 선택
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5 = similarities[:5]
    
    # 결과 데이터에 추가
    result_row = {
        'word': word,
        'pronunciation': pronunciation
    }
    
    # 상위 5개 단어를 컬럼으로 추가
    for i, (similar_word, similarity) in enumerate(top_5, 1):
        result_row[f'similar_word_{i}'] = similar_word
        result_row[f'similarity_{i}'] = similarity
    
    result_data.append(result_row)
    
    # 매핑 결과 출력
    print(f"\n단어: {word}")
    print(f"발음: {pronunciation}")
    print("유사 단어:")
    for i, (similar_word, similarity) in enumerate(top_5, 1):
        print(f"{i}. {similar_word} (유사도: {similarity:.4f})")

# DataFrame으로 변환
result_df = pd.DataFrame(result_data)

# 새로운 CSV 파일로 저장
result_df.to_csv('dict/final/kss-dict-ipa-pronunciation-mapped-result.csv', index=False, encoding='utf-8-sig')

print(f"\n총 {len(result_df)}개의 단어에 대한 유사 단어 매핑이 완료되었습니다.")