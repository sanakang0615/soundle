"""
한글 발음 유사도 측정 라이브러리

이 코드는 한글 단어의 발음 유사도를 측정하기 위한 라이브러리입니다.
음성학적 특성을 고려한 가중치 레벤슈타인 거리(Weighted Levenshtein Distance)를 구현하였습니다.
"""

import re
import numpy as np

class KoreanPhoneticSimilarity:
    def __init__(self):
        # 초성, 중성, 종성 분리를 위한 유니코드 정보
        self.CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        
        # 음성학적 특성에 기반한 자음 특성 정의
        # [조음 위치, 조음 방법, 유/무성, 긴장성]
        # 조음 위치: 1=양순음, 2=치경음, 3=경구개음, 4=연구개음, 5=성문음
        # 조음 방법: 1=파열음, 2=파찰음, 3=마찰음, 4=비음, 5=유음
        # 유/무성: 0=무성, 1=유성
        # 긴장성: 0=이완, 1=긴장
        self.consonant_features = {
            'ㄱ': [4, 1, 0, 0],  # 연구개 파열음, 무성, 이완
            'ㄲ': [4, 1, 0, 1],  # 연구개 파열음, 무성, 긴장
            'ㄴ': [2, 4, 1, 0],  # 치경 비음, 유성, 이완
            'ㄷ': [2, 1, 0, 0],  # 치경 파열음, 무성, 이완
            'ㄸ': [2, 1, 0, 1],  # 치경 파열음, 무성, 긴장
            'ㄹ': [2, 5, 1, 0],  # 치경 유음, 유성, 이완
            'ㅁ': [1, 4, 1, 0],  # 양순 비음, 유성, 이완
            'ㅂ': [1, 1, 0, 0],  # 양순 파열음, 무성, 이완
            'ㅃ': [1, 1, 0, 1],  # 양순 파열음, 무성, 긴장
            'ㅅ': [2, 3, 0, 0],  # 치경 마찰음, 무성, 이완
            'ㅆ': [2, 3, 0, 1],  # 치경 마찰음, 무성, 긴장
            'ㅇ': [4, 4, 1, 0],  # 연구개 비음, 유성, 이완
            'ㅈ': [3, 2, 0, 0],  # 경구개 파찰음, 무성, 이완
            'ㅉ': [3, 2, 0, 1],  # 경구개 파찰음, 무성, 긴장
            'ㅊ': [3, 2, 0, 0],  # 경구개 파찰음, 무성, 이완(ㅎ첨가)
            'ㅋ': [4, 1, 0, 0],  # 연구개 파열음, 무성, 이완(ㅎ첨가)
            'ㅌ': [2, 1, 0, 0],  # 치경 파열음, 무성, 이완(ㅎ첨가)
            'ㅍ': [1, 1, 0, 0],  # 양순 파열음, 무성, 이완(ㅎ첨가)
            'ㅎ': [5, 3, 0, 0],  # 성문 마찰음, 무성, 이완
        }
        
        # 모음 특성 정의 [전설/후설, 고/중/저, 원순성]
        # 전설/후설: 1=전설, 2=중설, 3=후설
        # 고/중/저: 1=고, 2=중, 3=저
        # 원순성: 0=비원순, 1=원순
        self.vowel_features = {
            'ㅏ': [3, 3, 0],  # 후설 저모음 비원순
            'ㅐ': [1, 2, 0],  # 전설 중모음 비원순
            'ㅑ': [3, 3, 0],  # 후설 저모음 비원순(요화)
            'ㅒ': [1, 2, 0],  # 전설 중모음 비원순(요화)
            'ㅓ': [3, 2, 0],  # 후설 중모음 비원순
            'ㅔ': [1, 2, 0],  # 전설 중모음 비원순
            'ㅕ': [3, 2, 0],  # 후설 중모음 비원순(요화)
            'ㅖ': [1, 2, 0],  # 전설 중모음 비원순(요화)
            'ㅗ': [3, 2, 1],  # 후설 중모음 원순
            'ㅘ': [3, 3, 1],  # 후설 저모음 원순(복합)
            'ㅙ': [2, 2, 1],  # 중설 중모음 원순(복합)
            'ㅚ': [1, 2, 1],  # 전설 중모음 원순
            'ㅛ': [3, 2, 1],  # 후설 중모음 원순(요화)
            'ㅜ': [3, 1, 1],  # 후설 고모음 원순
            'ㅝ': [3, 2, 1],  # 후설 중모음 원순(복합)
            'ㅞ': [2, 2, 1],  # 중설 중모음 원순(복합)
            'ㅟ': [1, 1, 1],  # 전설 고모음 원순
            'ㅠ': [3, 1, 1],  # 후설 고모음 원순(요화)
            'ㅡ': [3, 1, 0],  # 후설 고모음 비원순
            'ㅢ': [3, 1, 0],  # 후설 고모음 비원순(복합)
            'ㅣ': [1, 1, 0],  # 전설 고모음 비원순
        }
        
        # 복합 종성에 대한 분해 정보
        self.complex_jongsung = {
            'ㄳ': ['ㄱ', 'ㅅ'],
            'ㄵ': ['ㄴ', 'ㅈ'],
            'ㄶ': ['ㄴ', 'ㅎ'],
            'ㄺ': ['ㄹ', 'ㄱ'],
            'ㄻ': ['ㄹ', 'ㅁ'],
            'ㄼ': ['ㄹ', 'ㅂ'],
            'ㄽ': ['ㄹ', 'ㅅ'],
            'ㄾ': ['ㄹ', 'ㅌ'],
            'ㄿ': ['ㄹ', 'ㅍ'],
            'ㅀ': ['ㄹ', 'ㅎ'],
            'ㅄ': ['ㅂ', 'ㅅ']
        }
    
    def decompose(self, char):
        """한글 문자를 초성, 중성, 종성으로 분해"""
        if re.match('[가-힣]', char):
            char_code = ord(char) - ord('가')
            cho = char_code // (21 * 28)
            jung = (char_code % (21 * 28)) // 28
            jong = char_code % 28
            
            return (self.CHOSUNG[cho], self.JUNGSUNG[jung], self.JONGSUNG[jong])
        else:
            # 한글이 아닌 경우 그대로 반환
            return (char, '', '')
    
    def consonant_distance(self, cons1, cons2):
        """자음 간의 거리 계산"""
        if cons1 not in self.consonant_features or cons2 not in self.consonant_features:
            return 1.0  # 한글 자음이 아닌 경우 최대 거리
        
        features1 = self.consonant_features[cons1]
        features2 = self.consonant_features[cons2]
        
        # 특성별 가중치
        weights = [0.4, 0.3, 0.15, 0.15]  # 조음위치, 조음방법, 유/무성, 긴장성
        
        distance = 0
        for i in range(len(features1)):
            if features1[i] != features2[i]:
                # 조음 위치와 방법은 값의 차이에 비례하여 거리 계산
                if i < 2:
                    distance += weights[i] * abs(features1[i] - features2[i]) / 4
                else:
                    distance += weights[i]
        
        return distance
    
    def vowel_distance(self, vowel1, vowel2):
        """모음 간의 거리 계산"""
        if vowel1 not in self.vowel_features or vowel2 not in self.vowel_features:
            return 1.0  # 한글 모음이 아닌 경우 최대 거리
        
        features1 = self.vowel_features[vowel1]
        features2 = self.vowel_features[vowel2]
        
        # 특성별 가중치
        weights = [0.4, 0.4, 0.2]  # 전설/후설, 고/중/저, 원순성
        
        distance = 0
        for i in range(len(features1)):
            if features1[i] != features2[i]:
                # 전설/후설, 고/중/저는 값의 차이에 비례하여 거리 계산
                if i < 2:
                    distance += weights[i] * abs(features1[i] - features2[i]) / 2
                else:
                    distance += weights[i]
        
        return distance
    
    def jamo_distance(self, jamo1, jamo2):
        """자모 간의 거리 계산"""
        if jamo1 == jamo2:
            return 0.0
        
        # 자음인 경우
        if jamo1 in self.consonant_features and jamo2 in self.consonant_features:
            return self.consonant_distance(jamo1, jamo2)
        
        # 모음인 경우
        elif jamo1 in self.vowel_features and jamo2 in self.vowel_features:
            return self.vowel_distance(jamo1, jamo2)
        
        # 자음과 모음 사이의 거리는 최대
        else:
            return 1.0
    
    def weighted_levenshtein(self, str1, str2):
        """가중치 레벤슈타인 거리 계산"""
        # 문자열을 초성, 중성, 종성으로 분해
        jamos1 = [self.decompose(char) for char in str1]
        jamos2 = [self.decompose(char) for char in str2]
        
        # 초성, 중성, 종성 각각에 대한 문자열 생성
        cho1 = [j[0] for j in jamos1]
        jung1 = [j[1] for j in jamos1]
        jong1 = [j[2] for j in jamos1]
        
        cho2 = [j[0] for j in jamos2]
        jung2 = [j[1] for j in jamos2]
        jong2 = [j[2] for j in jamos2]
        
        # 복합 종성 처리: 복합 종성을 개별 자음으로 확장
        expanded_cho1, expanded_jung1, expanded_jong1 = self._expand_jamos(cho1, jung1, jong1)
        expanded_cho2, expanded_jung2, expanded_jong2 = self._expand_jamos(cho2, jung2, jong2)
        
        # 각 자모열에 대해 가중치 레벤슈타인 거리 계산
        cho_dist = self._weighted_levenshtein_jamo(expanded_cho1, expanded_cho2)
        jung_dist = self._weighted_levenshtein_jamo(expanded_jung1, expanded_jung2)
        jong_dist = self._weighted_levenshtein_jamo(expanded_jong1, expanded_jong2)
        
        # 초성, 중성, 종성에 대한 가중치
        weights = [0.4, 0.4, 0.2]
        
        # 각 자모열의 최대 길이로 정규화
        max_cho_len = max(len(expanded_cho1), len(expanded_cho2))
        max_jung_len = max(len(expanded_jung1), len(expanded_jung2))
        max_jong_len = max(len(expanded_jong1), len(expanded_jong2))
        
        if max_cho_len > 0:
            cho_dist /= max_cho_len
        if max_jung_len > 0:
            jung_dist /= max_jung_len
        if max_jong_len > 0:
            jong_dist /= max_jong_len
        
        # 가중 평균 거리
        total_dist = weights[0] * cho_dist + weights[1] * jung_dist + weights[2] * jong_dist
        
        # 유사도 반환 (0: 완전히 다름, 1: 동일)
        return 1.0 - total_dist
    
    def _expand_jamos(self, cho, jung, jong):
        """복합 종성을 개별 자음으로 확장"""
        expanded_cho = []
        expanded_jung = []
        expanded_jong = []
        
        for i in range(len(cho)):
            expanded_cho.append(cho[i])
            
            if jung[i]:
                expanded_jung.append(jung[i])
            
            if jong[i] in self.complex_jongsung and jong[i] != ' ':
                # 복합 종성은 분해
                for j in self.complex_jongsung[jong[i]]:
                    expanded_jong.append(j)
            elif jong[i] != ' ':
                expanded_jong.append(jong[i])
        
        return expanded_cho, expanded_jung, expanded_jong
    
    def _weighted_levenshtein_jamo(self, seq1, seq2):
        """가중치 레벤슈타인 거리 계산 (자모 시퀀스)"""
        len1, len2 = len(seq1), len(seq2)
        
        # DP 테이블 초기화
        dp = np.zeros((len1 + 1, len2 + 1))
        
        # 첫 행과 열 초기화
        for i in range(len1 + 1):
            dp[i, 0] = i
        for j in range(len2 + 1):
            dp[0, j] = j
        
        # DP 테이블 채우기
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    # 삭제, 삽입, 대체 중 최소값 선택
                    del_cost = dp[i-1, j] + 1
                    ins_cost = dp[i, j-1] + 1
                    
                    # 음성학적 특성을 고려한 대체 비용
                    sub_cost = dp[i-1, j-1] + self.jamo_distance(seq1[i-1], seq2[j-1])
                    
                    dp[i, j] = min(del_cost, ins_cost, sub_cost)
        
        return dp[len1, len2]


# 사용 예시
def demonstrate_usage():
    kps = KoreanPhoneticSimilarity()
    
    # 테스트 케이스
    test_cases = [
        ("사과", "사과"),    # 동일 단어
        ("사과", "사카"),    # 유사 발음 (조음 위치 유사)
        ("사과", "사바"),    # 유사 발음 (조음 방법 유사)
        ("사과", "바나나"),  # 완전히 다른 단어
        ("집", "짚"),       # 경음화/격음화 차이
        ("감기", "강기"),    # 비음화 차이
        ("불", "블"),       # 모음 생략
        ("학교", "항꾜"),    # 음운 변동
        ("빵", "뿡"),       # 모음 변화
    ]
    
    print("한글 발음 유사도 측정 결과:")
    print("-" * 40)
    print(f"{'단어 1':<8} {'단어 2':<8} {'유사도':<8}")
    print("-" * 40)
    
    for word1, word2 in test_cases:
        similarity = kps.weighted_levenshtein(word1, word2)
        print(f"{word1:<8} {word2:<8} {similarity:.6f}")


if __name__ == "__main__":
    demonstrate_usage()