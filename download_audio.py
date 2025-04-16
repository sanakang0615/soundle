import os
import requests
import pandas as pd
from urllib.parse import urljoin
from pathlib import Path

def download_audio_and_update_csv(csv_path, audio_base_url):
    """
    CSV 파일의 단어들을 기반으로 오디오 파일을 다운로드하고 상대 경로를 업데이트합니다.
    
    Args:
        csv_path (str): CSV 파일 경로
        audio_base_url (str): 오디오 파일의 기본 URL
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # audio 디렉토리 생성
    audio_dir = Path("audio")
    audio_dir.mkdir(exist_ok=True)
    
    # 결과를 저장할 리스트
    results = []
    
    for word in df['word']:
        # 단어의 첫 두 글자로 디렉토리 생성
        first_two = word[:2].lower()
        sub_dir = audio_dir / first_two
        sub_dir.mkdir(exist_ok=True)
        
        # 오디오 파일 URL 생성
        audio_filename = f"{word}_en_us_1.mp3"
        audio_url = urljoin(audio_base_url, f"{first_two}/{audio_filename}")
        
        # 파일 경로 설정
        local_path = sub_dir / audio_filename
        relative_path = str(local_path)
        
        try:
            # 파일 다운로드
            response = requests.get(audio_url)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                results.append(relative_path)
            else:
                results.append("DOWNLOAD_FAILED")
        except Exception as e:
            print(f"Error downloading {word}: {str(e)}")
            results.append("DOWNLOAD_FAILED")
    
    # CSV에 결과 추가
    df['audio_path'] = results
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")

if __name__ == "__main__":
    # 사용 예시
    csv_path = "dict/kss_with_naive_mnemonics.csv"
    audio_base_url = "https://ssl.gstatic.com/dictionary/static/pronunciation/2024-04-19/audio/"
    download_audio_and_update_csv(csv_path, audio_base_url) 