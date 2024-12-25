## 버전 맞추기
python : 3.11.0 (버전 맞추는 거 추천)

    pip install -r requirements.txt
  
가상환경 만든 후, 거기에 라이브러리 깔아서 실행하기 (버전이 많이 충돌하므로 필수)

## 코드 실행 방법

    # 일부 모달리티만 사용하면, 그 모달리티에만 가중치가 부여되고 나머지는 가중치가 0임 (0 벡터 입력되는 꼴)
    python run.py --model_path="vatt_model.h5" --text="안녕하세요" --audio="firework.wav" --video="gizmo.mp4"

    # 단일 모달리티 확인하기
    python single_run.py --modality audio --input "firework.wav"

## 현재 문제점

단일 모달리티 성능은 다 그대로 나오지만, 합친 run.py에서 text를 사용하기만 하면 joy가 99~100%퍼로 수직 상승한다. (해결 못했습니다..)
  
