# Settlix v.1.1.1

Settlix는 성토 높이와 토양 데이터를 시각화하고 분석할 수 있는 GUI 응용 프로그램입니다. 이 프로그램은 PyQt5와 pyqtgraph를 사용하여 그래픽 사용자 인터페이스를 제공하며, 사용자가 데이터를 불러오고, 수정하고, 예측할 수 있는 기능을 제공합니다.

## 기능

- CSV 파일에서 데이터 로드
- 토양 데이터 설정
- 성토 높이 설정 및 시각화
- 예측 일수 설정
- 데이터 필터링
- 모델 선택 및 설정
- 그래프 저장 (PNG 형식)
- CSV 파일로 데이터 저장

## 파일 구조

```
Settlix/
├── main.py
├── README.md
├── requirements.txt
├── resource
│   ├── data
│   ├── img
│   │   └── splash_img.png
│   ├── models
│   └── ui
│       ├── data_window.ui
│       ├── height_window.ui
│       └── main_window.ui
└── source
    ├── gui
    │   ├── _impl
    │   │   ├── data_window.py
    │   │   ├── height_window.py
    │   │   ├── main_window.py
    │   │   ├── splash_window.py
    │   │   └── __pycache__
    │   ├── __init__.py
    └── utils
        ├── _impl
        │   ├── data_loader.py
        │   ├── file_saver.py
        │   ├── graph_plotter.py
        │   ├── viewbox_sync.py
        │   ├── widget_manager.py
        │   └── __pycache__
        └── __init__.py
```

## 설치 및 실행

### 요구 사항

- Python 3.8 이상
- pip

### 설치

1. 프로젝트를 클론합니다.
    ```sh
    git clone https://github.com/yourusername/settlix.git
    cd settlix
    ```

2. 가상 환경을 생성하고 활성화합니다.
    ```sh
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. 필요 패키지를 설치합니다.
    ```sh
    pip install -r requirements.txt
    ```

### 실행

```sh
python main.py
```

## 사용 방법

1. 프로그램을 실행한 후, 시작 화면이 표시됩니다.
2. 파일을 열어 데이터를 불러옵니다.
3. 토양 데이터를 설정하고, 성토 높이와 예측 일수를 설정할 수 있습니다.
4. 설정된 데이터를 바탕으로 그래프를 확인하고, 필요시 데이터를 PNG 및 CSV 파일로 저장할 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조해주세요.