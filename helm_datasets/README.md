# HeLM Data Pipeline (MVP)

- `extract_frames.py`: mp4 -> outputs 안에 1Hz 프레임 저장
- `annotate_app.py`: Streamlit 이벤트 boundary 라벨링/검증/수정
- `build_helm.py`: scenarios + annotations + frames -> ShareGPT JSONL 생성(assistant 출력은 YAML)

의존성: streamlit, pyyaml, pillow, opencv-python, tqdm
