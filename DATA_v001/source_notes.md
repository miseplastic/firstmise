# 데이터 출처와 확보 기록

- DATA 버전: DATA_v001
- 원본 URL과 제공자: https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision v1. Kaggle 제작자 표기는 Panats MP Project. 팀원 1이 같은 다운로드 경로임을 확인했다.
- 다운로드 날짜: 미기록
- 이용 조건과 확인 근거: Kaggle 표기 라이선스는 CC0. 상위 출처의 이용 조건은 미확인이므로 저장소를 공개하기 전에 확인한다.
- 원본 압축파일명과 SHA256: 미기록. 로컬 파일 합계 26,226,182바이트는 Kaggle v1의 totalBytes와 같다. 재다운로드 대조는 하지 않았다.
- 원본 보존 위치: 이 저장소의 train/ 577장, valid/ 204장과 각 폴더의 _annotations.csv. original/은 작업 중 사라졌고, 사라지기 전에 train/, valid/와 파일 783개의 SHA-256이 같았다. 작업본은 이 두 폴더를 읽기만 해서 만들었다.
- 클래스 ID와 이름의 근거: 원본 CSV의 클래스명은 Microplastic 하나다. 숫자 클래스 ID는 원본에 없다. 작업본 classes.txt의 첫 줄이므로 YOLO 클래스 ID는 0이다. CSV의 xmin, ymin, xmax, ymax는 픽셀 모서리 좌표로 보고 변환했고, 이 관례는 출처 문서로 확인하지 않았다. 왕복 좌표 불일치는 0건이다.
- 제공 train/val/test 설명과 경로 매핑: 제공 분할은 train 577장, valid 204장이고 test는 없다. 작업본은 train을 images/train, valid를 images/val로 옮겼다. 대응은 review/file_mapping.csv에 있다. test 폴더는 만들지 않았다.
- 원 논문 자료와 배포본의 관계 및 확인 근거: https://github.com/zayedarju/microplastic-detection 의 데이터는 1990/250/250장으로 구성과 형태가 달라 보인다. 원 논문은 접근이 차단되어 읽지 못했다. 같다고 확인되기 전에는 논문 수치와 비교하지 않고, 원 논문 재검증이라고 쓰지 않는다.
- 미확인 사항과 담당: 다운로드 날짜와 압축파일 해시는 미기록이다. 상위 출처 이용 조건, 파일명 접두와 번호가 식별 정보인지, 박스 좌표 관례의 출처 문서 확인은 남아 있다. 라벨 내용과 중복 쌍의 판정은 비어 있으며 팀원 1이 채운다. 그룹 확정과 분할은 팀원 2의 일이다. TEAM_AGENDA.md의 안건은 아직 결정이 없다.
