# 참조 구현

AP의 confidence 정렬·101점 보간·IoU 일대일 매칭 규칙은 다음 공식 구현을 기준으로 한다.

- https://github.com/ultralytics/yolov5/blob/9777cd00e86382dfa5a12e6244ec4f609bc4774c/utils/metrics.py
- https://github.com/ultralytics/yolov5/blob/9777cd00e86382dfa5a12e6244ec4f609bc4774c/val.py
- 해당 참조 코드의 라이선스: AGPL-3.0, https://github.com/ultralytics/yolov5/blob/9777cd00e86382dfa5a12e6244ec4f609bc4774c/LICENSE

GPU worker는 설치된 해당 commit의 process_batch·ap_per_class를 직접 호출한다. CPU 재계산은 동일 기록의 공식 AP 및 FP/FN matcher와 비교한 뒤 export 완료를 인증한다. 고정 manifest 순서를 AP 입력 순서로 사용한다. 동점 confidence의 정렬은 NumPy argsort 규칙을 따른다.
