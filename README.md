# Recommender

## Recommender이란?
신규 데이터에 대한 유사도 측정 기반 추천 시비스로, 신규 데이터와 유사한 기존 데이터를 추천 합니다.

## Recommender의 목표 주요 기능
- 신규 데이터에 대한 기존 데이터 추천
- 주기적인 기존 데이터에 대한 벡터화

## 실행 방법
- docker run -it --rm -v ${ConfigFilePath}:/app/config_templates/config.yml -p 8080:8080 repo.iris.tools/datafabric/recommender:${ImageTag}

## Swagger 문서
- http://127.0.0.1:8080/recommender/docs
- http://192.168.109.254:30628/recommender/docs