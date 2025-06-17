# Product Categorization Service 
## Задача
Категоризовать продукт по его url
## Структура проекта
```
├── alertmanager
│   ├── alertmanager.yml
│   └── templates
│       └── telegram.tmpl
├── business_service_api.py
├── core
│   ├── config.py
│   └── logger.py
├── data
│   ├── label-studio-data
│   ├── prepared
│   ├── raw
│   └──ru_ecomm_tree_category.json
├── docker-compose.yaml
├── Dockerfile
├── dvc.lock
├── dvc.yaml
├── LICENSE
├── mlflow
│   ├── Dockerfile
│   └── requirements.txt
├── models
├── model_service_api.py
├── notebooks
├── prometheus
│   ├── alert_rules.yml
│   └── prometheus.yml
├── pyproject.toml
├── README.md
├── reports
├── requirements.txt
├── src
│   ├── apis
│   │   ├── business_service
│   │   │   ├── routers
│   │   │   │   └── router_business.py
│   │   │   └── schemas
│   │   │       └── business_service.py
│   │   └── model_service
│   │       ├── routers
│   │       │   └── router_service.py
│   │       └── schemas
│   │           └── model_service.py
│   ├── ml
│   │   ├── emeddings
│   │   │   ├── emb_image.py
│   │   │   └── emb_text.py
│   │   ├── loader.py
│   │   ├── model_registry.py
│   │   ├── predict.py
│   │   └── upload_model.py
│   └── scripts
│       ├── parse
│       │   ├── donwload_image.py
│       │   └──parse_url.py
│       ├── pipeline_maas.py
│       ├── s3
│       │   └── downloader.py
│       └── service_model.py
├── tests
│   └── tests.py
└── tracking_commit.lock
```
## Что используется?
- Mlflow - Трекинг экспериментов, версионирование моделей
- PostgresDB - (mlflow db)
- S3 - Хранение данных, моделей
- DVC - Версионирование данных, .ipynb ноутбуков
- FastAPI - API для запросов к модели и бизнесс-сервису
- Prometheus - TS база данных для хранения метрик работы API's
- Grafana - Визуализация данных полученных из Prometheus
- Alertmanager - Алертменеджер, отправляет нотификейшн в телегу
## Как поднять проект
- Создать .env файл указав все необходимые переменные для старта
- Создать файл в папке alertmanager/.bot_token
- Выполнить команду docker-compose up -d --build
