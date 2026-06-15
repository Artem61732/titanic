# Titanic

[Kaggle Titanic](https://www.kaggle.com/c/titanic) — предсказание `Survived` по данным пассажиров.

## Установка и запуск

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

## Команды

Все команды — из корня репозитория.

| Задача | Команда |
|--------|---------|
| ML: полный пайплайн (CV + tune + ensemble + сабмит) | `python main.py` |
| ML: только CV | `python -m ml.main --stage models` |
| ML: ансамбли | `python -m ml.main --stage ensemble` |
| ML: сабмит | `python -m ml.create_submission` |
| ML: тюнинг Optuna | `python -m ml.tune` |
| ML: тюнинг (выбор моделей) | `python -m ml.tune --models catboost lightgbm --n-trials 50` |
| DL: CV | `python -m dl.main` |
| DL: CV + grid / cosine | `python -m dl.main --grid --cosine` |
| DL: тюнинг Optuna | `python -m dl.tune --n-trials 20` |
| DL: сабмит | `python -m dl.create_submission` |
| DL: сабмит через `main.py` | `python main.py --pipeline dl` |

Сабмиты: `submission.csv` (ML), `submission_dl.csv` (DL).

Модели для Optuna задаются в `ml/config.yaml` → `tune.models`. Сабмит по умолчанию — `ensemble_diverse_voting` (`submission.model`).

Примеры:

```bash
python main.py --quick                              # быстрый ML-прогон
python -m ml.main --stage models --quick
python -m ml.tune --models catboost --n-trials 10
python -m dl.main --modes onehot embedding
python -m dl.tune --n-trials 10 --n-splits 3
python -m ml.main --stage submit
python -m dl.create_submission --mode embedding
python -m dl.create_submission train.lr=0.001 feature_mode=embedding
python main.py --pipeline all                       # ML + DL
```

Makefile: `make install`, `make run`, `make run-quick`, `make run-dl`.

## Артефакты

| Файл | Описание |
|------|----------|
| `outputs/ml/results.csv` | все CV-результаты |
| `outputs/ml/results.json` | то же в JSON |
| `outputs/ml/tune/` | best params после Optuna |
| `outputs/dl/best_params.json` | лучшие гиперпараметры DNN |
| `submission.csv` | ML-сабмит |
| `submission_dl.csv` | DL-сабмит |

Метрика — accuracy на stratified K-fold (`cv.n_splits`, по умолчанию 5). Актуальные цифры смотри в `results.csv` после прогона; на полном гриде Random Forest обычно ~0.844–0.845.

## Структура

```
titanic/
├── main.py              # точка входа
├── config.py            # merge YAML + DL-хелперы (set_seed, FeatureMode, …)
├── config.yaml          # пути, random_state, cv
├── paths.py
├── bootstrap.py         # sys.path для запуска из корня
├── data/                # train.csv, test.csv
├── notebooks/eda.ipynb
├── ml/
│   ├── main.py          # CV, tune, ensembles, orchestration
│   ├── tune.py
│   ├── create_submission.py
│   ├── feature_engineering.py
│   └── config.yaml
└── dl/
    ├── main.py
    ├── tune.py
    ├── train.py
    ├── create_submission.py
    ├── feature_engineering.py  # матрицы для MLP; FE — из ml/
    └── config.yaml
```

Данные только в `data/`. Папки `ml/data/` и `dl/data/` не используются.

## Конфигурация

Три YAML мержатся в `config.load_config()`:

| YAML | Содержимое |
|------|------------|
| `config.yaml` | `paths`, `random_state`, `cv` |
| `ml/config.yaml` | модели, validation, tune, ensemble, submission |
| `dl/config.yaml` | `train.*`, `feature_mode`, `tune`, `val_size` |

Сабмит по умолчанию — `ensemble_diverse_voting` (`ml/config.yaml` → `submission.model`).

## Данные

- `data/train.csv` — 891 строк, таргет `Survived`
- `data/test.csv` — 418 строк, без таргета
