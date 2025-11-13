# Lab0 - Fundamentals of Continuous Integration

#### Author: Unai Lalana

## Overview
This project implements a **data preprocessing toolkit** with a CLI using Python. It demonstrates basic **Continuous Integration (CI)** practices: linting, formatting, unit and integration testing, and code coverage.

---

## Project Structure
```
Lab0/
├── src/
│ ├── preprocessing.py # Core logic
│ └── cli.py # CLI using Click
├── tests/
│ ├── test_preprocessing.py # Unit tests
│ └── test_cli.py # CLI integration tests
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Functionality

- **Clean**: remove/fill missing values  
- **Numeric**: normalize, standardize, clip, convert to int, log transform  
- **Text**: tokenize, remove punctuation, remove stopwords  
- **Struct**: shuffle, flatten, get unique values  

---

## Installation

```bash
git clone https://github.com/UnaiLalana/Lab0
cd Lab0
uv init
uv sync
uv add click black pylint pytest pytest-cov
```
---

## Usage Examples
```bash
uv run python -m src.cli clean remove-missing --values "[1, None, 2, '', 3]"
uv run python -m src.cli numeric normalize --values "[10, 20, 30]"
uv run python -m src.cli struct shuffle --values "[1,2,3,4]"
uv run python -m src.cli text tokenize --texts "Hello world!"
```

---

## Testing

```bash
uv run python -m pytest -v
uv run python -m pytest -v --cov=src
```

---

## Code Quality

```bash
uv run python -m pylint src/*.py
uv run black src/*.py
```