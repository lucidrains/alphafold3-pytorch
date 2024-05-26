cp .env.example .env
pip install -e '.[test]'
pytest tests/
