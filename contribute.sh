cp .env.sample .env
pip install -e '.[test]'
pytest tests/
