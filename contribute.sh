cp .env.sample .env
pip install uv
uv pip install -e '.[test]'
pytest tests/
