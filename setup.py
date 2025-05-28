# setup.py
from setuptools import setup
from pathlib import Path

# читаем список зависимостей из requirements.txt
here = Path(__file__).parent
install_requires = (here / "requirements.txt").read_text().splitlines()

setup(
    name="hardocr",
    version="0.1.3",
    description="None",
    install_requires=install_requires,
)
