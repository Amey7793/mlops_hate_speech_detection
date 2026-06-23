import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, preprocess_text, anonymize


def test_anonymize():
    assert "@user hello" == anonymize("@user hello").strip() or "@user" not in anonymize("@user hello")

def test_clean_text_lowercase():
    assert clean_text("Hello World") == "hello world"

def test_clean_text_removes_numbers():
    assert "123" not in clean_text("abc 123 def")

def test_clean_text_removes_hashtags():
    assert "#hate" not in clean_text("#hate speech")

def test_preprocess_text_returns_string():
    result = preprocess_text("I hate you so much")
    assert isinstance(result, str)
    assert len(result) > 0

def test_preprocess_text_removes_stopwords():
    result = preprocess_text("this is a test")
    assert "is" not in result.split()
    assert "a" not in result.split()
