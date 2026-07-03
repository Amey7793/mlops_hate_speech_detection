import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, preprocess_text, anonymize


# ---- anonymize tests ----

def test_anonymize_removes_username():
    result = anonymize("@user hello world")
    assert "@user" not in result

def test_anonymize_keeps_rest_of_text():
    result = anonymize("@user hello world")
    assert "hello" in result

def test_anonymize_no_username():
    result = anonymize("hello world")
    assert result == "hello world"

def test_anonymize_multiple_usernames():
    result = anonymize("@user1 and @user2 are here")
    assert "@user1" not in result
    assert "@user2" not in result


# ---- clean_text tests ----

def test_clean_text_lowercase():
    assert clean_text("Hello World") == "hello world"

def test_clean_text_removes_numbers():
    assert "123" not in clean_text("abc 123 def")

def test_clean_text_removes_hashtags():
    assert "#hate" not in clean_text("#hate speech")

def test_clean_text_removes_punctuation():
    result = clean_text("hello, world!")
    assert "," not in result
    assert "!" not in result

def test_clean_text_removes_newlines():
    result = clean_text("hello\nworld")
    assert "\n" not in result

def test_clean_text_returns_string():
    assert isinstance(clean_text("Hello World"), str)

def test_clean_text_empty_string():
    assert isinstance(clean_text(""), str)


# ---- preprocess_text tests ----

def test_preprocess_text_returns_string():
    result = preprocess_text("I hate you so much")
    assert isinstance(result, str)

def test_preprocess_text_removes_stopwords():
    result = preprocess_text("this is a test")
    assert "is" not in result.split()
    assert "a" not in result.split()

def test_preprocess_text_not_empty_for_valid_input():
    result = preprocess_text("I hate you")
    assert len(result) > 0

def test_preprocess_text_removes_username():
    result = preprocess_text("@user I hate you")
    assert "@user" not in result

def test_preprocess_text_removes_hashtag():
    result = preprocess_text("#hate speech is bad")
    assert "#hate" not in result
