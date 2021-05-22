import re

from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    replaced_text = '\n'.join(text.splitlines())  # skip header by [2:]
    replaced_text = clean_url(replaced_text)
    replaced_text = clean_html_tags(replaced_text)
    replaced_text = clean_html_and_js_tags(replaced_text)
    replaced_text = replaced_text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub(r'https?:\/\/.*.?[\r\n ]?', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    return replaced_text


def clean_html_tags(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text


def clean_html_and_js_tags(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text


def clean_url(html_text: str) -> str:
    """
    \S+ matches all non-whitespace characters (the end of the url)
    :param html_text:
    :return:
    """
    clean_text = re.sub(r'http\S+', '', html_text)
    return clean_text


if __name__ == "__main__":
    example_txts = [
        "@picachu 草でチュウ",
        "This is url https://hoge",
        "year <span color=\"red\"> wvec </span>",
        "normal",
        "【草】草草草の草【朗報】"
    ]

    cleaned_txts = [clean_text(txt) for txt in example_txts]

    for txt, cleaned in zip(example_txts, cleaned_txts):
        print("Given:   " + txt)
        print("Cleaned: " + cleaned)
        print()