
from corpus import get_corpus
import re
import requests
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from bs4 import BeautifulSoup
from storage import get_crawled_dataset, save_crawled_dataset
from urllib.parse import urlparse
import uuid

def crawl_dataset(dataset_name: str):
    corpus = get_corpus(dataset_name)
    
    new_docs = []
    for doc in corpus.values():
        new_docs.extend(_expand_document_with_crawled_data(doc))

    for doc in new_docs:
        corpus[str(uuid.uuid4())] = doc

    save_crawled_dataset(corpus, dataset_name)


def __extractURLs(content):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
    return urls


def _expand_document_with_crawled_data(doc_content: str) -> list:
    document_included_urls = __extractURLs(doc_content)
    new_docs = []
    if len(document_included_urls) > 0:
        for url in document_included_urls:
            crawled_text = __crawl(url)
            new_docs += crawled_text
    return new_docs


def __is_text_url(url):
    # send a HEAD request to the URL to retrieve the headers
    response = requests.head(url)

    # check the Content-Type and Content-Disposition headers
    content_type = response.headers['Content-Type']
    content_disposition = response.headers.get('Content-Disposition', '')
    if 'text' in content_type and 'attachment' not in content_disposition:
        # check the file extension of the URL
        resource = urlparse(url).path
        file_extension = resource.split('.')[-1]
        # array of common text files extensions
        text_file_extensions = ['txt', 'html', 'htm', 'xml', 'csv', 'json', 'md', 'rst', 'php', 'asp', 'aspx', 'css',
                                'js', 'py', 'rb', 'java', 'c', 'cpp', 'h', 'sh', 'bat', 'log', 'ini', 'conf', 'yml',
                                'yaml']
        if file_extension in text_file_extensions:
            return True

        # download a small portion of the response and check its contents
        response = requests.get(url, stream=True)
        content = response.raw.read(1024)
        if all(32 <= c < 127 or c in (9, 10, 13) for c in content):
            return True

    return False


def __crawl(url):
    try:
        if __is_text_url(url):
            print(f"crawling {url}")
            html = urlopen(url).read()
            soup = BeautifulSoup(html, features="html.parser")

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()

            # ###### some text processing #######
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        else:
            return ''
    except (HTTPError, URLError, RemoteDisconnected) as e:
        return ''
    except Exception as e:
        return ''


crawl_dataset("lifestyle")

corpus = get_corpus("lifestyle")
print(len(corpus))

crawled_corpus = get_crawled_dataset("lifestyle")
print(len(crawled_corpus))


# crawl_dataset("antique")

# corpus = get_corpus("antique")
# print(len(corpus))

# crawled_corpus = get_crawled_dataset("antique")
# print(len(crawled_corpus))