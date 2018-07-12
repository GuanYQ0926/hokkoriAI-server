from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import tsv
import json


def simple_get(url):
    def is_good_response(resp):
        content_type = resp.headers['Content-Type'].lower()
        return (resp.status_code == 200
                and content_type is not None
                and content_type.find('html') > -1)

    def log_error(e):
        print(e)

    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        log_error('Erro during requests to {0} : {1}'.format(url, str(e)))


def get_information(url):
    response = simple_get(url)
    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        dls = html.select('dl')
        q_a_list = []
        # iterate all quesiton and answer pairs
        for dl in dls:
            # get question
            questions = dl.find('dt').find_all('p')
            if not questions:
                continue
            res_question = ''
            for question in questions:
                temp = question.text
                temp = ''.join(temp.split('\r\n'))
                temp = ''.join(temp.split('\n'))
                res_question += temp
            # get answer
            answers = dl.find('dd').find_all('p')
            if not answers:
                continue
            res_answer = ''
            for answer in answers:
                temp = answer.text
                temp = ''.join(temp.split('\r\n'))
                temp = ''.join(temp.split('\n'))
                res_answer += temp
            q_a_list.append(dict(q=res_question, a=res_answer,
                                 s='Editorial', m=''))
        return q_a_list
    else:
        raise Exception('Error retrieving contents at {}'.format(url))


def generate_qa_sheets(file_path):
    writer = tsv.TsvWriter(open(file_path, 'w'))
    # title
    writer.line('Question', 'Answer', 'Source', 'Metadata')
    with open('../../data/environment.json', 'r') as f:
        environ = json.load(f)
    url_list = environ['url_list']
    for url in url_list:
        results = get_information(url)
        for res in results:
            # if not res['q'] or not res['a'] or not res['s'] or not res['m']:
            #     continue
            writer.line(res['q'], res['a'], res['s'], res['m'])
    writer.close()


if __name__ == '__main__':
    file_path = '../../data/text/Q&A_sheet.tsv'
    generate_qa_sheets(file_path)
