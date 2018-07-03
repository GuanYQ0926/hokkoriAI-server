import os
import logging
import numpy as np
import MeCab
from gensim.models import word2vec


class Nlp():
    def __init__(self):
        if not os.path.exists('../models/image/wiki.model'):
            self.mode_training()
        else:
            print('load model')
            self.model = word2vec.Word2Vec.load('../models/image/wiki.model')

    def mode_training(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        sentences = word2vec.Text8Corpus('../models/wiki_wakati.txt')
        model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
        model.save('../models/image/wiki.model')

    def process_question(self, question, threshold=0.8):
        # input existed question vector & answer list
        mt = MeCab.Tagger('')
        mt.parse('')

        def get_vector_words(words):
            sum_vec = np.zeros(200)
            word_count = 0
            for word in words:
                try:
                    sum_vec += self.model.wv[word]
                    word_count += 1
                except:
                    pass
            return sum_vec / word_count

        def cos_sim(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) *
                                         np.linalg.norm(vec2))
        q_vec = get_vector_words(question)
        result = ''
        max_value = threshold
        # suppose self.question = [{question:, answer, vec}, {}]
        for info in self.question_answer:
            vec = info['vec']
            temp = cos_sim(q_vec, vec)
            if temp > max_value:
                result = info['answer']
                max_value = temp
        return result
