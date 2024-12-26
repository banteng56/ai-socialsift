
from datasets import load_dataset
from collections import defaultdict, Counter
import numpy as np
import math

dataset=load_dataset("grit-id/id_nergrit_corpus","statement",trust_remote_code=True)
label=dataset['train'].features['ner_tags'].feature.names
print('label=',label)
# Kelas HMM
class HMM2:
    def __init__(self,label):
        self.transition_probs = defaultdict(Counter)
        self.emission_probs=defaultdict(dict)
        self.start_probs=Counter()
        self.tags = label
        self.word_document_counts=Counter()
        self.smoothing_value= 1e-6
    
    def train(self, sentences, tags):
        tag_word_counts = defaultdict(Counter)
        tag_total_words = Counter()
        total_documents = len(sentences)

        for sentence, tag_seq in zip(sentences, tags):
            prev_tag = None
            sentence_words = set(sentence) 

            for word, tag_id in zip(sentence, tag_seq):
                tag = self.tags[tag_id]

                if prev_tag is None:
                    self.start_probs[tag] += 1
                else:
                    self.transition_probs[prev_tag][tag] += 1

                tag_word_counts[tag][word] += 1
                tag_total_words[tag] += 1
                prev_tag = tag

            for word in sentence_words:
                self.word_document_counts[word] += 1

        total_start = sum(self.start_probs.values())
        for tag in self.start_probs:
            self.start_probs[tag] /= total_start

        for tag in self.transition_probs:
            total_trans = sum(self.transition_probs[tag].values()) + self.smoothing_value * len(self.tags)
            for next_tag in self.tags:
                self.transition_probs[tag][next_tag] = (self.transition_probs[tag][next_tag] + self.smoothing_value) / total_trans

        for tag in tag_word_counts:
            for word, count in tag_word_counts[tag].items():
                tf = count / tag_total_words[tag]

                df = self.word_document_counts[word]
                idf = math.log((total_documents + 1) / (df + 1))  

                tf_idf_score = tf * idf
                self.emission_probs[tag][word] = tf_idf_score

            total_emission = sum(self.emission_probs[tag].values()) + self.smoothing_value * len(self.emission_probs[tag])
            for word in self.emission_probs[tag]:
                self.emission_probs[tag][word] = (self.emission_probs[tag][word] + self.smoothing_value) / total_emission

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        for tag in self.tags:
            V[0][tag] = self.start_probs[tag] * self.emission_probs[tag].get(sentence[0], self.smoothing_value)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for tag in self.tags:
                (prob, best_prev_tag) = max(
                    (V[t-1][prev_tag] * self.transition_probs[prev_tag].get(tag, self.smoothing_value) *
                     self.emission_probs[tag].get(sentence[t], self.smoothing_value), prev_tag)
                    for prev_tag in self.tags
                )
                V[t][tag] = prob
                new_path[tag] = path[best_prev_tag] + [tag]

            path = new_path

        n = len(sentence) - 1
        (prob, best_tag) = max((V[n][tag], tag) for tag in self.tags)
        return path[best_tag]
        
    
train_sentences = [example["tokens"] for example in dataset["train"]]
train_tags = [example["ner_tags"] for example in dataset["train"]]

hmm_ner_tfidf = HMM2(label)
hmm_ner_tfidf.train(train_sentences, train_tags)
