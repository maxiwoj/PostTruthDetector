from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, \
    Embedding, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l2

from post_truth_detector.additional import clickbait_words_path, \
    clickbait_phrases_path


class Vocabulary:
    def __init__(self):
        with open(clickbait_words_path) as file:
            self.words = file.readlines()[0].split()

        with open(clickbait_phrases_path) as file:
            phrases = file.readlines()
        for i, phrase in enumerate(phrases):
            phrases[i] = phrase.rstrip()
        self.phrases = list(filter(lambda x: len(x) > 0, phrases))

        self.inverse_vocabulary = dict((word, i) for i, word in enumerate(
            self.words))

    def update_vocabulary(self, words):
        with open(clickbait_words_path, "a+") as file:
            last_elem_in_vocab = max(self.inverse_vocabulary.values()) + 1
            for word in words:
                if self.inverse_vocabulary.get(word, -1) == -1:
                    self.inverse_vocabulary[word] = last_elem_in_vocab
                    file.write(" " + word.lower().encode('ascii', 'ignore')
                               .decode("ascii"))
                    last_elem_in_vocab += 1
        self.words += words


vocabulary = Vocabulary()


def convolutional_net(vocabulary_size, input_length):
    embedding_dimension = 30

    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_dimension,
                        input_length=input_length, trainable=False))

    model.add(Convolution1D(32, 2, kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())

    model.add(Convolution1D(32, 2, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())

    model.add(Convolution1D(32, 2, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(17))
    model.add(Flatten())

    model.add(Dense(1, use_bias=True, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())

    return model


def map_sentence(inverse_vocabulary, sentence):
    voc_len = len(inverse_vocabulary)
    phrases_mark = list()
    for i, phrase in enumerate(vocabulary.phrases):
        if phrase in sentence:
            phrases_mark.append(voc_len + i)
    for x in sentence.split():
        if x.isupper():
            phrases_mark.append(voc_len + len(vocabulary.phrases) + 1)
            break
    return phrases_mark + [
        inverse_vocabulary.get(word, voc_len + len(vocabulary.phrases) + 2)
        for word in
        sentence.lower().split()]
