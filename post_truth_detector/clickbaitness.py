from keras.preprocessing import sequence

from post_truth_detector.learn.convolutional_net import convolutional_net, \
    map_sentence

# provided model has 92 percent accuracy
clickbait_model_weights_path = "../models/clickbait_model_weights"

with open("phrases.txt") as file:
    phrases = file.readlines()
for i, phrase in enumerate(phrases):
    phrases[i] = phrase.rstrip()
phrases = list(filter(lambda x: len(x) > 0, phrases))

with open('words.txt') as file:
    words = file.readlines()[0].split()

inverse_vocabulary = dict((word, i) for i, word in enumerate(words))


class Predictor:

    def __init__(self):
        self.sequence_length = 20
        model = convolutional_net(vocabulary_size=len(words),
                                  input_length=self.sequence_length)
        model.load_weights(clickbait_model_weights_path)
        self.model = model

    def predict(self, heading):
        inputs = sequence.pad_sequences([map_sentence(inverse_vocabulary,
                                                      heading.lower())],
                                        maxlen=self.sequence_length)
        return self.model.predict(inputs)[0, 0]
