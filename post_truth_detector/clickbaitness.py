from keras.preprocessing import sequence

from post_truth_detector.additional import clickbait_model_weights_path
from post_truth_detector.learn.clickbaitness_net import convolutional_net, \
    map_sentence, vocabulary


class Predictor:
    def __init__(self):
        self.sequence_length = 20
        model = convolutional_net(vocabulary_size=len(vocabulary.words +
                                                      vocabulary.phrases) + 3,
                                  input_length=self.sequence_length)
        model.load_weights(clickbait_model_weights_path)
        self.model = model

    def predict(self, heading):
        inputs = sequence.pad_sequences([map_sentence(
            vocabulary.inverse_vocabulary, heading.lower())],
            maxlen=self.sequence_length)
        return self.model.predict(inputs)[0, 0]
