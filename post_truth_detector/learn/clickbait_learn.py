import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
from textblob import TextBlob

from post_truth_detector.additional import clickbait_model_weights_path, \
    clickbait_titles_path, genuine_titles_path
from post_truth_detector.learn.clickbaitness_net import map_sentence, \
    convolutional_net, vocabulary


def clickbaitness_learn(sentences=None, sequence_length=20, epochs=4):
    with open(clickbait_titles_path) as clickbait_file:
        clickbait = clickbait_file.readlines()
    for i, title in enumerate(clickbait):
        clickbait[i] = title.rstrip()

    with open(genuine_titles_path) as genuine_file:
        genuine = genuine_file.readlines()
    for i, title in enumerate(genuine):
        genuine[i] = title.rstrip()

    if sentences is not None:
        vocabulary.update_vocabulary([word for sentence in sentences for word
                                      in TextBlob(sentence).words])

    clickbait = np.array(clickbait)
    genuine = np.array(genuine)
    x = np.concatenate([clickbait, genuine], axis=0)
    y = np.array([[1] * clickbait.shape[0] + [0] * genuine.shape[0]],
                 dtype=np.int32).T
    p = np.random.permutation(y.shape[0])
    x = x[p]
    y = y[p]

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

    x_train = sequence.pad_sequences(
        [map_sentence(vocabulary.inverse_vocabulary, sentence) for
         sentence in x_train], maxlen=sequence_length)
    x_test = sequence.pad_sequences(
        [map_sentence(vocabulary.inverse_vocabulary, sentence) for sentence
         in x_test], maxlen=sequence_length)

    model = convolutional_net(vocabulary_size=len(vocabulary.words +
                                                  vocabulary.phrases) + 3,
                              input_length=sequence_length)

    model.compile(loss="logcosh", optimizer="adam", metrics=["acc"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32,
              epochs=epochs, shuffle=True,
              callbacks=[EarlyStopping(monitor="val_loss", patience=2)])
    model.save_weights(clickbait_model_weights_path)
