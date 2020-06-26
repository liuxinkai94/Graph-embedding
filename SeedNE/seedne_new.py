import tensorflow as tf
import numpy as np

from model_new import SeedNEModel
from utils_new import DBLPDataLoader
import pickle
import time
from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE
from tensorflow.python.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import os

# Multiple classification test
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list =[len(l) for l in Y]
        self.predict(X, top_k_list)
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        return results


    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    if skip_head:
        fin.readline()
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('./data/cora/cora_labels.txt')
    tr_frac = 0.8
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    return clf.split_train_evaluate(X, Y, tr_frac)


#training
def train(inilearning_rate=0.025, num_batches=2600, batch_size=8, K=3):
    graph_file = './data/cora/cora_edgelist.txt'
    data_loader = DBLPDataLoader(graph_file=graph_file)
    num_of_nodes = data_loader.num_of_nodes
    model = SeedNEModel(num_of_nodes, 100, K=K)
    tt = time.ctime().replace(' ', '-')
    tt = tt.replace(':', '-')
    path = 'SeedNE_new' + '-' + tt
    fout = open(path + "-log.txt", "w")

    with tf.Session() as sess:
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        a = 0.00001
        c = 0.00005
        sampling_time, training_time = 0, 0
        learning_rate = inilearning_rate
        for b in range(num_batches):
            t1 = time.time()
            cur_embedding = sess.run(model.embedding)
            lu = sess.run(model.lu,feed_dict={model.a: a, model.b: c})
            u_i, u_j, label = data_loader.fetch_batch(batch_size=batch_size, K=K, embedding=cur_embedding, lu=lu)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label,
                         model.learning_rate: learning_rate, model.a: a, model.b: c}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 50 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2

            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                learning_rate = max(inilearning_rate * 0.0001, learning_rate*0.99)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 500 ==0:
                a = a * 2
            if b % 50 == 0 or b == (num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

                fout.write("epochs:%d classification: lu=%f\n" % (b, lu))
                fout.write(str(evaluate_embeddings(data_loader.embedding_mapping(normalized_embedding))))
                fout.write('\n')
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s%s.pkl' % (path, str(b)), 'wb'))
            fout.flush()
    fout.close()


if __name__ == '__main__':
    train()