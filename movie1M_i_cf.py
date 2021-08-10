# coding=UTF-8
"""
The version of package.
Python: 3.6.9
Keras: 2.0.8
Tensorflow-base:1.10.0
"""
import gc
import time
import keras
from time import time
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, multiply, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input, merge, Conv2D, Layer, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from code_keras.idea3.LoadMovieData import load_itemGenres_as_matrix
from code_keras.idea3.LoadMovieData import load_negative_file
from code_keras.idea3.LoadMovieData import load_rating_file_as_list
from code_keras.idea3.LoadMovieData import load_rating_train_as_matrix
from code_keras.idea3.LoadMovieData import load_user_attributes
from code_keras.idea3.evaluateml import evaluate_model
import code_keras.idea3.ml1m_pre as uicf


class Self_Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Self_Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
def get_train_instances(user_gender_mat, user_age_mat, user_oc_mat, ratings, items_genres_mat):
    user_gender_input, user_age_input, user_oc_input, item_attr_input, user_id_input, item_id_input, labels = [], [], [], [], [], [], []
    num_users, num_items = ratings.shape
    num_negatives = 10

    for (u, i) in ratings.keys():
        # positive instance
        user_gender_input.append(user_gender_mat[u])
        user_age_input.append(user_age_mat[u])
        user_oc_input.append(user_oc_mat[u])
        user_id_input.append([u])
        item_id_input.append([i])
        item_attr_input.append(items_genres_mat[i])
        labels.append([1])

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)
            user_gender_input.append(user_gender_mat[u])
            user_age_input.append(user_age_mat[u])
            user_oc_input.append(user_oc_mat[u])
            user_id_input.append([u])
            item_id_input.append([j])
            item_attr_input.append(items_genres_mat[j])
            labels.append([0])


    array_user_gender_input = np.array(user_gender_input)
    array_user_age_input = np.array(user_age_input)
    array_user_oc_input = np.array(user_oc_input)
    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_item_attr_input = np.array(item_attr_input)
    array_labels = np.array(labels)

    del user_gender_input, user_age_input, user_oc_input, user_id_input, item_id_input, item_attr_input, labels
    gc.collect()

    return array_user_gender_input, array_user_age_input, array_user_oc_input, array_user_id_input, array_item_attr_input, array_item_id_input, array_labels



def get_lCoupledCF_model(num_users, num_items):

    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # Input
    user_gender_input = Input(shape=(2,), dtype='float32', name='user_gender_input')  # 用户属性信息
    user_age_input = Input(shape=(7,), dtype='float32', name='user_age_input')
    user_oc_input = Input(shape=(21,), dtype='float32', name='user_oc_input')

    item_attr_input = Input(shape=(18,), dtype='float32', name='item_attr_input')  # 项目属性信息

    user_gender_embedding = Dense(6, activation="relu", name="user_gender_embedding_ui")(user_gender_input)


    user_age_embedding = Dense(6, activation="relu", name="user_age_embedding2_ui")(user_age_input)

    user_oc_embedding = Dense(6, activation="relu", name="user_oc_embedding_ui")(user_oc_input)

    user_gender_embedding = Reshape((1, 6))(user_gender_embedding)
    user_age_embedding = Reshape((1, 6))(user_age_embedding)
    user_oc_embedding = Reshape((1, 6))(user_oc_embedding)

    item_attr_embedding = Dense(6, activation="relu",name="item_att_embedding_ui")(item_attr_input)  # 1st hidden layer
    #item_attr_embedding = Dense(4, activation="relu", name="item_att_embedding1_ui")(item_attr_embedding)
    item_attr_embedding = Reshape((1, 6))(item_attr_embedding)





    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding_ui',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_latent_vector = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding_ui',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_latent_vector = Flatten()(item_id_Embedding(item_id_input))


    #print(u_i_id.shape)
    u_i_gender = multiply([user_gender_embedding, item_attr_embedding])
    #u_i_gender = Position_Embedding()(u_i_gender)
    u_i_gender = Self_Attention(3,2)([u_i_gender, u_i_gender, u_i_gender])
    u_i_gender = GlobalAveragePooling1D()(u_i_gender)
    u_i_age = multiply([user_age_embedding, item_attr_embedding])
    #u_i_age = Position_Embedding()(u_i_age)
    u_i_age = Self_Attention(3,2)([u_i_age, u_i_age, u_i_age])
    u_i_age = GlobalAveragePooling1D()(u_i_age)
    u_i_oc = multiply([user_oc_embedding, item_attr_embedding])
    #u_i_oc = Position_Embedding()(u_i_oc)
    u_i_oc = Self_Attention(3,2)([u_i_oc, u_i_oc, u_i_oc])
    u_i_oc = GlobalAveragePooling1D()(u_i_oc)
    u_i_attr = merge([u_i_gender, u_i_age, u_i_oc], mode="concat")

    u_i_id = multiply([user_latent_vector, item_latent_vector])
    predict_vector = merge([u_i_id, u_i_attr], mode="concat")


    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(predict_vector)

    # Final prediction layer
    model = Model(input=[user_gender_input,user_age_input,user_oc_input, item_attr_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def load_pretrain_model(model, gmf_model):
    # MF embeddings
    gmf_user_gender_embeddings = gmf_model.get_layer('user_gender_embedding').get_weights()
    gmf_user_age_embeddings_2 = gmf_model.get_layer('user_age_embedding2').get_weights()
    gmf_user_oc_embedding = gmf_model.get_layer('user_oc_embedding').get_weights()
    gmf_item_attr_embedding = gmf_model.get_layer('item_att_embedding').get_weights()

    model.get_layer('user_gender_embedding_ui').set_weights(gmf_user_gender_embeddings)
    model.get_layer('user_age_embedding2_ui').set_weights(gmf_user_age_embeddings_2)
    model.get_layer('user_oc_embedding_ui').set_weights(gmf_user_oc_embedding)
    model.get_layer('item_att_embedding_ui').set_weights(gmf_item_attr_embedding)

    gmf_user_embeddings = gmf_model.get_layer('user_id_Embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_id_Embedding').get_weights()

    model.get_layer('user_id_Embedding_ui').set_weights(gmf_user_embeddings)
    model.get_layer('item_id_Embedding_ui').set_weights(gmf_item_embeddings)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('topLayer').get_weights()

    model.get_layer('topLayer').set_weights(gmf_prediction)
    return model


def main():
    learning_rate = 0.001
    num_epochs = 100
    verbose = 1
    topK = 10
    out = 1
    dataset = "ml_m"
    num_factor = 32
    mf_pretrain = ""

    #mf_pretrain = "Pretrain/ml_1M_uicf_32_1604554464.h5"

    evaluation_threads = 1
    startTime = time()
    model_out_file = 'Pretrain/%s_movie1mUICF_%d_%s_%d.h5' % (dataset, num_factor, 5, time())
    # load data
    num_users, user_gender_mat, user_age_mat, user_oc_mat = load_user_attributes()  # 用户，用户属性
    num_items, items_genres_mat = load_itemGenres_as_matrix()  # 项目，项目属性
    ratings = load_rating_train_as_matrix()  # 评分矩阵

    # load model
    # change the value of 'theModel' with the key in 'model_dict'
    # to load different models

    theModel = "movie1m_uicf"
    model = get_lCoupledCF_model(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    to_file = 'Model_' + theModel + '.png'
    plot_model(model, show_shapes=True, to_file=to_file)
    model.summary()

    # Load pretrain model
    if mf_pretrain != '':
        gmf_model = uicf.get_lCoupledCF_model(num_users, num_items)
        gmf_model.load_weights(mf_pretrain)

        model = load_pretrain_model(model, gmf_model)
        print("Load pretrained movieMF (%s) done. " % (mf_pretrain))

    # Init performance
    testRatings = load_rating_file_as_list()
    testNegatives = load_negative_file()
    (hits, ndcgs,recalls) = evaluate_model(model, testRatings, testNegatives,
                                   user_gender_mat, user_age_mat, user_oc_mat, items_genres_mat, topK,
                                   evaluation_threads)
    hr, ndcg,recall = np.array(hits).mean(), np.array(ndcgs).mean(),np.array(recalls).mean()
    print('Init: HR = %.4f, NDCG = %.4f, recall = %.4f' % (hr, ndcg,recall))
    best_hr, best_ndcg,best_recall, best_iter = hr, ndcg,recall, -1
    if out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(num_epochs):
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_gender_input, user_age_input, user_oc_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(
            user_gender_mat, user_age_mat, user_oc_mat,
            ratings,
            items_genres_mat)
        hist5 = model.fit(
            [user_gender_input, user_age_input, user_oc_input, item_attr_input, user_id_input, item_id_input], labels,
            epochs=1,
            batch_size=256, verbose=2, shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs,recalls) = evaluate_model(model, testRatings, testNegatives,
                                           user_gender_mat, user_age_mat, user_oc_mat, items_genres_mat, topK,
                                           evaluation_threads)
            hr, ndcg,recall, loss5 = np.array(hits).mean(), np.array(ndcgs).mean(),np.array(recalls).mean(), hist5.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f,recall = %.4f, loss5 = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg,recall, loss5, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            if recall > best_recall:
                best_recall = recall
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f,best recall = %.4f,time = %.1f s" %
          (best_hr, best_ndcg,best_recall, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f,recall = %.4f' % (hr, ndcg,recall))
    if out > 0:
        print("The best movie1m_uicf model is saved to %s" % (model_out_file))


if __name__ == '__main__':
    main()
