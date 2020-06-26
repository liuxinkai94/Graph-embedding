import tensorflow as tf


class SeedNEModel:
    def __init__(self, num_of_nodes, embedding_dim, batch_size=8, K=3, window_size=2, walk_length=8):

        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size*(walk_length*K
                                                                     + walk_length*window_size*2
                                                                     - window_size*window_size-window_size)])

        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size*(walk_length*K
                                                                     + walk_length*window_size*2
                                                                     - window_size*window_size-window_size)])

        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[batch_size*(walk_length*K
                                                                           + walk_length*window_size*2
                                                                           - window_size*window_size-window_size)])
        self.embedding = tf.get_variable('target_embedding', [num_of_nodes, embedding_dim],
                                         initializer=tf.random_uniform_initializer(minval=0., maxval=1.))

        self.a = tf.placeholder(name='a', dtype=tf.float32)
        self.b = tf.placeholder(name='b', dtype=tf.float32)
        self.u = tf.Variable(tf.constant(1.), name='u')
        self.one = tf.constant(1.)
        self.lu_ = self.a * self.u * self.u + self.b

        def f1():
            return self.lu_

        def f2():
            return self.one

        self.lu = tf.cond(tf.less(self.lu_, self.one), f1, f2)

        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=num_of_nodes), self.embedding)

        self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=num_of_nodes), self.embedding)

        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)/embedding_dim

        self.loss_j = tf.reduce_mean(tf.log_sigmoid(self.inner_product * self.label))
        ##loss=log_sigmoid(u_i*u_p)+log_sigmoid(-u_i*u_j)

        self.loss = -(self.loss_j + self.lu)

        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)


