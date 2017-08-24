import tensorflow as tf
import numpy as np
import string
import settings as s
import warnings
import sys
import os

MAXCHUNK = 10000


def weight_variable(shape, var='scaled', name='weights'):
    """
    create weight variable with random initialization
    either fixed variance of 0.1 or scaled to sqrt(2/n) where n is number of
        inputs

    Args:
        shape (array): dimensions of weight tensor
        var (float): fixed or scaled variance
        name (str): name of variable

    Returns:
        tf.Variable: weight variable

    Raises:
        ValueError: if invalid value is specified for "var"
    """
    with tf.device('/cpu:0'):
        if var == 'fixed':
            v = tf.get_variable(
                name, shape,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        elif var == 'scaled':
            v = tf.get_variable(
                name, shape,
                initializer=tf.truncated_normal_initializer(
                    mean=0, stddev=np.sqrt(2./np.prod(shape[0:3]))))
        else:
            raise ValueError('var must be either \'fixed\' or \'scaled\'')
    return v


def bias_variable(shape, name='biases'):
    """
    create bias variable with constant initialization

    Args:
        shape (array): dimensions of bias tensor
        name (str): name of variable

    Returns:
        tf.Variable: bias variable
    """
    with tf.device('/cpu:0'):
        v = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
    return v


def conv2d(x, W, name='conv2d'):
    """
    create 2d convolution

    Args:
        x (tf.Tensor): input to convolutions
        W (tf.Tensor): weights for convolutions
        name (str): name of operation

    Returns:
        tf.Tensor: resulting tensor after convolutions are performed
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',
                        name=name)


def batch_norm(x, depth):
    """perform batch normalization
    
    Args:
        x (tf.Tensor): input tensor to normalize
        depth (int): length of dimension to normalize over
    
    Returns:
        tf.Tensor: normalized tensor
    """
    mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
    variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))

    batch_mean, batch_variance = tf.nn.moments(x, [0, 1, 2])
    assign_mean = mean.assign(batch_mean)
    assign_variance = variance.assign(batch_variance)
    with tf.control_dependencies([assign_mean, assign_variance]):
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, beta, gamma, 1e-4, True, name='batch_norm')


def create_conv_layer(input_, n_in, n_out, name='conv', bias=True,
                      norm=False):
    """
    create layer with tensor augmentation, convolution and relu

    Args:
        input_ (tf.Tensor): input to convolutional layer
        n_in (int): number of dimensions in input data
        n_out (int): number of dimensions (i.e. filters) in output
        name (str): name of layer
        bias (bool): whether or not to include bias term
        norm (bool): whether or not to apply batch normalization

    Returns:
        tf.Tensor, tf.Tensor: weights, outputs
    """
    with tf.variable_scope(name):
        W = weight_variable([3, 1, n_in, n_out])
        if bias:
            b = bias_variable([n_out])

        conv = conv2d(input_, W)
        if bias:
            conv = conv + b
        if norm:
            relu = tf.nn.relu(batch_norm(conv, n_out), name='relu')
        else:
            relu = tf.nn.relu(conv, name='relu')
    return W, relu


def create_fully_connected_layer(input_, n_in, n_out=1, relu=False,
                                 name='fully_connected'):
    """create fully connected layer

    Args:
        input_ (tf.Tensor): input to convolutional layer
        n_in (int): number of dimensions in input data
        n_out (int): number of dimensions in output
        relu (bool): whether or not to include relu
        name (str): name of layer

    Returns:
        tf.Tensor, tf.Tensor: weights, outputs
    """
    with tf.variable_scope(name):
        flat = tf.reshape(input_, [-1, n_in])
        W = weight_variable([n_in, n_out])
        b = bias_variable([n_out])

        h_fc = tf.matmul(flat, W) + b
        if relu:
            h_fc = tf.nn.relu(h_fc, name='relu')
    return W, h_fc


def seq_lengths(tensor):
    """get length of each zero-padded sequence
    
    Args:
        tensor (tf.Tensor): input tensor
    
    Returns:
        tf.Tensor: lengths of input sequences"""
    used = tf.sign(tf.reduce_max(tf.abs(tensor), reduction_indices=2))
    return tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)


class NN(object):
    """
    base class for neural networks
    """

    def __init__(self, units, write=False, learning_rate=1e-4,
                 optimizer='adam', name='', lowmem=False, num_gpus=1,
                 **kwargs):
        """
        initialize flow graph and start session

        Args:
            units (str): specifies sizes of layers
            write (bool): whether or not to write log
            learning_rate (float): learning rate for training
            optimizer (str): optimizer for training can be "gradient" for
                gradient descent, "adam", "adagrad", or "rmsprop"
            name (str): name of model for saving to file
            lowmem (bool): whether or not to use low memory mode
            num_gpu (int): number of gpus to use
        """
        self.write = write
        self.num_gpus = num_gpus
        self.name = name + '_' + \
            ''.join(np.random.choice(list(string.ascii_lowercase), 6))
        if optimizer not in ['gradient', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'gradient\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        
        # inputs to model
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True))
        self.build(units, learning_rate, optimizer, **kwargs)
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)
        if write:
            self.init_writer()
        self.i = 0
        self.lowmem = lowmem

    def init_writer(self):
        """
        initialize writer objects
        """
        self.write = True
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(s.LOGS_DIR, self.sess.graph)

    def _process_data(self, data):
        """
        processes fetched data, add one matrix for error if necessary

        Args:
            data (tuple): data returned from object

        Returns:
            tuple: three matrices - sequence, dG, dG error
        """
        if len(data) == 2:
            return data[0], data[1], np.ones(data[1].shape)
        elif len(data) == 3:
            return data
        else:
            raise ValueError('data object\'s get_train_batch() and '
                             'get_data() functions must return two or '
                             'three values')

    def average_gradients(self, tower_grads, clip=True):
        """
        compute the average of gradients computed across multiple towers
        
        Args:
            tower_grads (list) : list of lists of (gradient, variable) tuples,
                the outer list is over individual gradients, the inner list is
                over the gradient calculation for each tower
            clip (bool): whether or not to clip gradients
        
        Returns:
            list: of pairs of (gradient, variable) where the gradient has been
                averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # average over the 'tower' dimension.
            grad = tf.stack([g for g, _ in grad_and_vars if g is not None], 0)
            grad = tf.reduce_mean(grad, 0)

            # keep only variable from first tower
            average_grads.append((grad, grad_and_vars[0][1]))
        if clip:
            average_grads = [(tf.clip_by_norm(g, 1), v) for g, v in
                             average_grads]
        return average_grads

    def build(self, units, learning_rate, optimizer='adam', **kwargs):
        """
        build RNN
        
        Args:
            units (str or int): specifies number of units, either as an int, if
                only one layer, or as string 'LxU' where L is the number of
                layers and U is the number of units per layer
            learning_rate (float): specifies learning rate of optimization
            optimizer (str): method for optimization, either "gradient",
                "rmsprop", "adam" or "adagrad"
        """
        # inputs
        self.seq = tf.placeholder(tf.float32, [None, s.MAX_SEQ_LEN,
                                  len(s.bases)], name='seq')
        self.dG = tf.placeholder(tf.float32, [None], name='dG')
        self.dG_sterr = tf.placeholder(tf.float32, shape=[None],
                                       name='dG_sterr')
        self.keep_prob = tf.placeholder(tf.float32)
        
        # split dataset for each gpu
        self.seqs = tf.split(self.seq, self.num_gpus, 0)
        self.dGs = tf.split(self.dG, self.num_gpus, 0)
        self.dG_sterrs = tf.split(self.dG_sterr, self.num_gpus, 0)

        # optimizer
        if optimizer == 'gradient':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError('optimizer must be \'gradient\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        # rnn elements
        self.parse_units(units)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower%d' % i) as scope:
                        loss = self.build_tower(i, **kwargs)
                        tf.get_variable_scope().reuse_variables()
                        grads = self.optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = self.average_gradients(tower_grads)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.dGhat = tf.concat(tf.get_collection('dGhats'), 0, name='all_dGhat')
        self.train_step = self.optimizer.apply_gradients(grads)

        ratios = [tf.reshape(tf.truediv(
            gv[0] * learning_rate, gv[1]), [-1, 1])
            for gv in grads if gv[0] is not None]
        self.update_ratio = tf.concat([r for r in ratios if tf.rank(r) != 0], 0)
        
        if self.write:
            self.train_summ = tf.summary.scalar('training loss', self.loss)
            self.test_summ = tf.summary.scalar('test loss', self.loss)

    def train(self, data, epochs=10, batch_size=128, keepprob=0.5, test=None,
              sterr=False):
        """
        train NN on given data for given number of epochs

        Args:
            data (obj): object that holds data, must have get_train_batch()
                and get_data() functions that returns tuples of data, also
                must have attribute "n" that returns dataset size
            epochs (int): number of epochs to train for
            batch_size (int): number of data points per batch of training
            keepprob (float): proportion of data to keep in dropout layer
            test (obj): object similar to data that holds test data
            sterr (bool): whether or not to use standard errors in loss

        Returns:
            np.ndarray: loss for every 
        """
        epochlen = max(int(float(data.n) / batch_size / self.num_gpus), 1)
        iters = max(int(float(data.n) * epochs / batch_size / self.num_gpus), 1)
        period = epochlen
        # get full train and test datasets if not in low memory mode
        if not self.lowmem:
            allseq, alldG, alldGsterr = self._process_data(
                data.get_data(sterr=sterr))
            n = allseq.shape[0]
        if test is not None:
            if self.lowmem:
                print 'evaluating test loss on a batch due to low memory mode'
            else:
                testseq, testdG, testdGsterr = self._process_data(
                    test.get_data(sterr=sterr))
            loss = np.zeros((2, iters/period + 1))
        else:
            loss = np.zeros((1, iters/period + 1))
        self.save_graph()
        # train for given number of iterations
        for i in range(iters):
            seq, dG, dGsterr = self._process_data(data.get_train_batch(
                batch_size * self.num_gpus, sterr))
            # write losses every period
            if i % period == 0:
                if self.lowmem:
                    loss[0, i/period] = self.get_loss(seq, dG, dGsterr)
                else:
                    loss[0, i/period] = self.get_loss(allseq, alldG, alldGsterr)
                if test:
                    if self.lowmem:
                        testseq, testdG, testdGsterr = self._process_data(
                            test.get_train_batch(batch_size * self.num_gpus,
                                                 sterr))
                    loss[1, i/period] = self.get_loss(
                        testseq, testdG, testdGsterr, test=True)
                    print 'step %d: train rmse %g, test rmse %g' % \
                        (i, loss[0, i/period], loss[1, i/period])
                else:
                    print 'step %d: rmse %g' % (i, loss[0, i/period])
                sys.stdout.flush()
            # save model every 5 periods
            if i != 0 and i % (period * 5) == 0:
                if test:
                    modelsuffix = '_%f' % loss[1, i/period]
                else:
                    modelsuffix = '_train%f' % loss[0, i/period]
                self.save(modelsuffix)
            self.train_batch(seq, dG, dGsterr, keepprob, i % (period * 5) == 0)
            if (i+1) % epochlen == 0:
                print '%d epochs finished' % ((i+1) / epochlen)
                sys.stdout.flush()
        return loss

    def train_batch(self, seq, dG, dGsterr, keepprob=0.5, update_ratio=False):
        """
        train data on single batch of data

        Args:
            seq (np.ndarray): matrix representing sequence
            dG (np.ndarray): matrix of dGs
            dGsterr (np.ndarray): matrix of dG standard errors
            keepprob (float): proportion of data to keep in dropout layer
            update_ratio (bool): whether or not to calculate update ratio
        """
        if update_ratio:
            _, ur = self.sess.run([self.train_step, self.update_ratio],
                                  feed_dict={self.seq: seq, self.dG: dG,
                                             self.dG_sterr: dGsterr,
                                             self.keep_prob: keepprob})
            ur = np.log10(ur)
            ur = ur[np.isfinite(ur)]
            print '\tupdate ratio: mean %f, median %f' % (ur.mean(),
                                                          np.median(ur))
        else:
            self.sess.run([self.train_step],
                          feed_dict={self.seq: seq, self.dG: dG,
                                     self.dG_sterr: dGsterr,
                                     self.keep_prob: keepprob})
        self.i += 1

    def get_loss(self, seq, dG, dGsterr, test=False):
        """
        get value of loss function

        Args:
            seq (np.ndarray): matrix representing sequence
            dG (np.ndarray): matrix of dGs
            dGsterr (np.ndarray): matrix of dG standard errors
            test (bool): whether loss is for test data (for log)

        Returns:
            float: RMSE
        """
        # get variables to fetch
        if self.write:
            if test:
                fetches = [self.dGhat, self.test_summ]
            else:
                fetches = [self.dGhat, self.train_summ]
        else:
            fetches = [self.dGhat]

        # run operations, split into chunks if too large
        chunksize = MAXCHUNK if not self.lowmem else MAXCHUNK/10
        chunksize = chunksize * self.num_gpus
        nseqs = seq.shape[0]
        if nseqs > chunksize or nseqs % self.num_gpus:
            nchunks = int(np.ceil(float(nseqs)/chunksize))
            seq = np.pad(seq, ((0, nchunks * chunksize - nseqs), (0, 0),
                               (0, 0)),
                         'constant')
            dG = np.pad(dG, ((0, nchunks * chunksize - nseqs)), 'constant')
            dGsterr= np.pad(dGsterr, ((0, nchunks * chunksize - nseqs)),
                            'constant')
        if seq.shape[0] < chunksize:
            result = self.sess.run(
                fetches, feed_dict={self.seq: seq, self.dG: dG,
                                    self.dG_sterr: dGsterr,
                                    self.keep_prob: 1})
        else:
            chunk_dGhat = []
            nchunks = int(np.ceil(float(seq.shape[0])/chunksize))
            for i in range(nchunks):
                j, k = i * chunksize, (i+1) * chunksize
                result = self.sess.run(fetches, feed_dict={
                    self.seq: seq[j:k], self.dG: dG[j:k],
                    self.dG_sterr: dGsterr[j:k],
                    self.keep_prob: 1})
                chunk_dGhat.append(result[0])
            result[0] = np.hstack(chunk_dGhat)
        result[0] = result[0][0:nseqs]
        dG = dG[0:nseqs]
        dGsterr = dGsterr[0:nseqs]

        # write results
        if self.write:
            self.writer.add_summary(result[1], self.i)

        if np.all(np.isnan(result[0])):
            raise Exception('numerical overflow on step %d' % self.i)

        return np.sqrt(np.mean(np.square(result[0] - dG)))

    def save_graph(self):
        """
        save meta graph to file
        """
        tf.train.export_meta_graph('%s/%s.meta' % (s.MODELS_DIR, self.name))

    def save(self, suffix=''):
        """
        save variables to file

        Args:
            suffix (str): suffix for save filename
        """
        saver = tf.train.Saver()
        saver.save(self.sess, '%s/%s%s' % (s.MODELS_DIR, self.name, suffix),
                   global_step=self.i, write_meta_graph=False)

    def restore(self, filename):
        """
        restore variables from file

        Args:
            filename (str): filename to read variable values from
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        spl = filename.split('-')
        # if training on the same dataset, keep the name and iteration number
        if spl[0].startswith(self.name.split('_')[0]):
            self.name = os.path.basename(('-'.join(spl[:-1])).rsplit('_', 1)[0])
            self.i = int(spl[-1])

    def test(self, seq):
        """
        test NN on given data

        Args:
            seq (np.ndarray): input sequence matrix
        
        Returns:
            np.ndarray: predicted dGs
        """
        chunksize = MAXCHUNK if not self.lowmem else MAXCHUNK/10
        chunksize = chunksize * self.num_gpus
        nseqs = seq.shape[0]
        if nseqs > chunksize or nseqs % self.num_gpus:
            nchunks = int(np.ceil(float(nseqs)/chunksize))
            seq = np.pad(seq, ((0, nchunks * chunksize - nseqs), (0, 0), (0, 0)),
                         'constant')
        if seq.shape[0] < chunksize:
            return self.dGhat.eval(session=self.sess,
                                   feed_dict={self.seq: seq, self.keep_prob: 1})[0:nseqs]
        else:
            chunk_dGs = np.zeros(seq.shape[0])
            for i in range(nchunks):
                j, k = i * chunksize, (i+1) * chunksize
                chunk_dGs[j:k] = self.dGhat.eval(
                    session=self.sess, feed_dict={self.seq: seq[j:k], self.keep_prob: 1})
            return chunk_dGs[0:nseqs]

    def __del__(self):
        """ flush summary writer when finished """
        if self.write:
            self.writer.close()


class CNN(NN):
    """
    represents a convolutional neural net
    """

    def parse_units(self, units):
        """
        parse units to list of ints

        Args:
            units (str): number of units, comma-separated
        """
        try:
            self.units = []
            for u in units.split(','):
                try:
                    self.units.append(int(u))
                except ValueError:
                    spl = u.split('x')
                    self.units.extend([int(spl[1])]*int(spl[0]))
        except:
            raise ValueError('invalid format for units, requires '
                             'comma-separated list of integers')

    def build_tower(self, i, batch_norm=False):
        """
        build convolutional layers and loss function for CNN

        Args:
            i (int): tower number
            batch_norm (bool): whether or not to perform batch normalization
        """
        self.layers = [tf.expand_dims(self.seqs[i], 2)]
        self.weights = []

        # conv layers
        for j in range(len(self.units)):
            if j == 0:
                result = create_conv_layer(self.layers[j], s.SEQ_DIM,
                                           self.units[j], name='conv%d' % j,
                                           bias=False, norm=batch_norm)
            else:
                result = create_conv_layer(self.layers[j], self.units[j-1],
                                           self.units[j], name='conv%d' % j,
                                           norm=batch_norm)
            self.weights.append(result[0])
            self.layers.append(result[1])
        self.layers.append(tf.concat(self.layers, 3, name='combine_layers'))
        self.shape = tf.shape(self.layers[-1])
        n = sum(self.units)+s.SEQ_DIM
        
        # dropout
        self.layers.append(tf.nn.dropout(self.layers[-1], self.keep_prob))

        # fully connected layer
        result = create_fully_connected_layer(self.layers[-1],
                                              n * s.MAX_SEQ_LEN)
        self.weights.append(result[0])
        self.layers.append(result[1])
        
        # final prediction & loss
        dGhat = result[1]
        tf.add_to_collection('dGhats', dGhat)
        loss = tf.nn.l2_loss((dGhat-self.dGs[i])/self.dG_sterrs[i], name='l2loss')
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses', 'tower%d' % i),
                        name='tower_loss')

    def get_activation_profile(self, layer, unit, iters=1000,
                               learning_rate=1e-1, keepprob=1):
        """
        run gradient descent over possible input matrices to maximize
        activation of specified neuron

        Args:
            layer (int): layer number
            unit (int): unit number
            iters (int): number of iterations of gradient descent
            learning_rate (float): learning rate for gradient descent
            keepprob (float): dropout rate
        """
        if not self.write:
            raise ValueError('cannot get unit activation with write mode off')
        seq = np.random.uniform(size=[1, 8, 1, s.SEQ_DIM]).astype('float32')
        max_activation = tf.reduce_max(tf.slice(self.layers[layer],
                                       [0, 0, 0, unit], [1, s.SEQ_DIM, 1, 1]))
        grad_seq = tf.gradients(max_activation, self.seq)
        for i in range(iters):
            result = self.sess.run([grad_seq],
                                   feed_dict={self.seq: seq,
                                              self.keep_prob: keepprob})
            seq += learning_rate*result[0]
        seq_summ = tf.summary.image('activation/seq%d-%d' % (layer, unit),
                                    tf.transpose(self.seq, perm=[0, 3, 1, 2]))
        self.writer.add_summary(self.sess.run([seq_summ],
                                              feed_dict={self.seq: seq,
                                                         self.keep_prob: 1}))
        return

    def get_weights(self, layer):
        """
        get weight of given layer/unit

        Args:
            layer (int): layer number
        """
        return self.weights[layer].eval(self.sess)


class RNN(NN):
    """
    represents a recurrent neural net
    """

    def parse_units(self, units):
        """
        parses input in format LxU into number of layers L and number of units
          per layer U

        Args:
            units (str): string in format LxU describing number of layers and
                units per layer
        """
        try:
            if 'x' in units:
                spl = units.split('x')
                self.layers = int(spl[0])
                self.units = int(spl[1])
            else:
                self.layers = 1
                self.units = int(units)
        except:
            raise ValueError('invalid format for units, requires either one '
                             'int for single layer RNN or LxU where L is '
                             'number of layers and U is number of units')

    def build_tower(self, i, bidirectional=False):
        """ build graph for a single tower

        Args:
            i (int): tower number
            bidirectional (bool): whether or not to use bidirectional network

        Returns:
            tf.Tensor: gradients
        """
        lengths = seq_lengths(self.seqs[i])
            
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(self.units, reuse=i > 0),
            output_keep_prob=self.keep_prob) for _ in range(self.layers)])
        if bidirectional:
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                     tf.contrib.rnn.LSTMCell(self.units, reuse=True),
                     output_keep_prob=self.keep_prob) 
                 for _ in range(self.layers)])

        if bidirectional:
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw,
                self.seqs[i], dtype=tf.float32, sequence_length=lengths)
            self.fc_sizes = [4 * self.layers * self.units]
            final_state = tf.transpose(final_state, [3, 0, 1, 2, 4])
        else:
            outputs, final_state = tf.nn.dynamic_rnn(cell, self.seqs[i],
                dtype=tf.float32, sequence_length=lengths)
            self.fc_sizes = [2 * self.layers * self.units]
            final_state = tf.transpose(final_state, [2, 0, 1, 3])
        
        final_state = tf.reshape(final_state,
                                 [-1, self.fc_sizes[0]])

        # fully connected layers to get final predictions
        self.fc_sizes.extend([120, 84, 1])
        self.fc = [final_state]
        self.fc_weights = []
        for j in range(len(self.fc_sizes)-1):
            fc_layer = create_fully_connected_layer(self.fc[-1],
                self.fc_sizes[j], self.fc_sizes[j+1],
                relu=(j != len(self.fc_sizes)-2),
                name='fc%d' % j)
            self.fc_weights.append(fc_layer[0])
            self.fc.append(fc_layer[1])
        dGhat = tf.squeeze(self.fc[-1], name='dGhat')
        tf.add_to_collection('dGhats', dGhat)

        # loss function
        loss = tf.nn.l2_loss((dGhat-self.dGs[i])/self.dG_sterrs[i], name='l2loss')
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses', 'tower%d' % i),
                        name='tower_loss')

    
    def get_max_activator(self, layer, unit, learning_rate=1e-3,
                          grad_stop=1e-1, l=10, alpha=1e-1):
        """
        get input that results in strongest activation of specified unit

        Args:
            layer (int): layer number
            unit (int): unit number
            learning_rate (float): learning rate for gradient descent
            grad_stop (float): gradient at which to stop descent
            l (int): length of sequence to use for input
            alpha (float): regularization parameter

        Return:
            np.array: resulting sequence matrix
        """
        if unit >= self.fc_sizes[layer]:
            raise ValueError('only %d units in layer %d'
                             % (self.fc_sizes[layer], layer))

        # generate random seq
        seq = np.concatenate((np.random.uniform(size=[1, l, len(s.bases)]),
                              np.zeros((1, s.MAX_SEQ_LEN-l, len(s.bases)))),
                             axis=1).astype('float32')

        # get activation of desired neuron and gradients
        activation = tf.slice(self.fc[layer], [0, unit], [1, 1])
        grad_seq = tf.gradients(activation, self.seq)[0]
        result = [np.ones(1), np.ones(1)]
        i = 0
        while (np.any(np.abs(result[0]) > grad_stop) or
               np.any(np.abs(result[1]) > grad_stop)):
            result = self.sess.run(grad_seq,
                                   feed_dict={self.seq: seq,
                                              self.keep_prob: 1})
            
            # add in regularization
            # L1
            result = result + np.sign(seq) * alpha
            ## L2
            #result = result + 2 * seq * alpha

            # update
            seq = seq - learning_rate * result
            i += 1
            if i > 5000:
                print 'break at 5000 iterations'
                break
        print '%d iterations' % i

        if self.write:
            seq_summ = tf.summary.image('activation/seq%d-%d' % (layer, unit),
                np.expand_dims(np.transpose(seq[:, 0:l, :], [0, 2, 1]), -1))
            self.writer.add_summary(self.sess.run([seq_summ])[0])
        return seq[:, 0:l, :]

    def get_unit_activations(self, seq):
        """
        get the activations of each unit at each point along the length
        of the sequence

        Args:
            seq (np.array): input sequence matrix

        Returns:
            np.array: activations
        """
        activations = np.zeros((s.MAX_SEQ_LEN, 2*self.units*self.layers))
        for i in range(s.MAX_SEQ_LEN):
            trunc_seq = np.copy(seq)
            trunc_seq[:, i+1:, :] = 0
            activations[i, :] = self.fc[0].eval(session=self.sess,
                feed_dict={self.seq: trunc_seq, self.keep_prob: 1.})
        return activations
