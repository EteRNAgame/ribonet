import numpy as np
import tensorflow as tf
import argparse
import nn as n
import dataset as d
import os
import settings as s


def parse_args():
    """parse command line arguments
        
    Returns:
        obj: attributes are parsed arguments
    """
    parser = argparse.ArgumentParser(description='train a neural network to '
                                     'predict RNA-reporter affinity')
    parser.add_argument('filename', help='name of input file')
    parser.add_argument('n_units', help='numbers of units in hidden layers in '
                        'the form N,N,N for CNN or DxN for RNN')
    parser.add_argument('-t', '--testfile', help='name of test file')
    parser.add_argument('-o', '--optimizer', help='optimizer type (either '
                        '"gradient", "adam", "adagrad", or "rmsprop")',
                        default='rmsprop')
    parser.add_argument('-e', '--epochs', help='number of training epochs',
                        type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='number of data points per '
                        'gpu in each training batch', type=int, default=128)
    parser.add_argument('-k', '--keepprob', help='probability to keep a node '
                        'in dropout', type=float, default=1.)
    parser.add_argument('--reporter', help='whether or not to include '
                        'reporter sequence in input', action='store_true')
    parser.add_argument('-l', '--learning_rate', help='learning rate for '
                        'model training', type=float, default=1e-4)
    parser.add_argument('--batch_norm', help='enable batch '
                        'normalization', action='store_true')
    parser.add_argument('-r', '--restore', help='file to restore weights from')
    parser.add_argument('-w', '--write', help='write to log file',
                        action='store_true')
    parser.add_argument('-s', '--save', help='save variables',
                        action='store_true')
    parser.add_argument('-m', '--model', help='cnn or rnn', default='rnn')
    parser.add_argument('-d', '--seed', help='random seed', default=None,
                        type=int)
    parser.add_argument('-g', '--num_gpus', help='number of gpus to use',
                        default=1, type=int)
    parser.add_argument('--bidirectional', help='if rnn, whether or not it '
                        'is bidirectional', action='store_true')
    parser.add_argument('--lowmem', help='low memory setting',
                        action='store_true')
    parser.add_argument('--sterr', help='whether or not to use standard error '
                        'in loss computation', action='store_true')
    return parser.parse_args()


def main():
    """train a neural network with given data"""
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # read data from file(s)
    print 'reading data...'
    data = d.KdDataset(args.filename.split(','), reporter=args.reporter,
                       lowmem=args.lowmem)
    print '\t%s points in training data' % data.n
    print '\tstandard deviation of train data: %.2f' \
        % np.nanstd(data.data.dG_measured)
    if args.testfile is not None:
        testdata = d.KdDataset(args.testfile.split(','), reporter=args.reporter,
                               lowmem=args.lowmem)
        print '\t%s points in testing data' % testdata.n
        print '\tstandard deviation of test data: %.2f' \
            % np.nanstd(testdata.data.dG_measured)
    else:
        testdata = None
    
    # construct neural network model object
    if args.model == 'cnn':
        print 'building cnn...'
        files = '+'.join([os.path.splitext(os.path.basename(f))[0]
                          for f in args.filename.split(',')])
        name = '%s_%s_%sunits_%.2e' % \
            (files, args.model, args.n_units, args.learning_rate)
        if args.keepprob != 1.:
            name += '_keepprob%.2f' % args.keepprob
        model = n.CNN(args.n_units, args.write, args.learning_rate,
                      args.optimizer, name, batch_norm=args.batch_norm,
                      lowmem=args.lowmem, num_gpus=args.num_gpus)
    elif args.model == 'rnn':
        print 'building rnn...'
        files = '+'.join([os.path.splitext(os.path.basename(f))[0]
                          for f in args.filename.split(',')])
        name = '%s_%s_%sunits_%.2e' % \
            (files, args.model, args.n_units, args.learning_rate)
        if args.keepprob != 1.:
            name += '_keepprob%.2f' % args.keepprob
        model = n.RNN(args.n_units, args.write, args.learning_rate,
                      args.optimizer, name, bidirectional=args.bidirectional,
                      lowmem=args.lowmem, num_gpus=args.num_gpus)
    else:
        raise ValueError('model must be "cnn" or "rnn"')

    # restore model from file if specified
    if args.restore is not None:
        print 'restoring model parameters from file...'
        model.restore(args.restore)

    # train model
    print 'training model %s...' % model.name
    loss = model.train(data, args.epochs, args.batch_size, args.keepprob,
                       testdata, args.sterr)
    print 'finished training model'

    # save loss to file
    if not os.path.exists(s.RESULTS_DIR):
        os.makedirs(s.RESULTS_DIR)
    np.savetxt('%s/%s.loss' % (s.RESULTS_DIR, model.name), loss.T, delimiter='\t',
               header='train\ttest', comments='')

    # run train and test predictions for final model
    print 'getting final predictions...'
    if testdata:
        testdata.get_predictions(model)
        testdata.to_file('%s/%s_test.txt' % (s.RESULTS_DIR, model.name))
        testrmse = testdata.get_rmse()
        print 'final test: %.4f' % testrmse
        print 'final test rmse: %.4f' % testdata.get_rmse(sterr=False)
    data.get_predictions(model)

    # save and print results
    data.to_file('%s/%s_train.txt' % (s.RESULTS_DIR, model.name))
    print 'final: %.4f' % data.get_rmse()
    print 'final rmse: %.4f' % data.get_rmse(sterr=False)
    if args.save:
        print 'saving model parameters to file %s...' % model.name
        model.save('_%f' % testdata.get_rmse(sterr=False))


if __name__ == '__main__':
    main()
