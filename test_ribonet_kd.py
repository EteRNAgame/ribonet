import argparse
import nn as n
import dataset as d
import settings as s
import os
import re


def parse_args():
    """parse command line arguments
        
    Returns:
        obj: attributes are parsed arguments
    """
    parser = argparse.ArgumentParser(description='test a neural network for '
                                     'predicting RNA-reporter affinity')
    parser.add_argument('filename', help='name of input file for testing')
    parser.add_argument('restore', help='file to restore weights from')
    parser.add_argument('-n', '--n_units', help='numbers of units in hidden '
                        'layers')
    parser.add_argument('-o', '--optimizer', help='optimizer type (either '
                        '"descent", "adam", "adagrad", or "rmsprop")',
                        default='rmsprop')
    parser.add_argument('-m', '--model', help='cnn or rnn', default='rnn')
    parser.add_argument('--reporter', help='whether or not to include '
                        'reporter sequence in input', action='store_true')
    parser.add_argument('--lowmem', help='low memory setting',
                        action='store_true')
    return parser.parse_args()


def main():
    """ test existing model on given data """
    args = parse_args()
    data = d.KdDataset(args.filename.split(','), reporter=args.reporter,
                       lowmem=args.lowmem)

    # get number of units if not provided
    if args.n_units is None:
        m = re.search('([0-9]+x[0-9]+)units', args.restore)
        try:
            args.n_units = m.group(1)
        except:
            raise ValueError('unable to parse number of units from model name')

    # get model
    if args.model == 'cnn':
        print 'building cnn...'
        model = n.CNN(args.n_units, optimizer=args.optimizer, lowmem=args.lowmem)
    elif args.model == 'rnn':
        print 'building rnn...'
        model = n.RNN(args.n_units, optimizer=args.optimizer, lowmem=args.lowmem)
    else:
        raise ValueError('model must be "cnn" or "rnn"')
    model.restore(args.restore)

    data.get_predictions(model)
    print 'rmse: %.4f' % data.get_rmse(sterr=False)
    files = '+'.join([os.path.splitext(os.path.basename(f))[0]
                      for f in args.filename.split(',')])
    f = '%s/%s_%s.txt' % (s.RESULTS_DIR, os.path.basename(args.restore), files)
    data.to_file(f)
    print 'results written to %s' % f


if __name__ == '__main__':
    main()
