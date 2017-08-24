import numpy as np
import pandas as pd
import multiprocessing as mp
import string
import re
import subprocess as sp
import os
import functools as ft
import pickle
import sys
import settings as s

kd_ms2 = 1e-8
rt = 1.987e-3*310
beta = 1/rt


def rand_string(n=8):
    """random string of lowercase characters of length n
        
    Args:
        n (int): length of string
    """
    return ''.join(np.random.choice(list(string.ascii_lowercase), n))


def get_base_matrix(seq):
    """create one hot matrix representing sequence
    
    Args:
        seq (str): sequence of bases
    
    Return:
        np.array: (MAX_SEQ_LEN x SEQ_DIM) binary array representing sequence
    """
    n = len(seq)
    mat = np.zeros((s.MAX_SEQ_LEN, s.SEQ_DIM))
    for i in range(n):
        if seq[i] == '&':
            continue
        mat[i, s.bases[seq[i]]] = 1
    return mat


def get_base_matrix_full(seq, oligos, reporter=None):
    """
    create one hot matrix representing sequence with oligos and reporter

    Args:
        seq (str): sequence of bases
        oligos (str): oligos in the form "conc1 sequence1, conc2 sequence2, ..."
        reporter (str): reporter in the form "conc sequence"

    Return:
        np.array: (MAX_SEQ_LEN x SEQ_DIM) array representing measurement
    """
    n = len(seq)
    mat = np.zeros((s.MAX_SEQ_LEN, s.SEQ_DIM))
    i = 0
    for oligo, conc in oligos_to_dict(oligos).iteritems():
        for char in oligo:
            mat[i, s.bases[char]] = np.log10(conc)
            i += 1
        i += 1
    for char in seq:
        mat[i, s.bases[char]] = 4
        i += 1
    i += 1
    if reporter is not None:
        spl = reporter.split()
        for char in spl[1]:
            mat[i, s.bases[char]] = np.log10(float(spl[0]))
            i += 1
    return mat


def oligos_to_dict(oligos):
    """turn oligo list as string into dictionary
        
    Args:
        oligos (str): oligos in the form
            "concentration1 sequence1, concentration2 sequence2, ..."

    Returns:
        dict: sequences as keys, concentrations as values
    """
    d = {}
    if not pd.isnull(oligos):
        try:
            for oligo in oligos.split(','):
                spl = oligo.split()
                d[spl[1]] = float(spl[0])
        except:
            raise ValueError('invalid oligo specification, should be a comma-'
                             'separated series of concentration/sequence pairs'
                             ', e.g. "10 AUGC, 5 CGUA"')
    return d


class Dataset(object):
    """
    stores all information related to an experimental dataset
    """

    def __init__(self, filenames, fields, lowmem=False):
        """intialize dataset with data from given filenames

        Args:
            filenames (list): list of strings containing filenames of
                tab-delimited data
            fields (list): fields to extract from files
            lowmem (bool): whether or not to use low memory mode
        """
        self.pickle = pickle
        self.copy = False

        self.data = self.read_data(filenames, fields, lowmem)

    def read_data(self, filenames, fields, lowmem=False):
        """
        read in specified fields from given filenames

        Args:
            filenames (list): list of strings containing filenames of
                tab-delimited data
            fields (list): fields to extract from files
            lowmem (bool): whether or not to use low memory mode

        Raises:
            ValueError: if no data can be read
        """
        data = pd.DataFrame()
        for filename in filenames:
            if not os.path.isfile(filename):
                print '%s not found, skipping dataset' % filename
                continue
            new_data = pd.read_csv(filename, sep='\t', memory_map=lowmem)
            try:
                new_data = new_data[fields]
            except KeyError as e:
                print '%s not read' % filename
                print e
                continue
            data = data.append(new_data)
        data = data.reset_index(drop=True)
        if data.shape[0] == 0:
            raise ValueError('no data read, check filenames')
        return data

    def to_file(self, filename, fields):
        """ write data to tab delimited file
            
        Args:
            filenames (str): name of file to write to
            fields (list): fields to write
        """
        self.data[fields].to_csv(filename, sep='\t', index=False)


class KdDataset(Dataset):
    """
    represents a set of experimental KD measurements
    """

    def __init__(self, filenames, reporter=True, lowmem=False):
        """
        read in dG measurements from given filenames and calculate ensembles
        and motifs in given sequences
        
        Args:
            filenames (list): list of strings containing filenames of
                tab-delimited data
            reporter (bool): whether or not to include reporter in input
                sequence
            lowmem (bool): whether or not to use low memory mode
        """
        # read in sequence and dG data from given filenames
        fields = ['Sequence', 'dG_measured', 'dG_sterr', 'oligos', 'reporter']
        Dataset.__init__(self, filenames, fields, lowmem)
        self.n = self.data.shape[0]
        if reporter:
            self.data['seq_matrix'] = self.data.apply(
                lambda row: [get_base_matrix_full(row.Sequence, row.oligos,
                                                  row.reporter)], axis=1)
        else:
            self.data['seq_matrix'] = self.data.apply(
                lambda row: [get_base_matrix_full(row.Sequence, row.oligos)],
                axis=1)
        self.data['seq_matrix'] = self.data.seq_matrix.apply(lambda x: x[0])

    def to_file(self, filename):
        """ write data to tab delimited file
            
        Args:
            filename (str): name of file to write to
        """
        Dataset.to_file(self, filename, ['Sequence', 'dG_measured',
                                         'dG_predicted', 'dG_sterr', 'oligos'])

    def get_predictions(self, model, indices=None):
        """get dG predictions for all data using given model

        Args:
            model (obj): object that has test() function to take a sequence
                matrix and make dG prediction
            indices (list): list of indices to get predictions for, if not all
        """
        if indices is None:
            indices = self.data.index.values
        seq = self.get_data(indices)[0]
        self.data.ix[indices, 'dG_predicted'] = model.test(seq).flatten()

    def get_rmse(self, indices=None, sterr=False):
        """get rmse of dGs
            
        Args:
            indices (list): list of indices to evaluate RMSE over
            sterr (bool): whether or not to take standard errors into account

        Returns:
            float: RMSE
        """
        if indices is None:
            indices = self.data.index.values
        indices = [i for i in indices
                   if np.isfinite(self.data.ix[i, 'dG_predicted'])]
        err = self.data.ix[indices, 'dG_measured'] - \
            self.data.ix[indices, 'dG_predicted']
        if sterr:
            err /= self.data.ix[indices, 'dG_sterr']
        return np.sqrt(np.mean(np.square(err)))

    def get_train_batch(self, n=100, sterr=False):
        """get batch of data
        
        Args:
            n (int): size of batch
            sterr (bool): whether or not to get standard errors

        Returns:
            tuple: two or three np.ndarrays depending on sterr flag
        """
        i = np.random.choice(self.data.index.values, n,
                             replace=False)
        return self.get_data(i, sterr)

    def get_data(self, i=None, sterr=False):
        """returns sequence matrix, dGs for given indices
            
        Args:
            i (list): indicies of data to get
            sterr (bool): whether or not to get standard errors

        Returns:
            tuple: two or three np.ndarrays depending on sterr flag
        """
        if i is None:
            i = self.data.index.values 
        if sterr:
            return (np.stack(self.data.ix[i, 'seq_matrix'].values, axis=0),
                    self.data.ix[i, 'dG_measured'].values,
                    self.data.ix[i, 'dG_sterr'].values)
        else:
            return (np.stack(self.data.ix[i, 'seq_matrix'].values, axis=0),
                    self.data.ix[i, 'dG_measured'].values)
