# RiboNet

This software allows for the training of reucrrent and convolutional neural network architectures for modeling RNA sequences

# Setup

This software requires the following Python packages: `pandas`, `numpy`, and `tensorflow`.

# Usage

**train_ribonet_kd.py**

Example:

```python train_ribonet_kd.py examples/train.txt -b 5 2x512```

Full list of arguments:

```
positional arguments:
  filename              name of input file
  n_units               numbers of units in hidden layers in the form N,N,N
                        for CNN or DxN for RNN

optional arguments:
  -h, --help            show this help message and exit
  -t TESTFILE, --testfile TESTFILE
                        name of test file
  -o OPTIMIZER, --optimizer OPTIMIZER
                        optimizer type (either "gradient", "adam", "adagrad",
                        or "rmsprop")
  -e EPOCHS, --epochs EPOCHS
                        number of training epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of data points per gpu in each training batch
  -k KEEPPROB, --keepprob KEEPPROB
                        probability to keep a node in dropout
  --reporter            whether or not to include reporter sequence in input
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for model training
  --batch_norm          enable batch normalization
  -r RESTORE, --restore RESTORE
                        file to restore weights from
  -w, --write           write to log file
  -s, --save            save variables
  -m MODEL, --model MODEL
                        cnn or rnn
  -d SEED, --seed SEED  random seed
  -g NUM_GPUS, --num_gpus NUM_GPUS
                        number of gpus to use
  --bidirectional       if rnn, whether or not it is bidirectional
  --lowmem              low memory setting
  --sterr               whether or not to use standard error in loss
                        computation
```
