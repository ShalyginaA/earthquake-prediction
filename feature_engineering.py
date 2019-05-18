# Code from https://www.kaggle.com/fernandoramacciotti/wavelet-features-xgb-bayesianopt

import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
import pywt
from tqdm.auto import tqdm
import gc
from joblib import Parallel, delayed



def wavelet_coeffs(x, wavelet='db9', level=9):
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    return coeffs

class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None, fs=4e6, wavelet='db9', level=9):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.fs = fs
        self.wavelet = wavelet
        self.level = level
        self.test_files = []
        if self.dtype == 'train':
            self.filename = 'train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, 'test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                if df.time_to_failure.values[0] > df.time_to_failure.values[-1]:
                    x = df.acoustic_data.values
                    y = df.time_to_failure.values[-1]
                    seg_id = 'train_' + str(counter)
                    yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values
                yield seg_id, x, -999
                
    def features(self, x, y, seg_id, fs=4e6):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id
        coeffs = wavelet_coeffs(x, wavelet=self.wavelet, level=self.level)
        coeffs_diff = wavelet_coeffs(np.diff(x), wavelet=self.wavelet, level=self.level)
        percentiles_ranges = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 80, 75, 
                              50, 25, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        signals_list = {
            'regular': coeffs, 
            'diff': coeffs_diff,
        }
        for signal_name, signal_i in signals_list.items():
            for i, x_i in enumerate(signal_i):
                if i == 0:
                    name = '{}_cA'.format(signal_name)
                else:
                    name = '{}_cD{}'.format(signal_name, self.level - (i - 1))
                # statistics and centered moments
                feature_dict['rms_{}'.format(name)] = np.sqrt(np.mean(np.sum(x_i ** 2)))
                feature_dict['mean_{}'.format(name)] = np.mean(x_i)
                feature_dict['median_{}'.format(name)] = np.median(x_i)
                feature_dict['var{}'.format(name)] = np.var(x_i)
                feature_dict['skewnes_{}'.format(name)] = stats.skew(x_i)
                feature_dict['kurtosis_{}'.format(name)] = stats.kurtosis(x_i)
                # non-centered moments
                for m in range(2, 5):
                    feature_dict['moment_{}_{}'.format(m, name)] = np.mean(np.sum(x_i ** m))
                # percentile ranges
                for pct in percentiles_ranges:
                    feature_dict['percentile{}_{}'.format(str(pct), name)] = np.percentile(x_i, pct)
                # sum of energy of coefficients within bands
                chunks = 20
                step = len(x_i) // chunks
                for chunk_no, band in enumerate(range(0, len(x_i), step)):
                    feature_dict['energy_chunk{}_{}'.format(chunk_no, name)] = np.sum(x_i[band:band+step] ** 2)
                    feature_dict['energy_chunk_rms{}_{}'.format(chunk_no, name)] = np.sqrt(
                        np.mean(
                            np.sum(
                                feature_dict['energy_chunk{}_{}'.format(chunk_no, name)] ** 2)
                        )
                    )
                
        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs)(delayed(self.features)(x, y, s, fs=self.fs)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)