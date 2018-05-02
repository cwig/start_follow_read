from kaldi.hmm import TransitionModel
from kaldi.fstext import read_fst_kaldi, SymbolTable
from kaldi.decoder import FasterDecoderOptions, FasterDecoder, DecodableMatrixScaledMapped
from kaldi.util.io import xopen
from kaldi.matrix import Matrix
from kaldi.fstext.utils import get_linear_symbol_sequence

from lm_stats import LMStats
import numpy as np

import codecs
import re

KALDI_SPECIAL_SYM = {
    "<space>": " ",
    "<unk>": "",
    "<eps>": ""
}
DICT_MORPH = {'[': 'LSB', ']': 'RSB', ';': 'SEMI', ':': 'COLON', '_': 'USCORE',
   '!': 'EXCL', '=': 'EQUALS', '~': 'TILDE', '(': 'LRB', ')': 'RRB',
   '\'': 'APOS', '\"': 'QUOTE', '#': 'POUND', '+': 'PLUS', '%': 'PCT', '^': 'CARAT',
   '@': 'ATSIGN', '?': 'QMARK', '>': 'LTSIGN', '<': 'GTSIGN', '$': 'DOLLAR', '&': 'AND',
   '*': 'STAR', ',': 'COMMA', '.': 'PERIOD', '/': 'SLASH', ' ': 'SPACE', '0': 'NUM0',
   '1': 'NUM1', '2': 'NUM2', '3': 'NUM3', '4': 'NUM4', '5': 'NUM5', '6': 'NUM6',
   '7': 'NUM7', '8': 'NUM8', '9': 'NUM9', '-': 'DASH'
}

MIN_WEIGHT = np.log(5.0e-16)

def kaldi2str_single(kaldi_out):
    return u"".join([KALDI_SPECIAL_SYM.get(k, k) for k in kaldi_out])


def create_phone_map( filename, idx_to_char):

    #Old code for parsing. We can read in as utf8 directly with this method
    # dictSave = {}
    # with codecs.open(filename,'r',encoding='utf8') as f:
    #     data = f.read()
    #
    # for index, text in enumerate(data.split("\n")):
    #     entries = re.split('\s', text, 2)
    #     if (len(entries)<2 or len(entries[1])==0):
    #         continue
    #     dictSave[entries[0]] = int(entries[1])
    #
    # dictSave['EPS'] = dictSave['NON']
    # for key in DICT_MORPH:
    #     if (dictSave.get(DICT_MORPH[key],None) is None):
    #         continue
    #     dictSave[key] = dictSave[DICT_MORPH[key]]

    # I perfer to used the library to parse the symbol
    # table, but it doesn't read in as utf8
    ph_to_idx = {}
    phone_table = SymbolTable.read_text(filename)
    for i in range(phone_table.num_symbols()):
        phone_sym = phone_table.find_symbol(i).decode('utf8')
        ph_to_idx[phone_sym] = i

    ph_to_idx['EPS'] = ph_to_idx['NON']
    for key in DICT_MORPH:
        if ph_to_idx.get(DICT_MORPH[key],None) is None:
            continue
        ph_to_idx[key] = ph_to_idx[DICT_MORPH[key]]

    reorder_1 = []
    reorder_2 = []
    for pyphnid in range(len(idx_to_char)+1):
        if pyphnid == 0:
            a = "EPS"
        else:
            a = idx_to_char[pyphnid]
            a = DICT_MORPH.get(a,a)
        newa = ph_to_idx.get(a,None)
        if newa == None:
            continue

        reorder_1.append(newa-1)
        reorder_2.append(pyphnid)

    reorder_1 = np.array(reorder_1)
    reorder_2 = np.array(reorder_2)

    return reorder_1, reorder_2

class LMDecoder(object):
    def __init__(self, idx_to_char, params={}):

        self.idx_to_char = idx_to_char

        self.reorder_1, self.reorder_2 = create_phone_map(params['phones_path'], idx_to_char)
        self.word_syms = SymbolTable.read_text(params['words_path'])

        self.acoustic_scale = params.get('acoustic', 1.2)
        if self.acoustic_scale < 0:
            print "Warning: acoustic scale is less than 0"
        allow_partial = params.get('allow_partial', True)
        beam = params.get('beam', 13)
        self.alphaweight = params.get('alphaweight', 0.3)

        trans_model = TransitionModel()
        with xopen(params['mdl_path']) as ki:
            trans_model.read(ki.stream(), ki.binary)

        decoder_opts=FasterDecoderOptions()
        decoder_opts.beam = beam

        decode_fst = read_fst_kaldi(params['fst_path'])

        self.decoder_opts = decoder_opts
        self.trans_model = trans_model
        self.decode_fst = decode_fst

        self.stats = LMStats()
        self.stats_state = None
        self.add_stats_phase = True

    def decode(self, data, as_idx=False):
        if self.add_stats_phase:
            self.stats_state = self.stats.get_state()
            self.stats.reset()
            self.add_stats_phase = False

        if len(data.shape) == 3:
            return [ self.decode_one(d, as_idx) for d in data ]
        else:
            d = self.decode_one(data, as_idx)
            return d

    def decode_one(self, data, as_idx=False):

        #Reweight and reorder for LM
        reweighted = self.stats_state.reweight(data, self.alphaweight)
        reweighted = reweighted[:,self.reorder_2]

        reweighted_prime = np.full((reweighted.shape[0], self.reorder_1.max()+1), MIN_WEIGHT, dtype=np.float32)
        reweighted_prime[:,self.reorder_1] = reweighted

        #Apply LM
        reweighted = Matrix(reweighted_prime)
        decoder = FasterDecoder(self.decode_fst, self.decoder_opts)
        decodable = DecodableMatrixScaledMapped(self.trans_model, reweighted, self.acoustic_scale)
        decoder.decode(decodable)
        best_path = decoder.get_best_path()
        alignment, words, weight = get_linear_symbol_sequence(best_path)

        #Parse LM output
        kaldi_unicode = kaldi2str_single([self.word_syms.find_symbol(w).decode('utf8') for w in words])

        return kaldi_unicode, 0

    def add_stats(self, data):
        if not self.add_stats_phase:
            print "Reseting lm stats because more stats added after a decoding"
            self.add_stats_phase = True
        self.stats.add_stats(data)
