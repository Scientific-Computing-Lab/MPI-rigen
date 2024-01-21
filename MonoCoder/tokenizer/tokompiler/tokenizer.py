import pickle
from typing import List, Union


class BatchEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    @property
    def ids(self):
        return self.input_ids


class Tokompiler:
    '''
        Compiler oriented tokenizer
    '''
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            tokens = [token[:-1] for token in f.readlines()]

        self.special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]', '[MSK]', '[SEP]', '[CLS]']

        with open(vocab_path, 'r') as f:
            tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]', '[MSK]', '[SEP]', '[CLS]'] + tokens

        self.encoder = {token:idx for idx, token in enumerate(tokens, start=1)}
        self.decoder = {val:key for key, val in self.encoder.items()}

        self.do_padding = False
        self.do_truncation = False

        self.length = 256
        self.max_length = 256

        self.post_processor = None

    def __len__(self):
        return len(self.encoder)

    def __str__(self):
        return self.encoder

    def get_vocab_size(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self.encoder

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self.decoder

    @property
    def unk(self):
        return self.encoder['[UNK]']

    @property
    def cls(self):
        return self.encoder['[CLS]']

    @property
    def sep(self):
        return self.encoder['[SEP]']

    @property
    def pad(self):
        return self.encoder['[PAD]']

    @property
    def sos(self):
        return self.encoder['[SOS]']

    @property
    def eod(self):
        return self.encoder['[EOS]']

    @property
    def mask(self):
        return self.encoder['[MSK]']

    def enable_padding(self, length):
        """
        Enable padding for encodings.
        """
        self.do_padding = True
        self.length = length

    def no_padding(self):
        """
        Disable padding for encodings.
        """
        self.do_padding = False

    def enable_truncation(self, max_length):
        """
        Enable padding for encodings.
        """
        self.do_truncation = True
        self.max_length = max_length

    def no_truncation(self):
        """
        Disable padding for encodings.
        """
        self.do_truncation = False

    def add_tokens(self, new_tokens: List[str]) -> int:
        """
        Add a list of new tokens to the tokenizer class. 
        If the new tokens are not in the vocabulary, they are added to it with indices starting from 
        length of the current vocabulary.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token 
        embedding matrix of the model so that its embedding matrix matches the tokenizer.
        """
        num_tokens_added = 0
        max_id = max(self.decoder.keys())

        for token in new_tokens:
            if token in self.encoder:
                continue

            num_tokens_added += 1
            idx = max_id + num_tokens_added

            self.encoder[token] = idx
            self.decoder[idx] = token

        return num_tokens_added

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.
        """
        update_case = lambda token: token.lower() if ('ignore_case' in kwargs and kwargs['ignore_case']) and token not in self.special_tokens else token
        return [update_case(token)  for token in text.split()]

    def token_to_id(self, token):
        return self.encoder[token] if token in self.encoder else self.unk

    def id_to_token(self, id):
        return self.decoder[id]

    def post_process(self, ids):
        if self.post_processor == 'sos':
            ids = [self.sos] + ids
        elif self.post_processor == 'eos':
            ids = ids + [self.eod]

        return ids

    def encode(self, sequence: Union[str, List[str]], is_pretokenized=False, **kwargs) -> [List[int], List[int]]:
        """
        Encode a sequence to corresponding ids.

        Args:
            sequence (Union[str, List[str]]): Sequence to be encoded,
                when is_pre_tokenized is False, the type should be str,
                when is_pre_tokenized is True, the type should be List[str]
            is_pre_tokenized (bool): Whether the input is already pre-tokenized
        Returns:
            list[int], list[int]: indices and mask for sequence
        """
        if not is_pretokenized and isinstance(sequence, str):
            sequence = self.tokenize(sequence, **kwargs)

        ids = [self.token_to_id(token) for token in sequence]
        ids = self.post_process(ids)

        attention_mask = [1] * len(ids)

        if self.do_truncation:
            ids = ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        if self.do_padding:
            ids = ids[:self.length] + [self.pad] * (self.length - len(ids)) if len(ids) < self.length else ids[:self.length]
            attention_mask = attention_mask[:self.length] + [0] * (self.length - len(attention_mask)) if len(attention_mask) < self.length else attention_mask[:self.length]

        return ids, attention_mask

    def decode(self, ids: List[int], skip_special_tokens=False) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.
        """
        tokens = ' '.join([self.id_to_token(id) for id in ids if not (skip_special_tokens and id <= len(self.special_tokens))])

        return tokens

    def encode_batch(self, input:Union[List[str], List[List[str]]], is_pretokenized=False, **kwargs):
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        """
        encodings = []

        if 'pad' in kwargs and 'max_length' in kwargs:
            if kwargs['pad']:
                self.enable_padding(length=kwargs['max_length'])

        for sequence in input:
            input_ids, attention_mask = self.encode(sequence, is_pretokenized=is_pretokenized)
            encodings.append(BatchEncoding(input_ids, attention_mask))

        return encodings

    def decode_batch(self, sequences: List[List[int]], skip_special_tokens: bool, **kwargs) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.
        """
        return [self.decode(ids, skip_special_tokens) for ids in sequences]

    def save(self, tokenizer_path):
        """
        Save tokenizer as pickle
        """
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self, f)
