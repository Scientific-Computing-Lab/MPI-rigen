import lmppl_code
import pygments 
from pygments.lexers import get_lexer_by_name
import numpy as np 


import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from transformers import PretrainedConfig

from tokenizer import Tokompiler

from transformers import PreTrainedTokenizer
from typing import List


#WARNING inhertance made so many magic black box bugs I am just removing it dont add it without testing
class TokompilerHF():#PreTrainedTokenizer
    '''
        Hugging Face compatible version of Tokompiler
    '''
    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.tokompiler = Tokompiler(vocab_file)
        self.pad_token_id=self.tokompiler.encode('[PAD]') #DONT DARE CHANGE THIS!!!
        #self.pad_token_id=self.tokompiler.encoder['[PAD]']
        self.pad_token='[PAD]'

        #black magic for some reason removing the print changes
        #print(type(pad_token_id))
        #print(self.pad_token_id)
        #self.pad_token_id=None
        #self.pad_token_id=pad_token_id
        #print(self.pad_token_id) 
        #assert self.pad_token_id==pad_token_id
        #print(type(self.pad_token_id))
        #print(self.pad_token_id)
    
    def __call__(self,batch,**kwargs):
        seq=[self.tokompiler.encode(t) for t in batch]
        maxln=max(len(x) for x in seq)
        ans=[x+(maxln-len(x))*[self.pad_token_id[0]] for x in seq]
        mask=[len(x)*[1]+(maxln-len(x))*[0] for x in seq]
        #print(ans)
        return {'input_ids':torch.LongTensor(ans),'mask':torch.BoolTensor(mask)}        
    
    def _tokenize(self, text, **kwargs):
        return self.tokompiler.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokompiler.encode(token)

    def _convert_id_to_token(self, index):
        return self.tokompiler.decode([index])

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def _convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def _convert_ids_to_tokens(self, ids):
        return [self._convert_id_to_token(id) for id in ids]

    def get_vocab(self):
        return self.tokompiler.encoder.copy()

    @property
    def vocab_size(self):
        return len(self.tokompiler.encoder)
 



class DummyConfig(PretrainedConfig):
    model_type = "dummy_model"

    def __init__(self, vocab_size=30522, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class DummyModel(PreTrainedModel):
    def __init__(self, config):
        super(DummyModel, self).__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # initialize weights to zeros
        self.embeddings.weight.data.fill_(0)
        self.embeddings.weight.requires_grad = False

    def forward(self, input_ids, **kwargs):
        return {'logits':torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)}

    def get_input_embeddings(self):
        # return zero embeddings
        return self.embeddings

    def set_input_embeddings(self, value):
    	print('the embedings were set')
    	self.embeddings = value


# dummy model for causal language model
class DummyCausalModel(DummyModel, AutoModelForCausalLM):
    pass

# dummy model for sequence to sequence model
class DummySeq2SeqModel(DummyModel, AutoModelForSeq2SeqLM):
    pass




if __name__=='__main__':
    texts = [
    'I am happy.',
    'I am sad.  jussst stuff for padddingnngnng dd'
    ]
    count=lmppl_code.get_lex_count(texts,'c')

    config = DummyConfig(vocab_size=50000)
    model=DummyCausalModel(config)
    #model=AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer=TokompilerHF('tokenizer_vocab/vocab.txt')
    #tokenizer.pad_token_id=tokenizer.tokompiler.encode('[PAD]')[0]
    #print(tokenizer.pad_token_id[0])
    #print(tokenizer(texts))
    scorer = lmppl_code.LM(tokenizer=tokenizer, model_obj=model)
    print(scorer.get_perplexity(texts,count))


    tokenizer=AutoTokenizer.from_pretrained('gpt2')
    model=AutoModelForCausalLM.from_pretrained('gpt2')
    scorer = lmppl_code.LM(tokenizer=tokenizer, model_obj=model)
    print(scorer.get_perplexity(texts,count))

    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-small')
    model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    scorer = lmppl_code.EncoderDecoderLM(tokenizer=tokenizer, model_obj=model)
    print(scorer.get_perplexity(['' for _ in texts],texts,count))
    #print(decoder_only_perplexity(texts,'gpt2',count))
    #print(encoder_decoder_perplexity(texts,'google/flan-t5-small',count))

    config = DummyConfig(vocab_size=50000)
    model=DummySeq2SeqModel(config)
    #model=AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer=TokompilerHF('tokenizer_vocab/vocab.txt')
    print(scorer.get_perplexity(['' for _ in texts],texts,count))

