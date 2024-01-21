import sys
from nltk.translate.bleu_score import sentence_bleu
from code_bert_score import code_bert_score

sys.path.append("/mnt/lbosm1/home/Share/code-lms/polycoder/tasks/omp/hf/generations/analysis/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU")
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match



def calc_bleu(pred, label): 
    return sentence_bleu([label], pred)
    

def calc_code_bert_score(pred, label): 
    return code_bert_score.score(cands=[pred], refs=[label], lang='c')[2] # f1 score


def calc_code_bleu(pred, label): 
    lang = 'c_sharp'   # the closest language to C from choices=['java','js','c_sharp','php','go','python','ruby']
    alpha,beta,gamma,theta = 0.25,0.25,0.25,0.25

    # preprocess inputs
    pre_references = [[pred]]
    hypothesis = [label]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    return code_bleu_score

