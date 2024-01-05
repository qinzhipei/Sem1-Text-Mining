#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
from typing import List



'''这个类被注册为 AllenNLP 的 Predictor，允许在配置文件中通过 'semeval-predictor' 名称被引用'''
@Predictor.register('semeval-predictor')

class SemEvalPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True) 
        #初始化一个 SpaCy 的词分割器 SpacyWordSplitter，用于将输入的句子分割成单词
    
    #定义了如何对一个字符串类型的句子进行预测
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance([str(t) for t in tokens])
    
    '''它首先将输入的文本句子分词，然后将分词后的数据转换为模型可以处理的格式，最后调用模型进行预测。'''