# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:55:05 2023

@author: s3977226
"""
import glob
import html
import json
import logging
import os
import re
import string
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from nltk.corpus import stopwords
from overrides import overrides

# import SemEval
# from SemEval.models.semeval_classifier import SemEvalClassifier
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EMBEDDING_DIM = 100
HIDDEN_DIM = 200


@DatasetReader.register("SemEval2017-Task4-SubsetA")
class SemEvalDatasetReader(DatasetReader): #类 SemEvalDatasetReader 继承自 AllenNLP 的 DatasetReader 类
    """
    Reads a JSON-lines file containing papers from SemEval2017 Task4 SubsetA.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None: 
        '''lazy: 是否延迟加载数据。
tokenizer: 用于文本分词的工具。
token_indexers: 用于将分词后的文本转换为索引的工具。'''
        super().__init__(lazy)
        self.SEPARATOR = "\t"
        # data id set
        self.data = set() #初始化一个集合用于存储数据的唯一标识符
        # stop words list
        self.stop = stopwords.words('english') + list(string.punctuation) + ['rt', 'via']
        # tokenizer
        self._tokenizer = tokenizer or WordTokenizer()
        # token_indexers
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        } #设置用于文本处理的分词器和索引器。

    @overrides
    def _read(self, folder_path: str):
        # read files below the folder
        files = glob.glob(os.path.join(folder_path, "*.txt")) #使用glob库搜索指定文件夹下的所有.txt文件

        for file_path in files:
            with open(cached_path(file_path), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s",
                            file_path)
                for index, line in enumerate(data_file):
                    columns = line.rstrip().split(self.SEPARATOR) #移除每行末尾的空白字符，并以self.SEPARATOR（之前定义的制表符）为分隔符将行分割成列
                    if not columns:
                        continue
                    if len(columns)<3:
                        logger.info(index)
                        logger.info(columns)
                    tweet_id = columns[0]
                    sentiment = columns[1]
                    text = columns[2:]
                    text = self.clean_text(''.join(text)) #将文本数据拼接成一个字符串，并使用clean_text方法进行清洗
                    if tweet_id not in self.data:
                        self.data.add(tweet_id)
                        yield self.text_to_instance(sentiment, text) #使用text_to_instance方法处理文本和情感标签，并使用yield关键字返回结果
                    else:
                        continue

    @overrides
    #数据预处理：将文本数据和相应的情感标签转换成AllenNLP框架可以处理的格式
    def text_to_instance(self, sentiment: str,
                         text: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text) #使用类中定义的分词器（_tokenizer）对文本进行分词
        text_field = TextField(tokenized_text, self._token_indexers) 
        #TextField是AllenNLP中的一个类，用于处理和表示文本数据。self._token_indexers是用来将分词后的文本转换为模型能够处理的数值形式。
        fields = {'tokens': text_field}
        if sentiment is not None:
            fields['label'] = LabelField(sentiment)
        return Instance(fields)

    def clean_text(self, text: str):
        """
        Remove extra quotes from text files and html entities
        Args:
            text (str): a string of text
        Returns: (str): the "cleaned" text
        """
        text = text.rstrip() #移除 text 字符串尾部的所有空白字符（例如空格、制表符、换行符等)
        if '""' in text:
            if text[0] == text[-1] == '"':
                text = text[1:-1] #如果文本以双引号开头和结尾，这行代码将去除这两个双引号
            text = text.replace('\\""', '"') #将文本中的转义双引号 \"" 替换为普通的双引号
            text = text.replace('""', '"')

        text = text.replace('\\""', '"')

        text = html.unescape(text)

        text = ' '.join(text.split())
        return text

'''
执行 clean_text 方法后，输出的文本 text 将具有以下特点或格式：

去除多余引号：

如果文本以双引号开头和结尾，这些引号将被移除。
在文本中出现的连续双引号（如 ""）将被单个双引号替换。

去除转义字符：
特殊的转义字符（如 \"）将被处理，以确保文本不包含这些非标准格式。

HTML 实体解码：
任何 HTML 实体（如 &amp;、&lt; 等）将被转换为相应的字符（如 &、< 等）。

空格标准化：
连续的空格将被替换为单个空格。
文本的前后空格将被去除，以确保文本没有前导或尾随空格。''' 