
import pandas as pd
import re
import string

from nltk.corpus import stopwords

'''本文件包含了一些定义和辅助函数，包括
定义积极和消极情绪的表情符号列表。
正则表达式和函数来匹配和提取文本中的不同元素（如表情符号、单词、URL等）。
预处理函数来处理文本，包括分词和去除停用词等。
相关文件的路径和文件名定义。
utils.py 提供了一系列工具和定义，这些都可以在 datareader.py 中使用，来进行数据读取与转换
'''




# Emoticons
#定义了表示积极和消极情绪的表情符号
pos_emo = [':-)', ':)', '(-:', '(:', ':-]', '[-:', ':]', '[:', ':-d', ':>)', ':>d', '(<:', ':d', 'b-)', ';-)',
                '(-;', ';)', '(;', ';-d', ';>)', ';>d', '(>;', ';]', '=)', '(=', '=d', '=]', '[=', '(^_^)', '(^_~)',
                '^_^', '^_~', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '<-:', '<:', '(-8', '(8',
                '{-:', '{:', '(o:', '(^:', '=->', '=>', '=-}', '=}', '=o)', '=c)', '=^)', '<-=', '<=', '{-=', '{=',
                '(o=', '(^=', '8-]', '8]', ':o]', ':c]', ':^]', '[-8', '[8', '[o:', '[^:', '=o]', '=c]', '=^]', '[o=',
                '[^=', '8‑d', '8d', 'x‑d', 'xd', ':-))', '((-:', ';-))', '((-;', '=))', '((=', ':p', ';p', '=p']
neg_emo = ['#-|', ':-&', ':&', ':-(', ')-:', '(t_t)', 't_t', '8-(', ')-8', '8(', ')8', '8o|', '|o8', ':$', ':\'(',
                ':\'-(', ':(', ':-/', ')\':', ')-\':', '):', '\-:', ':\'[', ':\'-[', ':-[', ']\':', ']-\':', ']-:',
                '=-(', '=-/', ')\'=', ')-\'=', ')-=', '\-=', ':-<', ':-c', ':-s', ':-x', ':-|', ':-||', ':/', ':<',
                ':[', ':o', ':|', '=(', '=[', '=\'(', '=\'[', ')=', ']=', '>-:', 'x-:', '|-:', '||-:', '\:', '>:', ']:',
                'o:', '|:', '=|', '=x', 'x=', '|=', '>:(', ':((', '):<', ')):', '>=(', '=((', ')=<', '))=', ':{', ':@',
                '}:', '@:', '={', '=@', '}=', '@=', 'd‑\':', 'd:<', 'd:', 'd8', 'd;', 'd=', 'd‑\'=', 'd=<', 'dx']

#处理后的文件存放的目录
DATA_DIR = '../data/GOLD/Subtask_A/'
PROC_DIR = '../data/processed/'
EVAL1_DIR = '../data/Dev/'
EVAL2_DIR = '../data/Final/'

TRAIN = 'twitter-2016train-A.txt'
TEST = 'twitter-2016test-A.txt'
DEV = 'twitter-2016dev-A.txt'
DEVTEST = 'twitter-2016devtest-A.txt'
EVAL1 = 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
EVAL2 = 'SemEval2017-task4-test.subtask-A.english.txt'


#匹配表情符号
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

 #其他标记模式
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE) #用于匹配和提取包含各种不同元素（如表情符号、单词、URL等）的文本片段
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE) #精确匹配单独的表情符号




def tokenize(s):
    return tokens_re.findall(s) #调用 findall 方法在字符串 s 中查找所有匹配的模式。它返回一个列表，其中包含所有匹配项


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def main():

    data = {
        'train': DATA_DIR + TRAIN,
        'test': DATA_DIR + TEST,
        'dev': DATA_DIR + DEV,
        'devtest': DATA_DIR + DEVTEST,
        'eval-dev': EVAL1_DIR + EVAL1,
        'eval-final': EVAL2_DIR + EVAL2
        }

    punctuation = list(string.punctuation) #所有标准的英文标点符号
    stop = stopwords.words('english') + punctuation + ['rt', 'via'] #包含英语停用词、标点符号以及一些特定词汇（如 rt 和 via，这些常在推特文本中出现）的列表

    for dataset in data: #遍历 data 字典中的每个数据集
        #print(dataset)
        with open(data[dataset], 'r') as dataset_f:
            output_data = []
            for line in dataset_f:
                #print(line.split('\t'))

                info = line.strip().split('\t')
                id, label, text = info[0], info[1], ' '.join(info[2:]) #分离出 id（推特ID）、label（标签）和 text（推文内容）

                # process text 分词和标准化文本
                tokens = preprocess(text)

                # remove stopwords and others 移除停用词、标点符号以及特定词汇。
                tokens = [term.lower() for term in tokens if term.lower() not in stop]

                # remove hashtags 移除以 # 开头的话题标签和以 @ 开头的提及。
                tokens = [term for term in tokens if not term.startswith('#')]

                # remove profiles
                tokens = [term for term in tokens if not term.startswith('@')]

                

         #保存处理后的数据,并写入 CSV 文件
                d = {
                    'id': id,
                    'label': label,
                    'text': ' '.join(tokens)
                }
                output_data.append(d)

            df = pd.DataFrame(output_data)
            df.to_csv(PROC_DIR+dataset+'.csv')


if __name__ == '__main__':
    main()