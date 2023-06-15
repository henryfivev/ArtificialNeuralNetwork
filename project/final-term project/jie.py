import json
import jieba

# 读取源语言文本文件
with open('./data/train_ch.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# 统计词频
word_freq = {}
for sentence in sentences:
    words = jieba.cut(sentence.strip())
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

# 将统计词典保存为JSON文件
with open('vocab.json', 'w', encoding='utf-8') as file:
    json.dump(word_freq, file, ensure_ascii=False)
