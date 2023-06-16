import nltk

# 读取文件并逐行分词
def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    tokenized_lines = []
    for line in lines:
        line = line.strip()  # 去除开头和结尾的空白字符
        tokens = nltk.word_tokenize(line)
        tokenized_line = ' '.join(tokens)  # 使用空格连接分词后的tokens
        tokenized_lines.append(tokenized_line)
    
    return tokenized_lines

# 使用示例
input_file_path = './data/test_en.txt'
output_file_path = './data/test_en_seg.txt'

# 分词处理
tokenized_lines = tokenize_file(input_file_path)

# 保存分词结果到文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in tokenized_lines:
        file.write(line + '\n')

print(f"分词后的结果已保存到文件：{output_file_path}")
