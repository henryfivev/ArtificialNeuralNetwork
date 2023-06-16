import jieba

input_file = "./data/test_ch.txt"
output_file = "./data/test_ch_seg.txt"

with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

seg_lines = []
for line in lines:
    seg_words = jieba.cut(line.strip())
    seg_line = " ".join(seg_words)
    seg_lines.append(seg_line+'\n')

with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(seg_lines)

print("中文分词已完成，结果已保存到train_ch_seg.txt文件中。")
