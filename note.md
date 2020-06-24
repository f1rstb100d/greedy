# statistical 机器翻译
```
语言模型 LM:
1. 一个英文句子e，计算概率p(e)
2. 符合语法，那么p(e)高；不符合语法，p(e)低

翻译模型:
1. 一组<c,e>，c是翻译出来的中文e是给定的英文，计算p(c|e)
2. 如果语义相似度高，那么p(c|e)高；反之p(c|e)低

Decoding Algorithm:
给定语言模型，翻译模型和c，找出最优解使得p(e)p(c|e)最大
```

# Language Model 语言模型
```
一个好的模型:
p(He is studying AI) > p(He studying AI is)

怎么计算p(句子):
unigram p(He is studying AI) = p(He)p(is)p(studying)p(AI)
bi-gram p(He is studying AI) = p(He)p(is|He)p(studying|is)p(AI|studying)
Tri-gram p(He is studying AI) = p(He)p(is|He)p(studying|He is)p(AI|is studying)
```

# Markov Assumption
```
公式：P(x1x2x3x4)=p(x1)p(x2|x1)p(x3|x1x2)p(x4|x1x2x3)
根据联合概率定义p(x1x2)=p(x1)p(x2|x1)，这四项可以反向拼回原来的P

后面的三种gram都是基于Markov假设所作的简化：
P(x1x2x3x4)=p(x1)p(x2)p(x3)p(x4)
P(x1x2x3x4)=p(x1)p(x2|x1)p(x3|x2)p(x4|x3)
p(x1x2x3x4)=p(x1)p(x2|x1)p(x3|x1x2)p(x4|x2x3)
```

# 时间复杂度 空间复杂度
[时间复杂度与空间复杂度--reference](https://zhuanlan.zhihu.com/p/53286463)
```
for (i=0; i<N; i++){
  a = a + rand(); # N*1个操作 = O(n)
  b = b + rand(); # N*2个操作 = O(n)+O(n)=O(n)线性相加
}

for (i=0; i<N/2; i++){
  a = a + rand(); # N/2*1个操作 = 1/2*O(n) = O(n)
}
第一个的空间复杂度：2个内存空间=O(1)
第二个空间复杂度：O(1)

for (i=0; i<N; i++){
  for (j=N; j>i; j--){
    a = a + 1 #一共等差数列(1,2,..,N)求和次操作,为1/2*O(n^2)+1/2*O(n)=O(n^2)+O(n)=O(n^2)，空间复杂度O(3)=O(1)
  }
}

i = N;
while(i>0){
  a = a + i;
  i = i / 2; # 2*log(N)个操作，时间复杂度O(log N)
}

算法效率：O(1)>O(log n)>O(n)>O(nlog n)>O(n^2) n要足够大才能体现出优势
```

# merge-sort 归并排序
![](https://github.com/f1rstb100d/greedy/blob/master/jpg/Merge-sort%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F.gif)
```
思路：把数组对半分开，每一半继续递归调用归并排序算法，直到这两半都成递增顺序，然后两个指针分别从左往右扫描(合并两个有序数组)，并将小的添加到结果数组中
时间复杂度: T(n) = T(n/2) + T(n/2) + n = 2*T(n/2) + n
递归时间复杂度使用主元素分析法Master Theorem
```

# Master Theorem
![](https://github.com/f1rstb100d/greedy/blob/master/jpg/master%20theorem.jpg)
```
计算归并排序时间复杂度 T(n)=2T(n/2)+n
a=2, b=2, f(n)=n
n^logb(a)=n=f(n),所以对应规则2
根据公式带入ab，返回O(nlogn)
```

# 斐波那契数列
```
def fib(n):
  if n<3:
    return 1
  return fib(n-2)+fib(n-1)

时间复杂度：T(n)=T(n-2)+T(n-1)
不符合主定理公式格式，使用二叉树的结构计算：
        f(8)
      /      \
   f(6)      f(7)
  /   \     /   \
f(4) f(5) f(5) f(6)
第一行2^0个操作，第二行2^1个操作，第三行2^3个操作
一共需要计算n-2行，最后两行是base case，所以一共是2^0+2^1+2^2+...+2^(n-2)=2^(n-1)-1个操作，当n足够大忽略2^n和2^(n-1)的区别
所以时间复杂度是O(2^n)，空间复杂度就是二叉树总的节点个数

动态规划法 Dynamic Programming 能复用的尽量复用而不重新计算
另一种是维护一个list，每次计算前两个的和得到新的值append上去，时间复杂度O(N)
```

# P NP NP Hard NP Complete
![](https://github.com/f1rstb100d/greedy/blob/master/jpg/NP%20hard.png)
```
时间复杂度O(p^n):指数级 p是常数，不可以解决的问题(NP Hard)
Solution:1.n小的话可以用 2.近似算法，近似成O(n^p)，指出有多少误差

时间复杂度O(n^p):多项式级 p是常数，可以解决的问题(P问题)
```

# Pipeline
```
步骤：
1. 原始文本(raw data)
2. 分词(Segmentation)
3. 清洗(Cleaning) -> 无用的标签、特殊符号、停用词、大写转小写
4. 标准化(Normalization) -> Stemming、Lemmazation
5. 特征提取(Feature Extraction) -> tf-idf、word2vec
6. 建模(Modeling) -> 相似度算法、分类算法
```

# Word Segmentation(分词)
```
方法1：Max Matching(最大匹配)
前向最大匹配(forward-max matching) max_len=5
e.g. 我们经常有意见分歧
词典：[我们，经常，有，有意见，意见，分歧]
[我们经常有]意见分歧 -> ×
[我们经常] -> ×
[我们经] -> ×
[我们] -> √
然后往后移就是 [经常有意见]分歧 -> ×  以此类推
后向最大匹配(backward-max matching)
我们经常[有意见分歧] -> ×
[意见分歧] -> ×
[见分歧] -> ×
[分歧] -> √
然后往前移就是 我们[经常有意见]分歧 -> ×  以此类推

方法2：Incorporate Semantic(考虑语义)
对一个句子，列举出所有可能的分词情况，使用前面提到的language model的uni-gram形式计算p(x1x2x3x4)=p(x1)p(x2)p(x3)p(x4)，选择其中得分最高的作为分词结果，得分越高越符合语义。
其中计算p(x1)=(x1在文档中出现次数)/(文档的全部单词数)
所以p就很小，再乘到一起就更小了
改进：两边都取对数log p(x1)p(x2)p(x3)p(x4)=log p(x1)+log p(x2)+log p(x3)+log p(x4)
```
![](https://github.com/f1rstb100d/greedy/blob/master/jpg/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95.png)
```
因为上一个需要列举出句子所有的分词可能，复杂度太高，使用维特比算法简化。
同样的句子，假设已经计算出每个单词的p，防止p乘积太小，取p的以e为底的对数的负数作为新的概率，之前是求乘积最大，现在是求和最小。
画出图，将每条边的得分表上，所以现在就是找一条边从1到8使其边权之和最小。类似前面斐波那契问题，维护一个数组，用DP动态规划的思路从f(1)、f(2)往后推，直到得出最小的f(8)
```

# Spell Correction(拼写纠正)
```
通过计算用户输入的词和每一个候选词的编辑距离(需要修改几个字母就能变成候选词为几个编辑距离)，选最小编辑距离的词为纠正之后的词。
词库太大的话复杂度太高，优化：给定用户输入之后，仅生成编辑距离为1、2的所有字符串(replace add delete)(编辑距离为2的字符串是在编辑距离为1的字符串基础上进行那三个操作)，然后过滤返回结果
怎么过滤：输入s，最佳字符串为c，需要argmaxp(c|s),找最好的c
argmax(c|s)=argmax p(s|c)*p(c)/p(s)
因为p(s)在对每个c的时候都一样，忽略
p(s|c)为对一个正确的字符串，有百分之多少的人输入成了s的形式
p(c)为总的文本里，根据c出现的次数计算的概率
```

# Filtering Words
```
去停用词、去低频词
stemming、lemmatization：使用定义的规则来normalize单词 went->go
```

# Word Representation
```
one-hot representation: 词典出现的单词为1，其他位为0

所以sentence representation(Boolean)就是把所以存在的单词为位置1，其他位置0
另外的sentence representation(count)每个单词的值为其在句子中的出现次数，而不仅仅是1
```

# Sentence Similarity
```
1. 欧式距离 d=|s1-s2| 越小越好
2. 余弦相似度 d=s1*s2/(|s1|*|s2|) *为内积 越大越好
```

# tf-idf Representation
```
tfidf(w)=tf(d,w)*idf(w)
tf(d,w)：文档d中w的词频
idf(w)：log(N/N(w)) N:语料库文档总数  N(w):词语w出现在多少个文档
这是计算的每个单词的tfidf值，sentence representation的时候在这个单词位置的值是tfidf值，其他位置为0
```

# Measure Similarity Between Words
```
one-hot的表示方式导致计算欧式距离全部相同、余弦相似度全都是0，所以不能表示单词之前的相似度差别
one-hot的另一个问题：稀疏矩阵，0太多
所以向量的长度不应该等于字典长度，需要缩短尽量每一位都有值，改为distributed representation
```

# Word Embedding to Sentence Embedding
```
方法一：平均法，求句子每个单词向量各个位上的平均值
```

# 倒排表
```
对词典里的每个单词，统计出包含这个单词的文档编号
当我输入一个单词时，直接返回所有包含这个单词的文档，使用PageRank的方式对返回结果排个序
当我输入多个单词时，直接返回每个单词倒排表的之间的交集文档，然后排序返回
```
