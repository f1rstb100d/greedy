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