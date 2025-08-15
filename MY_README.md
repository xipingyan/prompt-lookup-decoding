# README

```
python -m venv venv
source venv/bin/activate
pip install torch transformers==4.55.0 accelerate

python transformers_example.py
```

``Note:``

Add "prompt_lookup_num_tokens=10" to generate to enable lookup.

# Algorithm

``1:`` get_candidates
<pre>
从当前input prompts的最后2个token，把这两个token作为ngram， 从input prompt开始查找有没有相同token。 <br>
默认ngram size=2，

如果没有找到：
    减少ngram size，再去找。 最小的ngram size=-1；
    如果还是没有找到，candidate就是0，

找到多个
    在第一个matched ngram后面，直接选取10个token，作为candidate 
    Return 
    这里的10，就是参数：prompt_lookup_num_tokens 
</pre>

``2:`` 推理
<pre>
loop 1： (prefill)
    把input prompt和candidates, 合并到一起，输入到网络，输出prompt_lookup_num_tokens+1 个token。

loop n:
    网络输入candidantes（prompt_lookup_num_tokens个token），输出prompt_lookup_num_tokens+1个token

match：
    对于输出的prompt_lookup_num_tokens+1，从开头和candidante进行比较（相同位置才进行比较），
    如果token id，有2个相同，认为匹配上2个token。
    直接把匹配上的token+1，作为输出，匹配上2个，则一次输出3个token。


和没有使用lookup相比，虽然second token，每次infer输入到seq length从1变成了prompt_lookup_num_tokens(10)，
但是second token一般是memory-bound，所以10个和1个相比，时间基本一致。

</pre>