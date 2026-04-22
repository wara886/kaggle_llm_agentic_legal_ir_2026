# Laws Truncation Audit

- split: `val`
- laws gold citations in corpus: `149`
- gold docs fitting first 500 chars: `137` / `149`
- gold docs fitting first 900 chars: `147` / `149`

## Signal Position

| bucket | count | ratio |
|---|---:|---:|
| within_500 | 81 | 0.543624 |
| within_900 | 0 | 0.000000 |
| beyond_900 | 0 | 0.000000 |
| no_detectable_query_signal | 68 | 0.456376 |

## Conclusion

- 截断不是当前主瓶颈之一；优先进入 laws-only hard negative mining + MiniLM fine-tune。
