# onnx2torchscript

## Notes

### 実装方針
torch.jit.scriptを使ってpythonで書かれたtorchscriptを使って実装されたopを元にtorchに戻す

```python
# opset versionに対応するグラフ
op_table: Dict[int, Dict[str, torch._C.Graph]]

# torch.jit.scriptでtorch._C.Graphに変換する
@torch.jit.script
def op_Add_9(*inputs, **attrs) -> Any: # torch.Tensorのリスト？
	outputs = …
	return outptuts

# op登録
op_table[9][“Add”] = op_Add_9.graph

# 変換のエントリポイント
# ノードごとにgraphを結合
# Valueはonnxのidで管理
def onnx2torch(model: onnx.ModelProto):
	ret: torch._C.Graph = torch._C.Graph()
	values: Dict[str, torch._C.Value] = {}
	for n in model.graph.node:
		op_table[model.opset_imports.opset_version][n._op_type](*n.inputs(), **attrs)
	return ret
```

### テスト方針
- torch.onnxで作る
- onnxにあるテストケースをうまくopset_versionごとにバラしてやる
- torchscriptに変換されたので再実行して結果を見る
- onnxruntimeとの突き合わせ
- 入出力を変える

## 履歴
- 2022/09/21
    - ざっくり書く
    - TODO: リポジトリ作る
        - https://github.com/take-cheeze/onnx2torchscript
    - TODO: 開発環境づくり
        - TODO: gitpod試す
            - 権限がめんどうだったのでやっぱりなし
        - とりあえずm1 macでやってみる。mps使えるし
- 2022/10/02
    - リハビリがてら進める
