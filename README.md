# tfm_baseline

- DataPreprocessing: 原始数据预处理, 保存为mmap格式. 如果重新生成把'1699886388616'这个替换成你自己的文件夹名称
- exp.py: 直接python exp.py configs.yaml / configsqlib.yaml可复现Vanilla Transformer的实验结果
- 云桌面用configs.yaml, 外网qlib数据用configsqlib.yaml
- Backtest: 从Logfiles里面读取模型结果, 做回测

效果如下, 可以看到1min数据如果只包含原始量价信息, 模型表现较差
qlib 1min特征包含: "$change", "$close", "$factor", "$high", "$low", "$open", "$paused", "$paused_num", "$volume"

![Vanilla-Transformer-epoch1-1.png](Logfiles%2FVanilla-Transformer-epoch1-1.png)
![Vanilla-Transformer-epoch1-2.png](Logfiles%2FVanilla-Transformer-epoch1-2.png)
![Vanilla-Transformer-epoch1-3.png](Logfiles%2FVanilla-Transformer-epoch1-3.png)
![Vanilla-Transformer-epoch1-4.png](Logfiles%2FVanilla-Transformer-epoch1-4.png)
![Vanilla-Transformer-epoch1-5.png](Logfiles%2FVanilla-Transformer-epoch1-5.png)
![Vanilla-Transformer-epoch2-1.png](Logfiles%2FVanilla-Transformer-epoch2-1.png)
![Vanilla-Transformer-epoch2-2.png](Logfiles%2FVanilla-Transformer-epoch2-2.png)
![Vanilla-Transformer-epoch2-3.png](Logfiles%2FVanilla-Transformer-epoch2-3.png)
![Vanilla-Transformer-epoch2-4.png](Logfiles%2FVanilla-Transformer-epoch2-4.png)
![Vanilla-Transformer-epoch2-5.png](Logfiles%2FVanilla-Transformer-epoch2-5.png)

![https://www.notion.so/baseline-1332027a6f308038829ac1e24c741b69]
