# SimpleHand 训练

## 第一阶段

1. 在 [scripts/train_s1.sh](../scripts/train_s1.sh) 里把 `task.env.object.type` 改成要训练的物体。训练 50 mm 方块时用 `task.env.object.type=cube`，训练 66 mm 方块时用 `task.env.object.type=cube66`。
2. 激活训练环境后，运行下面的命令。

   ```bash
   cd /opt/hora_simplehand_force
   ./scripts/train_s1.sh 0 42 101001cube50
   ```

   `0` 是 GPU 编号，`42` 是随机种子，`101001cube50` 是这次训练的输出名。训练物体由上面的 `type` 决定，结果保存在 `outputs/Hand2Hora/101001cube50/`。

## 第二阶段

1. 确认第一阶段已生成 `outputs/Hand2Hora/101001cube50/stage1_nn/best.pth`。第二阶段的 `task.env.object.type` 要和第一阶段一致，修改位置见 [scripts/train_s2.sh](../scripts/train_s2.sh)。
2. 在仓库目录下运行下面的命令。沿用第一阶段的输出名，脚本会自动加载对应的 `best.pth`。

   ```bash
   ./scripts/train_s2.sh 0 42 101001cube50
   ```

   第二阶段的权重保存在 `outputs/Hand2Hora/101001cube50/stage2_nn/model_last.ckpt`。
