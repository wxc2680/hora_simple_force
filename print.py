# interactive_inspect.py
import numpy as np
import matplotlib.pyplot as plt

def interactive_inspect(file_path):
    data = np.load(file_path)
    
    while True:
        print("\n" + "="*60)
        print(f"当前文件: {file_path}")
        print(f"数据形状: {data.shape}")
        print("\n选择操作:")
        print("1. 查看样本详情")
        print("2. 查看统计信息")
        print("3. 可视化数据分布")
        print("4. 查看特定样本")
        print("5. 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            print(f"\n数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"前3个样本:")
            for i in range(min(3, len(data))):
                print(f"样本 {i}: {data[i]}")
                
        elif choice == '2':
            print("\n统计信息:")
            print(f"总样本数: {len(data)}")
            print(f"最小值: {data.min():.6f}")
            print(f"最大值: {data.max():.6f}")
            print(f"平均值: {data.mean():.6f}")
            print(f"标准差: {data.std():.6f}")
            
            if data.ndim == 2:
                print("\n各列统计:")
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    print(f"列 {col:2d}: 最小值={col_data.min():8.4f}, 最大值={col_data.max():8.4f}, 均值={col_data.mean():8.4f}")
                    
        elif choice == '3':
            if data.ndim == 2:
                # 绘制前几列的分布
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(min(6, data.shape[1])):
                    axes[i].hist(data[:, i], bins=50, alpha=0.7)
                    axes[i].set_title(f'Column {i} Distribution')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.show()
            else:
                print("数据不是2维，无法绘制分布图")
                
        elif choice == '4':
            try:
                idx = int(input(f"输入样本索引 (0-{len(data)-1}): "))
                if 0 <= idx < len(data):
                    print(f"\n样本 {idx}:")
                    print(data[idx])
                    
                    # 如果是抓取数据，解释结构
                    if data.shape[1] == 32:  # 典型的抓取数据维度
                        print("\n数据解释:")
                        print(f"前16个值 (0-15): 手部关节位置")
                        print(f"后16个值 (16-31): 物体姿态 (位置 + 旋转)")
                        print(f"  位置: [{data[idx][16]:.3f}, {data[idx][17]:.3f}, {data[idx][18]:.3f}]")
                        print(f"  旋转四元数: [{data[idx][19]:.3f}, {data[idx][20]:.3f}, {data[idx][21]:.3f}, {data[idx][22]:.3f}]")
                else:
                    print("索引超出范围")
            except ValueError:
                print("请输入有效的数字")
                
        elif choice == '5':
            break
        else:
            print("无效选择")

# 使用
file_path = "cache/internal_hand2_grasp_50k_s07.npy"  # 替换为你的文件路径
interactive_inspect(file_path)