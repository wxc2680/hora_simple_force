import socket
import json
import time
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# 添加路径以便导入相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

HOST = '0.0.0.0'     # 容器服务器监听所有网卡
PORT = 22         # 可自定义端口号

class SimplePolicy(nn.Module):
    """简单的策略网络（需要与训练时的网络结构匹配）"""
    def __init__(self, input_dim=144, output_dim=16):
        super(SimplePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # 输出在 [-1, 1] 范围
        )
    
    def forward(self, x):
        return self.net(x)

class ActionGenerator:
    def __init__(self, mode="random", model_path=None):
        self.mode = mode
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 关节限制（弧度）- 根据你的硬件调整
        self.joint_limits = {
            'min': [-1.5, -1.5, -1.5, -1.5],  # 4个关节的最小值
            'max': [1.5, 1.5, 1.5, 1.5]       # 4个关节的最大值
        }
        
        if mode == "model" and model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的 .pth 模型"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
            print(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 根据模型文件结构加载
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'actor' in checkpoint:
                    self.model = checkpoint['actor']
                elif 'policy' in checkpoint:
                    self.model = checkpoint['policy']
                else:
                    # 如果不知道结构，尝试直接使用
                    self.model = checkpoint
            else:
                self.model = checkpoint
                
            # 如果模型是状态字典，需要创建网络结构
            if isinstance(self.model, dict):
                # 尝试推断输入输出维度
                input_dim = 144  # 默认值，根据你的观测空间调整
                output_dim = 16  # 默认值，根据你的动作空间调整
                
                # 从状态字典推断结构
                for key in self.model.keys():
                    if 'weight' in key and len(self.model[key].shape) == 2:
                        if self.model[key].shape[1] == input_dim:
                            output_dim = self.model[key].shape[0]
                
                self.model = SimplePolicy(input_dim, output_dim)
                self.model.load_state_dict(self.model)
            
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
            print("模型加载成功!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.mode = "random"  # 失败时回退到随机模式
    
    def generate_random_action(self):
        """生成带限制的随机动作"""
        # 在限制范围内生成随机位置
        action = []
        for i in range(4):  # 4个关节
            min_val = self.joint_limits['min'][i]
            max_val = self.joint_limits['max'][i]
            action.append(np.random.uniform(min_val, max_val))
        return action
    
    def generate_model_action(self, observation=None):
        """使用模型生成动作"""
        if self.model is None:
            print("模型未加载，使用随机动作")
            return self.generate_random_action()
        
        try:
            # 如果没有提供观测，创建假的观测数据
            if observation is None:
                observation = torch.zeros(1, 144, device=self.device)
            else:
                if isinstance(observation, np.ndarray):
                    observation = torch.from_numpy(observation).float().to(self.device)
                if len(observation.shape) == 1:
                    observation = observation.unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(self.model, '__call__'):
                    action = self.model(observation)
                else:
                    # 如果模型是状态字典或其他格式
                    action = torch.zeros(1, 16, device=self.device)
                
            # 提取前4个关节的动作（单指）
                action = action.cpu().numpy()[0]
                
                # 限制在合理范围内
                action = np.clip(action[:4], -1.0, 1.0)  # 只取前4个关节
                
                # 将 [-1, 1] 映射到关节限制范围
                scaled_action = []
                for i in range(4):
                    min_val = self.joint_limits['min'][i]
                    max_val = self.joint_limits['max'][i]
                    # 将 [-1, 1] 映射到 [min_val, max_val]
                    scaled_val = min_val + (action[i] + 1) * (max_val - min_val) / 2
                    scaled_action.append(float(scaled_val))
                
                return scaled_action
                
        except Exception as e:
            print(f"模型推理失败: {e}")
            return self.generate_random_action()
    
    def get_action(self, observation=None):
        """根据模式获取动作"""
        if self.mode == "model":
            return self.generate_model_action(observation)
        else:  # random mode
            return self.generate_random_action()

def send_action_to_client(conn, action):
    """发送动作到客户端"""
    data = json.dumps({"angles": action}).encode('utf-8')
    conn.sendall(data + b'\n')  # 一行表示一帧
    print(f"发送动作: {action}")

def main():
    # 配置参数
    MODE = "random"  # 可选: "random" 或 "model"
    MODEL_PATH = "outputs/debug_handv3_55_zonly/stage1_nn/best.pth"  # 你的 .pth 文件路径
    
    # 初始化动作生成器
    action_generator = ActionGenerator(mode=MODE, model_path=MODEL_PATH)
    
    print(f"服务器启动中，监听端口 {PORT} ...")
    print(f"模式: {MODE}")
    if MODE == "model":
        print(f"模型路径: {MODEL_PATH}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"等待客户端连接...")
        
        conn, addr = server_socket.accept()
        print(f"客户端已连接: {addr}")
        
        # 发送连接确认
        conn.sendall(b"CONNECTED\n")

        step = 0
        try:
            while True:
                # 生成动作
                if MODE == "model":
                    # 对于模型模式，可以在这里添加观测获取逻辑
                    # 例如: observation = get_current_observation()
                    observation = None  # 暂时使用None
                    action = action_generator.get_action(observation)
                else:
                    action = action_generator.get_action()
                
                # 发送动作到客户端
                send_action_to_client(conn, action)
                
                step += 1
                if step % 50 == 0:
                    print(f"已发送 {step} 帧数据")
                
                time.sleep(0.1)  # 控制发送频率（10Hz）

        except KeyboardInterrupt:
            print("手动中断")
        except BrokenPipeError:
            print("客户端断开连接")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            conn.close()

    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        server_socket.close()
        print("服务器已关闭")

if __name__ == "__main__":
    main()