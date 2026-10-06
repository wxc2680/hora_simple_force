import socket
import time
import json

# 服务器配置
HOST = '0.0.0.0'  # 监听所有网络接口
PORT = 50000      # 选择一个不常用的端口

def main():
    print("=== 简单服务器测试 ===")
    print(f"尝试在端口 {PORT} 启动服务器...")
    
    # 创建socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # 绑定端口
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"✓ 服务器启动成功!")
        print(f"✓ 监听端口: {PORT}")
        print(f"✓ 客户端应该连接这个端口")
        
        # 等待客户端连接
        print("等待客户端连接...")
        conn, addr = server_socket.accept()
        print(f"✓ 客户端连接成功: {addr}")
        
        # 发送欢迎消息
        welcome_msg = {"message": "服务器连接成功!", "status": "ready"}
        conn.sendall(json.dumps(welcome_msg).encode() + b'\n')
        
        counter = 0
        try:
            while True:
                # 发送简单的测试数据
                counter += 1
                test_data = {
                    "counter": counter,
                    "angles": [0.1 * counter, 0.2 * counter, 0.3 * counter, 0.4 * counter],
                    "timestamp": time.time()
                }
                
                # 发送数据
                data_str = json.dumps(test_data) + '\n'
                conn.sendall(data_str.encode())
                print(f"发送第 {counter} 条数据: {test_data['angles']}")
                
                time.sleep(1)  # 每秒发送一次
                
        except KeyboardInterrupt:
            print("\n用户中断")
        except BrokenPipeError:
            print("客户端断开连接")
        except Exception as e:
            print(f"发送数据错误: {e}")
        finally:
            conn.close()
            
    except OSError as e:
        print(f"✗ 端口 {PORT} 被占用，请换一个端口")
        print(f"错误信息: {e}")
    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        server_socket.close()
        print("服务器已关闭")

if __name__ == "__main__":
    main()