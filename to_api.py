import requests
import json
import paramiko
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder

def create_ssh_tunnel(ssh_host, ssh_port, ssh_username, ssh_password, remote_bind_address, remote_bind_port, local_bind_port):
    # 创建 SSH 隧道连接
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_bind_address=(remote_bind_address, remote_bind_port),
        local_bind_address=('127.0.0.1', local_bind_port)
    )
    tunnel.start()  # 启动隧道
    print(f"SSH隧道已建立，正在将本地 {local_bind_port} 映射到远程 {remote_bind_address}:{remote_bind_port}")
    return tunnel  # 返回隧道对象

def send_message_to_api(message):
    url = "http://127.0.0.1:6006/v1/chat/completions" 
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0.8,
        "top_p": 0.8,
        "max_tokens": 1500,
        "echo": False,
        "stream": False,
        "repetition_penalty": 1.1,
        "tools": None,
        "model": "chatglm3-6b" 
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() 
        return response.json()  
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

if __name__ == "__main__":
    user_message = "你好，今天的天气怎么样？"
    response = send_message_to_api(user_message)
    if response:
        print("响应:", response)
