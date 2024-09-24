import requests

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 检查请求是否成功
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"图片已保存到 {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"下载图片失败: {e}")
        return None

