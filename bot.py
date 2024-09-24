import os
import botpy
from botpy import logging
from botpy.ext.cog_yaml import read
from botpy.message import GroupMessage
from modules import qna
from modules.mnist_handler import download_image
from modules.Mnist.main import test_mydata



config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))
_log = logging.get_logger()

class MyTeachingAssistantBot(botpy.Client):
    async def on_ready(self):
        _log.info(f"QQ群教学机器人 「{self.robot.name}」 已启动!")

    # 群聊消息处理
    async def on_group_at_message_create(self, message: GroupMessage):
        _log.info(f"收到来自群 {message.group_openid} 的消息: {message.content}")
        msg = message.content.strip()
        # 问答助手
        if msg.startswith("/问答"):
            question = msg.replace("/问答", "").strip()
            result = qna.handle_qna(question)
            await message._api.post_group_message(
                group_openid=message.group_openid,
                msg_type=0,
                msg_id=message.id,
                content=f"问答助手: {result}"
            )
        elif msg.startswith("/识别数字") and message.attachments:
            for attachment in message.attachments:
                #print(attachment)
                if attachment.url and attachment.content_type.startswith("image"):
                    # 调用图片下载和识别功能
                    image_url = attachment.url  # 提取图片URL
                    save_path = os.path.join("downloaded_images", "qq_image.jpg")
                    os.makedirs("downloaded_images", exist_ok=True)

                    # 下载图片
                    image_path = download_image(image_url, save_path)
                    if image_path:
                        # 识别图片中的数字
                        recognized_number = test_mydata(image_path)
                        if recognized_number is not None:
                            # 返回识别结果
                            await message._api.post_group_message(
                                group_openid=message.group_openid,
                                msg_type=0,
                                msg_id=message.id,
                                content=f"识别结果是数字: {recognized_number[0]}"
                            )
                        else:
                            await message._api.post_group_message(
                                group_openid=message.group_openid,
                                msg_type=0,
                                msg_id=message.id,
                                content=f"图片中未能识别出数字。"
                            )
                    else:
                        await message._api.post_group_message(
                            group_openid=message.group_openid,
                            msg_type=0,
                            msg_id=message.id,
                            content=f"图片下载失败"
                        )



if __name__ == "__main__":
    intents = botpy.Intents(public_messages=True)
    client = MyTeachingAssistantBot(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])
