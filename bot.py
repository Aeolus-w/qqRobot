import os
import botpy
from botpy import logging
from botpy.ext.cog_yaml import read
from botpy.message import GroupMessage
from to_api import send_message_to_api

config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))
_log = logging.get_logger()


class MyTeachingAssistantBot(botpy.Client):
    async def on_ready(self):
        _log.info(f"QQ群教学机器人 「{self.robot.name}」 已启动!")

    # 群聊消息处理
    async def on_group_at_message_create(self, message: GroupMessage):
        _log.info(f"收到来自群 {message.group_openid} 的消息: {message.content}")
        msg = message.content.strip()
            # 调用API服务器处理消息
        response = send_message_to_api(msg)
        content1 = response['choices'][0]['message']['content']
        await message._api.post_group_message(
                        group_openid=message.group_openid,
                        msg_type=0,
                        msg_id=message.id,
                        content=content1
                    )

if __name__ == "__main__":
    intents = botpy.Intents(public_messages=True)
    client = MyTeachingAssistantBot(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])
