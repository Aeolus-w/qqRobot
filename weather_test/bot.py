import os
import botpy
from botpy import logging
from botpy.ext.cog_yaml import read
from botpy.message import GroupMessage

# 导入独立模块
from weather_handler import get_weather
from greet_handler import greet

config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))
_log = logging.get_logger()


class MyClient(botpy.Client):
    async def on_ready(self):
        _log.info(f"robot 「{self.robot.name}」 on_ready!")

    async def on_group_at_message_create(self, message: GroupMessage):
        openid = message.author.member_openid
        _log.info(f"robot {openid} on_ready!")
        msg = message.content.strip()

        # 查询天气功能
        if msg.startswith("/天气"):
            city_name = msg.replace("/天气", "").strip()
            result = get_weather(city_name)  # 调用模块中的函数
            await message._api.post_group_message(
                group_openid=message.group_openid,
                msg_type=0,
                msg_id=message.id,
                content=f"{result}"
            )

        # 打招呼功能
        elif msg == "/打招呼 你好":
            result = greet()  # 调用模块中的函数
            await message._api.post_group_message(
                group_openid=message.group_openid,
                msg_type=0,
                msg_id=message.id,
                content=f"{result}"
            )


if __name__ == "__main__":
    intents = botpy.Intents(public_messages=True)
    client = MyClient(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])
