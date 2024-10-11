import os
import botpy
from botpy import logging
from botpy.ext.cog_yaml import read
from botpy.message import GroupMessage

from to_api import send_message_to_api
from to_api import create_ssh_tunnel
from weather_handler import get_weather

config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))
_log = logging.get_logger()

class MyTeachingAssistantBot(botpy.Client):
    async def on_ready(self):
        _log.info(f"QQ群教学机器人 「{self.robot.name}」 已启动!")
    
    create_ssh_tunnel("connect.cqa1.seetacloud.com",28696,"root","Hemcuc1kuD7/","127.0.0.1",6006,6006)
    # 群聊消息处理
    async def on_group_at_message_create(self, message: GroupMessage):
        _log.info(f"收到来自群 {message.group_openid} 的消息: {message.content}")
        msg = message.content.strip()#这里msg是接收到的消息

        if msg.startswith("/美食"):
            # 调用API服务器处理消息
            food_name = msg.replace("/美食","").strip()
            food_prompt = food_name + "是我最近想吃的东西，你可以帮我在成都找一下推荐的店吗？"
            food_response = send_message_to_api(food_prompt)
            food_content = food_response['choices'][0]['message']['content']
            await message._api.post_group_message(
                            group_openid=message.group_openid,
                            msg_type = 0,
                            msg_id=message.id,
                            content=food_content
                        )
        
        elif msg.startswith("/天气"):
            city_name = msg.replace("/天气", "").strip()
            result = get_weather(city_name)  # 调用模块中的函数
            weather_prompt = result + ".这里是一些天气信息，请你为我的衣物安排提一些建议，与这几天的具体天气信息一起发给我,包括温度等等。"
            weather_response = send_message_to_api(weather_prompt)
            weather_content = weather_response['choices'][0]['message']['content']
            await message._api.post_group_message(
                group_openid=message.group_openid,
                msg_type=0,
                msg_id=message.id,
                content=weather_content
            )
        
        elif msg.startswith("/景点"):
            attraction_name = msg.replace("/景点","").strip()
            attraction_prompt = attraction_name + "是我最近考虑去的一个景点，你可以为我提供一些攻略建议吗？"
            attraction_response = send_message_to_api(attraction_prompt)
            attraction_content = attraction_response['choices'][0]['message']['content']
            await message._api.post_group_message(
                            group_openid=message.group_openid,
                            msg_type = 0,
                            msg_id=message.id,
                            content=attraction_content
                        )
        else:
            await message._api.post_group_message(
                            group_openid=message.group_openid,
                            msg_type = 0,
                            msg_id=message.id,
                            content="不好意思，这个问题我还暂时不能回答。"
                        )


if __name__ == "__main__":
    intents = botpy.Intents(public_messages=True)
    client = MyTeachingAssistantBot(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])
