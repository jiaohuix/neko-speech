"""
ViViAI / OpenAI 兼容图像接口最小测试：
  1. images.generate  →  生成动漫角色
  2. images.edit      →  在生成图基础上做角色编辑

输出图片到 ./imgs/
"""

import os
import re
import time
import base64
import requests
from openai import OpenAI

API_KEY = "sk-"
BASE_URL = "https://api.viviai.cc/v1"
MODEL = "gpt-image-2"

OUTPUT_DIR = "imgs"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def safe_filename(text: str, max_len: int = 20) -> str:
    text = text.strip()[:max_len]
    text = re.sub(r"[^\w一-鿿]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "image"


def save_image_auto(data, save_path: str):
    url = getattr(data, "url", None)
    b64_json = getattr(data, "b64_json", None)

    print(f"[DEBUG] url={url!r}  b64_len={len(b64_json) if b64_json else None}")

    if url:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return

    if b64_json:
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(b64_json))
        return

    raise ValueError(f"无法识别图片返回结构: {data}")


def generate_image(prompt: str, tag: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resp = client.images.generate(
        model=MODEL,
        prompt=prompt,
        size="1024x1024",
    )
    # print("[generate] full response:", resp.model_dump())

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_filename(tag)}_{int(time.time())}.png",
    )
    save_image_auto(resp.data[0], save_path)
    print(f"[generate] saved -> {save_path}")
    return save_path


def edit_image(src_path: str, prompt: str, tag: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(src_path, "rb") as f:
        resp = client.images.edit(
            model=MODEL,
            image=f,
            prompt=prompt,
            size="1024x1024",
        )
    print("[edit] full response:", resp.model_dump())

    save_path = os.path.join(
        OUTPUT_DIR,
        f"edit_{safe_filename(tag)}_{int(time.time())}.png",
    )
    save_image_auto(resp.data[0], save_path)
    print(f"[edit] saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    # gen_prompt = (
    #     "动漫风格全身立绘，一位粉色长发蓝眼的少女角色，"
    #     "白色连衣裙，温柔微笑，干净白色背景，二次元赛璐璐画风，高清"
    # )
    # gen_path = generate_image(gen_prompt, tag="anime_girl")

    # edit_prompt = (
    #     "保持同一角色与五官，将其服装替换为黑色女仆装，"
    #     "背景改为欧式咖啡馆室内，光线柔和"
    # )
    # edit_image(gen_path, edit_prompt, tag="maid_cafe")


    # gen_prompt = (
    #     "二次元赛璐璐全身立绘，可爱红发双马尾小妹妹，圆溜溜大眼睛，软萌幼态脸蛋，甜糯乖巧神态，浅浅害羞笑，磨砂玻璃朦胧柔雾质感，通透柔光，纯白色极简背景，细腻平涂上色，高清精致细节"
    # )
    # gen_path = generate_image(gen_prompt, tag="red_twintail_loli")

    # edit_prompt = (
    #     "保留原角色红发双马尾、五官脸型与软萌幼态妹妹特征，服装换成可爱蕾丝花边黑色女仆装，场景换成暖调欧式咖啡馆室内，柔和暖光，全程保留磨砂玻璃朦胧磨皮雾面氛围感，画面柔和通透"
    # )
    # edit_image(gen_path, edit_prompt, tag="maid_cafe_twintail")

    gen_prompt = (
        "二次元赛璐璐全身立绘，红发高双马尾高中少女，软萌可爱长相，清透灵动眼眸，清甜浅笑，磨砂玻璃朦胧柔雾滤镜质感，柔和漫射光影，干净纯白背景，线条干净，高清细腻画质"
    )
    gen_path = generate_image(gen_prompt, tag="red_twintail_highschoolgirl")

    edit_prompt = (
        "完全保留原红发双马尾少女五官、脸型与高中少女身形气质，服装更换为蕾丝花边黑色女仆装，场景切换为暖光欧式咖啡馆室内，柔和氛围感，全程保留磨砂玻璃朦胧雾面柔焦质感"
    )
    edit_image(gen_path, edit_prompt, tag="maid_cafe_twintail")
