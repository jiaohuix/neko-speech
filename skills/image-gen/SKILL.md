# Skill: ViViAI 图像生成与编辑接口

OpenAI 兼容的图像接口（`api.viviai.cc/v1`，模型 `gpt-image-2`）。支持 **文生图** 与 **图像编辑** 两种用法，调用方式与官方 `openai` Python SDK 完全一致。

## 接口能力

| 用法     | SDK 调用                  | 主要入参                                | 返回                  |
| ------ | ----------------------- | ----------------------------------- | ------------------- |
| 文生图    | `client.images.generate` | `model`, `prompt`, `size`           | `data[0].b64_json`  |
| 图像编辑   | `client.images.edit`     | `model`, `image=file`, `prompt`, `size` | `data[0].b64_json`  |

- `BASE_URL = "https://api.viviai.cc/v1"`、`MODEL = "gpt-image-2"`
- `size` 实测可用 `1024x1024`
- 返回数据通常是 **base64** (`b64_json`)，不是 URL；保存时要二者都兼容
- `images.edit` 不传 `mask`，效果是「以参考图为视觉锚点，按 prompt 重绘整张」——适合保角色换装/换场景，但不是局部修补
- 用量结算字段：`usage.input_tokens` / `output_tokens`，编辑请求会多出 `input_tokens_details.image_tokens`

## 最小骨架

```python
from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url="https://api.viviai.cc/v1")

# 文生图
r = client.images.generate(model="gpt-image-2", prompt="...", size="1024x1024")

# 图像编辑（角色一致性 / 换装 / 换场景）
with open(src_png, "rb") as f:
    r = client.images.edit(model="gpt-image-2", image=f, prompt="...", size="1024x1024")

# 保存：先看 url，再看 b64_json
data = r.data[0]
if data.url:
    open(out, "wb").write(requests.get(data.url, timeout=120).content)
else:
    open(out, "wb").write(base64.b64decode(data.b64_json))
```

完整可运行脚本：`run.py`（文生图 → 拿到 PNG → 编辑为女仆装咖啡馆场景）。

## 注意事项

- App 背景图替换前先验证渲染层级：如果页面上有 `SurfaceView` / `GLSurfaceView` / canvas / 视频层，先确认它是否是不透明 EGL surface 或覆盖在背景之上。不要把“背景没变”直接归因于生成图或资源替换失败；先用现有图片做可见性验证，再决定是否重新生成。
- 返回 base64 体积可能很大（百万字符级），不要 `print(response)` 整个对象，会刷屏；只打印 `len(b64_json)`
- 长耗时生成必须给 SDK client 或外层命令设置明确 timeout；如果单次请求超时，先降级为单张重试，不要让批量请求无期限阻塞后续工程改动。
- 不要把 API key 写进 here-doc / `python - <<` 命令体；Linux `ps` 可能暴露整段命令。优先从已有脚本常量、环境变量、受控临时 stdin 或本地未提交配置读取。
- `images.edit` 的 `image` 参数传**文件句柄**（`open(path, "rb")`），不要 `base64` 字符串
- 编辑模型对角色保真度有限，prompt 中显式写「保持同一角色与五官」会更稳
- 文件名建议从 prompt 截断 + 时间戳，避免中文/特殊字符导致路径问题（脚本里的 `safe_filename` 已处理）
