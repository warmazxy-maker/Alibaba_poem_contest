# 复制本文件为 env.bailian.local.ps1，然后把下面的占位符改成你的百炼 API Key。
# env.bailian.local.ps1 已被 .gitignore 忽略，不要把真实 key 写进源码或提交给别人。

$env:DASHSCOPE_API_KEY = "sk-0724191515fb421ab0682a951ed3ac43"

# 中国大陆（北京）地域 OpenAI 兼容接口：
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 如果你使用国际（新加坡）地域，改用这一行：
# $env:DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# 常用模型示例：qwen-plus / qwen-turbo / qwen-max
$env:DASHSCOPE_MODEL = "qwen-plus"

$env:LLM_TEMPERATURE = "0"
$env:LLM_MAX_TOKENS = "2048"
$env:DEBATE_ROUNDS = "2"
