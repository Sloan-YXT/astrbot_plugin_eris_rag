import json
import traceback
import aiohttp

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.message.components import Poke


@register(
    "astrbot_plugin_eris_rag",
    "utt.yao",
    "Eris RAG 人格增强插件",
    "1.0.0",
    "",
)
class ErisRAGPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.enabled: bool = config.get("enabled", True)
        self.rag_url: str = config.get("rag_server_url", "http://192.168.3.22:8787").rstrip("/")
        self.timeout_s: float = max(config.get("timeout_ms", 10000), 1000) / 1000.0
        self.context_count: int = min(max(config.get("context_count", 6), 0), 20)
        self.debug_log: bool = config.get("debug_log", False)
        self.private_poke_enabled: bool = config.get("private_poke_enabled", True)

        self._session: aiohttp.ClientSession | None = None

        logger.info(
            f"[ErisRAG] 初始化完成, enabled={self.enabled}, "
            f"url={self.rag_url}, timeout={self.timeout_s}s"
        )

    # ── HTTP ──

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            )
        return self._session

    async def terminate(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 工具 ──

    @staticmethod
    def _extract_text(content) -> str:
        """从 OpenAI 格式的 content 中提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def _build_conversation_context(self, req: ProviderRequest) -> list[str]:
        """从 req.contexts 提取最近 N 条对话文本（user/assistant 交替）。"""
        msgs: list[str] = []
        for ctx in (req.contexts or []):
            role = ctx.get("role", "")
            if role not in ("user", "assistant"):
                continue
            text = self._extract_text(ctx.get("content", ""))
            if text:
                msgs.append(text)
        return msgs[-self.context_count:] if self.context_count else []

    # ── 核心：on_llm_request 注入 RAG 人格 ──

    @filter.on_llm_request(priority=50)
    async def inject_rag_persona(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.enabled:
            return

        # 1) 提取用户信息
        sender_id = str(event.get_sender_id())
        sender_nickname = event.get_sender_name() or ""
        # 戳一戳等事件下 nickname 可能回退为 QQ 号，不传给服务器
        if sender_nickname == sender_id:
            sender_nickname = ""
        user_message = req.prompt or ""
        conversation_context = self._build_conversation_context(req)

        if not user_message:
            return

        # 2) 请求 RAG 服务器
        payload = {
            "user_message": user_message,
            "conversation_context": conversation_context,
            "sender_id": sender_id,
            "sender_nickname": sender_nickname,
        }

        try:
            session = await self._get_session()
            async with session.post(f"{self.rag_url}/retrieve", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            # 超时或失败 → 静默 fallback，保留 AstrBot 原 prompt
            logger.warning(f"[ErisRAG] RAG 请求失败，fallback:\n{traceback.format_exc()}")
            return

        enhanced_prompt = data.get("enhanced_system_prompt", "")
        if not enhanced_prompt:
            return

        # 3) 追加 RAG 人格到 AstrBot 已有的 system_prompt 之后
        req.system_prompt = (req.system_prompt or "") + "\n\n" + enhanced_prompt

        # 日志（调试用）
        metadata = data.get("metadata", {})
        logger.info(
            f"[ErisRAG] 已注入, tokens≈{metadata.get('total_tokens', '?')}, "
            f"L1={metadata.get('l1_modules_used', [])}, "
            f"L3={metadata.get('l3_scenes_used', [])}"
        )
        if self.debug_log:
            logger.info(
                f"[ErisRAG] === 完整 system_prompt ===\n{req.system_prompt}\n"
                f"[ErisRAG] === END ==="
            )

    # ── 私聊戳一戳：截取并调 LLM ──

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_private_poke(self, event: AstrMessageEvent):
        if not self.private_poke_enabled:
            return

        # 仅处理 aiocqhttp 平台
        if event.get_platform_name() != "aiocqhttp":
            return

        raw = getattr(event.message_obj, "raw_message", None)
        if not raw or not isinstance(raw, dict):
            return

        # 必须是戳一戳事件
        if (
            raw.get("post_type") != "notice"
            or raw.get("notice_type") != "notify"
            or raw.get("sub_type") != "poke"
        ):
            return

        # 仅私聊（无 group_id）
        if raw.get("group_id"):
            return

        # 必须戳的是 bot
        bot_id = raw.get("self_id")
        target_id = raw.get("target_id")
        if str(target_id) != str(bot_id):
            return

        sender_id = str(raw.get("user_id", ""))
        logger.info(f"[ErisRAG] 私聊戳一戳, sender={sender_id}")

        # 获取会话历史上下文
        umo = event.unified_msg_origin
        contexts = []
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if curr_cid:
                conv = await self.context.conversation_manager.get_conversation(umo, curr_cid)
                if conv and conv.history:
                    contexts = json.loads(conv.history)
        except Exception:
            pass

        # 提取最近对话文本给 RAG 服务器
        conv_ctx = []
        for ctx in contexts:
            role = ctx.get("role", "")
            if role in ("user", "assistant"):
                text = self._extract_text(ctx.get("content", ""))
                if text:
                    conv_ctx.append(text)
        conv_ctx = conv_ctx[-self.context_count:] if self.context_count else []

        # 请求 RAG 服务器获取人格 prompt
        system_prompt = ""
        poke_msg = "对方戳了戳你"
        if self.enabled:
            payload = {
                "user_message": poke_msg,
                "conversation_context": conv_ctx,
                "sender_id": sender_id,
                "sender_nickname": "",
            }
            try:
                session = await self._get_session()
                async with session.post(f"{self.rag_url}/retrieve", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                system_prompt = data.get("enhanced_system_prompt", "")
            except Exception:
                logger.warning(f"[ErisRAG] 私聊戳一戳 RAG 请求失败:\n{traceback.format_exc()}")

        # 调 LLM 生成回复
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo)
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=poke_msg,
                system_prompt=system_prompt or None,
                contexts=contexts,
            )
            reply = (resp.completion_text or "").strip()
            if reply:
                yield event.plain_result(reply)
        except Exception:
            logger.error(f"[ErisRAG] 私聊戳一戳 LLM 调用失败:\n{traceback.format_exc()}")

        # 阻止后续处理
        event.should_call_llm(False)

    # ── /ask 命令：直接查询 RAG 知识库 ──

    @filter.command("ask")
    async def ask_command(self, event: AstrMessageEvent, query: GreedyStr):
        query_str = str(query).strip()
        if not query_str:
            event.set_result(event.make_result().message("用法: /ask <问题>"))
            return

        payload = {"query": query_str, "top_k": 3, "format": "raw"}

        try:
            session = await self._get_session()
            async with session.post(f"{self.rag_url}/query", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            logger.error(f"[ErisRAG] /ask 查询失败:\n{traceback.format_exc()}")
            event.set_result(event.make_result().message("RAG 服务器无响应，稍后再试。"))
            return

        raw_text = data.get("raw_text", "")
        results = data.get("results", [])

        if not raw_text and not results:
            event.set_result(event.make_result().message("没有找到相关内容。"))
            return

        # 格式化结果
        lines = []
        for r in results[:3]:
            scene = r.get("scene_id", "?")
            vol = r.get("volume", "?")
            ch = r.get("chapter", "?")
            text = r.get("text", "")[:200]
            lines.append(f"[卷{vol} 第{ch}章 {scene}]\n{text}")

        reply = "\n\n".join(lines) if lines else raw_text[:500]
        event.set_result(event.make_result().message(reply))

    # ── /rag_health 命令：检查服务器状态 ──

    @filter.command("rag_health")
    async def health_command(self, event: AstrMessageEvent):
        try:
            session = await self._get_session()
            async with session.get(f"{self.rag_url}/health") as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            event.set_result(event.make_result().message("RAG 服务器不可达。"))
            return

        status = data.get("status", "unknown")
        scenes = data.get("scene_count", 0)
        emb = data.get("embedding_loaded", False)
        rerank = data.get("reranker_loaded", False)
        l1 = data.get("l1_loaded", False)
        l2 = data.get("l2_loaded", False)
        uptime = int(data.get("uptime_seconds", 0))

        msg = (
            f"RAG 状态: {status}\n"
            f"场景数: {scenes}\n"
            f"Embedding: {'OK' if emb else 'X'} | Reranker: {'OK' if rerank else 'X'}\n"
            f"L1: {'OK' if l1 else 'X'} | L2: {'OK' if l2 else 'X'}\n"
            f"运行时间: {uptime // 3600}h {uptime % 3600 // 60}m"
        )
        event.set_result(event.make_result().message(msg))
