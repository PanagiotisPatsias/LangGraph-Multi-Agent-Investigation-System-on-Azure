"""Smoke test: verifies Langfuse + Azure OpenAI integration end-to-end.

Makes one tiny LLM call with the Langfuse callback attached, then flushes.
If successful, the trace appears at https://cloud.langfuse.com within seconds.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.config.llm_factory import create_llm
from src.config.settings import get_langfuse_callback, get_settings


def main() -> None:
    settings = get_settings()
    handler = get_langfuse_callback()

    if handler is None:
        print("✗ Langfuse keys not set in .env — aborting")
        return

    print(f"→ Langfuse host: {settings.langfuse_host}")
    print(f"→ Public key:    {settings.langfuse_public_key[:12]}…")
    print("→ Sending test prompt to Azure OpenAI with Langfuse trace…")

    llm = create_llm(temperature=0.0, max_tokens=30)
    response = llm.invoke(
        [HumanMessage(content="Say 'Langfuse smoke test OK' in exactly five words.")],
        config={"callbacks": [handler], "metadata": {"langfuse_tags": ["smoke-test"]}},
    )

    print(f"← LLM said: {response.content!r}")

    from langfuse import get_client

    get_client().flush()
    print("\n✓ Trace flushed. Check the dashboard:")
    print(f"  {settings.langfuse_host}")


if __name__ == "__main__":
    main()
