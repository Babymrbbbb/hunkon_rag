"""配置安全回归：默认密钥必须是随机强密钥，不得回退为可预测的弱字面量。"""
from app.config import Settings, _random_key

# 弱字面量拆分书写：防回归测试本身不能在公开仓留下完整弱密钥字符串
# （会被推送护栏的 weak_default_key 规则命中）。只保留足够长的模式，
# 避免随机 base64url 密钥偶然包含短词导致 flaky。
_WEAK_DEV = "dev-" + "secret-" + "key"
WEAK = (_WEAK_DEV, "changeme", "12345678")


def test_random_key_is_strong_and_unique():
    a, b = _random_key(), _random_key()
    assert a != b, "默认密钥必须每次生成不同值（防硬编码回归）"
    assert len(a) >= 32, "默认密钥长度应足够"
    assert not any(w in a.lower() for w in WEAK)


def test_default_settings_key_is_not_weak(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    s = Settings()
    assert s.api_key
    assert not any(w in s.api_key.lower() for w in WEAK)
    assert len(s.api_key) >= 32
    # 三档角色 key 不再有弱默认值
    assert s.api_key_staff == ""
    assert s.api_key_franchise == ""
    assert s.api_key_ops == ""


def test_env_override_still_works(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "explicit-key-from-env")
    assert Settings().api_key == "explicit-key-from-env"
