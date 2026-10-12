"""Filter repos with Zhipu GLM AI, fallback to star-sort."""
import json
import requests

ZHIPU_API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def _build_prompt(repos: list[dict]) -> str:
    """Build prompt asking AI to score each repo."""
    repo_list = []
    for i, r in enumerate(repos, 1):
        desc = r["description"][:200] if r["description"] else "无描述"
        topics = ", ".join(r["topics"][:5]) if r["topics"] else "无"
        repo_list.append(
            f"{i}. [{r['full_name']}]({r['html_url']})\n"
            f"   描述: {desc}\n"
            f"   语言: {r['language']} | Stars: {r['stars']} | Topics: {topics}"
        )
    joined = "\n\n".join(repo_list)
    return f"""你是GitHub工具审核专家。以下{len(repos)}个GitHub仓库，请筛选出真正实用的开发者工具。
评分标准（1-5分）：
- 5分：非常实用，开发者必备
- 4分：有价值，值得关注
- 3分：一般，可用但非必需
- 2分：不太实用，或文档不全
- 1分：低质量/纯宣传/教程合集
排除以下类型：面试题合集、教程文档、XX大全、Awesome-list、空壳项目。

对每个项目，请返回：
- score: 1-5分
- summary: 用一句通顺的中文（30-60字）介绍这个项目是做什么的、解决什么问题，不要翻译原文，要用自己的话概括
- reason: 一句话说明推荐理由（20字以内）

只返回JSON数组，不要markdown代码块，不要其他文字：
[
  {{"index": 1, "score": 4, "summary": "xxx是一个用于xxx的xxx工具，可以帮助开发者xxx", "reason": "简洁的一句推荐理由"}},
  ...
]
以下是仓库列表：
{joined}"""


def filter_with_ai(repos: list[dict], api_key: str) -> list[dict]:
    """Score repos with Zhipu GLM, return those scoring >= 4."""
    if not repos:
        return []
    if not api_key:
        print("[filter] No Zhipu API key, falling back to star-sort")
        return filter_by_stars(repos)
    prompt = _build_prompt(repos)
    try:
        resp = requests.post(
            ZHIPU_API,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "glm-4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[filter] Zhipu API error {resp.status_code}: {resp.text[:200]}")
            return filter_by_stars(repos)
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_scores(content)
    except (requests.RequestException, KeyError, json.JSONDecodeError, TypeError) as e:
        print(f"[filter] AI filter failed: {e}, falling back to star-sort")
        return filter_by_stars(repos)

    scored = []
    for s in scores:
        idx = s["index"] - 1
        if 0 <= idx < len(repos) and s["score"] >= 4:
            r = repos[idx].copy()
            r["score"] = s["score"]
            r["summary"] = s.get("summary", "")
            r["reason"] = s.get("reason", "")
            scored.append(r)
    scored.sort(key=lambda r: r["score"], reverse=True)
    print(f"[filter] AI scored {len(scores)} repos, {len(scored)} passed (>=4)")
    return scored


def filter_by_stars(repos: list[dict], top_n: int = 10) -> list[dict]:
    """Fallback: take top N repos by star count."""
    sorted_repos = sorted(repos, key=lambda r: r["stars"], reverse=True)[:top_n]
    for r in sorted_repos:
        r["score"] = 0
        r["summary"] = r.get("description", "")
        r["reason"] = "Star排名自动入选（AI未评分）"
    return sorted_repos


def _parse_scores(content: str) -> list[dict]:
    """Parse AI response. Handle markdown code blocks if present."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:]) if lines[0].startswith("```") else content
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return json.loads(content)
