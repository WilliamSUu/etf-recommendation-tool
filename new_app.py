from __future__ import annotations
from datetime import datetime
from pathlib import Path
import sys
import threading

from flask import Flask, render_template_string, jsonify, request
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from etf_recommender.data.etf_market import MarketDataError, fetch_selected_etf_quotes, get_quote_map
from etf_recommender.services.cache import load_news_cache, load_recommendations_cache
from etf_recommender.services.refresh import RefreshError, refresh_dashboard_data
from etf_recommender.services.scheduler import run_missed_refreshes, start_background_scheduler

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

_app_state = {
    "investment_horizon": "短线",
    "last_quote_records": [],
    "last_quote_success_at": None,
}

def format_time(value: object) -> str:
    if not value:
        return "--"
    return str(value).replace("T", " ")[:19]

def init_app_bg():
    try:
        start_background_scheduler()
        run_missed_refreshes()
    except Exception as e:
        print(f"后台初始化: {e}")

threading.Thread(target=init_app_bg, daemon=True).start()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/news')
def api_news():
    try:
        news_cache = load_news_cache()
        if not news_cache:
            return jsonify({"success": False, "message": "暂无资讯缓存，请点击刷新"})
        items = news_cache.get("items", [])
        return jsonify({"success": True, "refreshed_at": format_time(news_cache.get('refreshed_at')), "items": items[:8]})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/recommendations')
def api_recommendations():
    try:
        # 获取用户选择的投资周期
        investment_horizon = request.args.get('horizon', '短线')
        print(f"[API] 推荐列表请求，投资周期: {investment_horizon}")
        
        rec_cache = load_recommendations_cache()
        if not rec_cache:
            return jsonify({"success": False, "message": "暂无推荐缓存，请点击刷新"})
        
        items = rec_cache.get("items", [])
        
        # 这里的 items 应该已经按照投资周期进行了硬性筛选
        # 因为在 refresh_dashboard_data 中已经应用了 investment_horizon
        # 但缓存可能是用不同的周期生成的，所以我们需要重新筛选
        # 注意：实际的硬性筛选应该在刷新数据时进行
        
        print(f"[API] 返回 {len(items)} 条推荐")
        return jsonify({
            "success": True,
            "refreshed_at": format_time(rec_cache.get('refreshed_at')),
            "candidate_count": rec_cache.get('candidate_count', '-'),
            "items": items[:10],
            "investment_horizon": investment_horizon
        })
    except Exception as e:
        print(f"[API] 推荐列表错误: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/quotes')
def api_quotes():
    try:
        codes = request.args.getlist('codes')
        if not codes:
            return jsonify({"success": True, "quotes": {}, "status_text": "无数据"})
        try:
            quotes_df = fetch_selected_etf_quotes(codes)
            _app_state["last_quote_records"] = quotes_df.to_dict("records")
            _app_state["last_quote_success_at"] = datetime.now().strftime("%H:%M:%S")
        except MarketDataError:
            if _app_state.get("last_quote_records"):
                quotes_df = pd.DataFrame(_app_state["last_quote_records"])
            else:
                quotes_df = pd.DataFrame()
        quote_map = get_quote_map(quotes_df) if not quotes_df.empty else {}
        quotes_dict = {}
        for code, quote in quote_map.items():
            if quote:
                quotes_dict[code] = {"name": quote.name, "price": float(quote.price) if quote.price else None, "change_pct": float(quote.change_pct) if quote.change_pct else None}
        last_success_at = _app_state.get("last_quote_success_at") or datetime.now().strftime("%H:%M:%S")
        return jsonify({"success": True, "quotes": quotes_dict, "status_text": f"行情更新：{last_success_at}"})
    except Exception as e:
        return jsonify({"success": True, "quotes": {}, "status_text": "行情加载失败"})

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    try:
        investment_horizon = request.json.get('investment_horizon', _app_state["investment_horizon"])
        _app_state["investment_horizon"] = investment_horizon
        refresh_dashboard_data(trigger="manual", investment_horizon=investment_horizon)
        return jsonify({"success": True, "message": "刷新完成"})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF 推荐工具</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #f8f6f2;
            --bg-secondary: #faf8f4;
            --text-dark: #1a1410;
            --text-muted: #7a7370;
            --border-light: #e8e4dc;
            --accent: #b3872e;
        }
        html, body { background: var(--bg-primary); color: var(--text-dark); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; height: 100%; }
        body { overflow: hidden; }
        .container { display: flex; flex-direction: column; height: 100vh; padding: 1.2rem 1.5rem; }
        .header { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 1rem; margin-bottom: 0.8rem; align-items: center; }
        h1 { font-size: 2rem; font-weight: 700; margin: 0; }
        .subtitle { margin: 0.2rem 0 0; color: var(--text-muted); font-size: 0.9rem; }
        .clock { font-family: "SF Mono", monospace; font-size: 1.3rem; text-align: right; font-weight: 600; }
        .toolbar { display: grid; grid-template-columns: 0.8fr 2.2fr 1.2fr; gap: 1rem; margin-bottom: 1rem; align-items: center; }
        button { background: linear-gradient(135deg, #3b3422 0%, #4a3d2a 100%); color: #f5dfa5; border: 1px solid rgba(245,223,165,0.3); border-radius: 6px; font-weight: 600; padding: 0.5rem 1rem; cursor: pointer; height: 2.4rem; transition: all 0.2s; }
        button:hover { background: linear-gradient(135deg, #4a3d2a 0%, #5a4a32 100%); }
        button:disabled { opacity: 0.6; }
        #horizonBtn { height: 2.0rem; padding: 0 0.4rem; display: flex; align-items: center; justify-content: flex-end; gap: 0; border: none; }
        .horizon-option { flex: 1; justify-content: center; display: inline-flex; align-items: center; gap: 0.2rem; padding: 0 0.5rem; cursor: pointer; font-size: 0.8rem; user-select: none; }
        .horizon-dot { font-weight: bold; min-width: 0.4rem; }
        .horizon-divider { display: inline-block; width: 1px; height: 1.2rem; background: rgba(245, 223, 165, 0.4); margin: 0 0.2rem; }
        .toolbar-label { color: var(--text-muted); font-size: 0.8rem; }
        .content { display: grid; grid-template-columns: 0.95fr 1.05fr; gap: 1.5rem; flex: 1; min-height: 0; }
        .panel { display: flex; flex-direction: column; min-height: 0; }
        h3 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
        .panel-caption { color: var(--text-muted); font-size: 0.8rem; margin-bottom: 0.5rem; }
        .panel-content { flex: 1; overflow-y: auto; padding-right: 0.5rem; }
        .panel-content::-webkit-scrollbar { width: 6px; }
        .panel-content::-webkit-scrollbar-thumb { background: #d8d0c4; border-radius: 3px; }
        .loading { text-align: center; padding: 2rem; color: var(--text-muted); }
        .loading::after { content: ""; display: inline-block; width: 1rem; height: 1rem; margin-left: 0.5rem; border: 2px solid var(--text-muted); border-right-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* 新闻卡片 */
        .news-card { background: linear-gradient(135deg, var(--bg-secondary) 0%, #fbf9f5 100%); border: 0.75px solid var(--border-light); border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0; cursor: pointer; transition: all 0.2s; }
        .news-card:hover { background: #fff; box-shadow: 0 4px 12px rgba(26, 20, 16, 0.08); }
        .news-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem; }
        .news-index { display: inline-block; background: #e8e4dc; color: #5a4a35; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.7rem; font-weight: 600; min-width: 1.5rem; text-align: center; }
        .news-content { display: none; }
        .news-card.expanded .news-content { display: block; }
        .news-summary { font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.5rem; color: var(--text-dark); }
        .news-section-title { font-weight: 600; font-size: 0.8rem; margin-top: 0.4rem; margin-bottom: 0.2rem; }
        .news-points { font-size: 0.75rem; line-height: 1.4; margin-left: 1rem; }
        .news-points li { margin: 0.2rem 0; }
        .news-assets { font-size: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.3rem 0; }
        .asset-tag { display: inline-block; background: #fdfbf8; border: 1px solid #ddd4c8; color: #5a4a35; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 500; }
        .news-meta { font-size: 0.7rem; color: var(--text-muted); padding-top: 0.4rem; border-top: 1px solid var(--border-light); line-height: 1.5; }
        .news-meta a { color: var(--accent); text-decoration: none; font-weight: 600; }
        
        /* ETF 卡片 */
        .etf-card { background: linear-gradient(135deg, var(--bg-secondary) 0%, #fbf9f5 100%); border: 0.75px solid var(--border-light); border-left: 4px solid var(--accent); border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0; transition: all 0.2s; position: relative; overflow: hidden; }
        .etf-card:hover { background: #fff; box-shadow: 0 4px 12px rgba(26, 20, 16, 0.08); }
        .etf-header { display: grid; grid-template-columns: auto 1fr auto; gap: 0.75rem; align-items: start; margin-bottom: 0.5rem; }
        .etf-rank { display: flex; align-items: center; justify-content: center; width: 2.2rem; height: 2.2rem; border-radius: 50%; background: linear-gradient(135deg, #3b3422 0%, #4a3d2a 100%); color: #f5dfa5; font-weight: 700; font-size: 0.85rem; box-shadow: 0 2px 6px rgba(58, 43, 18, 0.2); }
        .etf-info { min-width: 0; }
        .etf-name { font-size: 0.9rem; font-weight: 600; color: var(--text-dark); }
        .etf-code { font-size: 0.75rem; color: #8a7d75; font-family: monospace; margin-top: 0.05rem; }
        .etf-price-block { text-align: right; }
        .etf-price { font-size: 1.2rem; font-weight: 700; color: var(--text-dark); font-family: monospace; line-height: 1; }
        .etf-change { font-size: 0.9rem; font-weight: 700; margin-top: 0.1rem; }
        .etf-change.up { color: #c23b2e; }
        .etf-change.down { color: #16824d; }
        .etf-scores { display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #8a7d75; margin: 0.4rem 0; }
        .etf-reason { font-size: 0.8rem; line-height: 1.4; margin: 0.4rem 0; color: var(--text-dark); }
        .etf-related { font-size: 0.7rem; color: var(--text-muted); margin: 0.3rem 0; }
        .etf-badge { display: inline-block; background: rgba(139, 111, 71, 0.1); color: #8b6f47; border: 1px solid rgba(139, 111, 71, 0.25); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 600; margin-top: 0.3rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>ETF 推荐工具</h1>
                <p class="subtitle">腾讯资讯 · DeepSeek 分析 · A 股 ETF 实时行情</p>
            </div>
            <div></div>
            <div class="clock" id="clock">--:--:--</div>
        </div>
        <script>
            function updateClock() {
                const now = new Date();
                const h = String(now.getHours()).padStart(2, "0");
                const m = String(now.getMinutes()).padStart(2, "0");
                const s = String(now.getSeconds()).padStart(2, "0");
                document.getElementById("clock").textContent = h + ":" + m + ":" + s;
            }
            updateClock();
            setInterval(updateClock, 1000);
        </script>

        <div class="toolbar">
            <button id="refreshBtn" onclick="handleRefresh()">刷新资讯</button>
            <div class="toolbar-label">自动刷新：交易日 09:00 / 15:00 运行；行情每 3 秒刷新。</div>
            <button id="horizonBtn" class="horizon-btn">
                <span class="horizon-option" data-value="超短线"><span class="horizon-dot">○</span> 超短线</span>
                <span class="horizon-divider"></span>
                <span class="horizon-option" data-value="短线"><span class="horizon-dot">●</span> 短线</span>
                <span class="horizon-divider"></span>
                <span class="horizon-option" data-value="中线"><span class="horizon-dot">○</span> 中线</span>
                <span class="horizon-divider"></span>
                <span class="horizon-option" data-value="长线"><span class="horizon-dot">○</span> 长线</span>
            </button>
        </div>

        <div class="content">
            <div class="panel">
                <h3>财经资讯</h3>
                <div class="panel-caption" id="newsCaption">加载中...</div>
                <div class="panel-content" id="newsContent"><div class="loading">加载中</div></div>
            </div>
            <div class="panel">
                <h3>ETF 推荐榜</h3>
                <div class="panel-caption" id="recCaption">加载中...</div>
                <div class="panel-content" id="recContent"><div class="loading">加载中</div></div>
            </div>
        </div>
    </div>

    <script>
        let allNewsItems = [];
        let allRecItems = [];
        let appState = {
            investmentHorizon: "短线",
        };

        async function loadNews() {
            try {
                const response = await fetch("/api/news");
                const data = await response.json();
                
                if (!data.success) {
                    document.getElementById("newsCaption").textContent = data.message;
                    document.getElementById("newsContent").innerHTML = `<div style="padding:1rem; color:var(--text-muted);">${data.message}</div>`;
                    return;
                }

                allNewsItems = data.items;
                document.getElementById("newsCaption").textContent = `最近刷新：${data.refreshed_at} · 来源：腾讯财经/自选股`;
                renderNews();
            } catch (e) {
                console.error("新闻加载失败:", e);
                document.getElementById("newsContent").innerHTML = `<div style="padding:1rem; color:red;">加载失败: ${e.message}</div>`;
            }
        }

        function renderNews() {
            let html = "";
            allNewsItems.forEach((item, i) => {
                const keyPoints = (item.key_points || []).map(p => `<li>${escapeHtml(p)}</li>`).join("");
                const assets = (item.assets || []).map(a => `<span class="asset-tag">${escapeHtml(a)}</span>`).join("");
                const expandedClass = i === 0 ? ' expanded' : '';  // 第一条资讯默认展开

                html += `
                    <div class="news-card${expandedClass}" onclick="this.classList.toggle('expanded')">
                        <div class="news-title">
                            <span class="news-index">${i+1}</span>
                            ${escapeHtml(item.title || "")}
                        </div>
                        <div class="news-content">
                            <div class="news-summary">${escapeHtml(item.summary || "")}</div>
                            ${keyPoints ? `<div><div class="news-section-title">关键影响点</div><ul class="news-points">${keyPoints}</ul></div>` : ""}
                            ${assets ? `<div><div class="news-section-title">涉及行业/资产</div><div class="news-assets">${assets}</div></div>` : ""}
                            <div class="news-meta">
                                来源：${escapeHtml(item.source || "腾讯财经")}<br>
                                时间：${formatTime(item.published_at)}<br>
                                链接：<a href="${escapeHtml(item.url || "#")}" target="_blank">打开原文</a>
                            </div>
                        </div>
                    </div>
                `;
            });
            document.getElementById("newsContent").innerHTML = html;
        }

        async function loadRecommendations() {
            try {
                console.log("📊 开始加载推荐列表...");
                
                const response = await fetch("/api/recommendations?horizon=" + appState.investmentHorizon);
                console.log("📊 获取响应:", response.status);
                
                const data = await response.json();
                console.log("📊 解析 JSON 完成，数据:", data);
                
                if (!data.success) {
                    console.warn("⚠️  API 返回失败:", data.message);
                    document.getElementById("recCaption").textContent = data.message;
                    document.getElementById("recContent").innerHTML = `<div style="padding:1rem; color:var(--text-muted);">${data.message}</div>`;
                    return;
                }

                console.log("📊 设置推荐数据:", data.items.length, "项");
                allRecItems = data.items;
                
                console.log("📊 更新标题信息");
                document.getElementById("recCaption").textContent = `推荐生成：${data.refreshed_at} · 候选池：${data.candidate_count}`;
                
                console.log("📊 渲染推荐列表");
                renderRecommendations();
                console.log("📊 推荐列表渲染完成");
                
                // 启动行情刷新（3秒）
                console.log("📊 启动行情更新");
                if (window.quoteRefreshInterval) clearInterval(window.quoteRefreshInterval);
                window.quoteRefreshInterval = setInterval(updateQuotes, 3000);
                updateQuotes();
                
                console.log("✅ 推荐列表加载完成");
            } catch (e) {
                console.error("❌ 推荐加载失败:", e);
                console.error("错误堆栈:", e.stack);
                document.getElementById("recContent").innerHTML = `<div style="padding:1rem; color:red;">加载失败: ${e.message}</div>`;
            }
        }

        function renderRecommendations() {
            console.log("🎨 开始渲染推荐列表，项数:", allRecItems.length);
            let html = "";
            allRecItems.forEach((item, i) => {
                const code = String(item.code || "").padStart(6, "0");
                const relevance = parseFloat(item.relevance_score || 0).toFixed(1);
                const impact = parseFloat(item.impact_score || 0).toFixed(1);
                const duration = parseFloat(item.duration_score || 0).toFixed(1);
                const total = parseFloat(item.total_score || 0).toFixed(1);
                const relatedNews = (item.related_news || []).join("、");
                
                html += `
                    <div class="etf-card" data-code="${code}">
                        <div class="etf-header">
                            <div class="etf-rank">#${i+1}</div>
                            <div class="etf-info">
                                <div class="etf-name">${escapeHtml(item.name || "")}</div>
                                <div class="etf-code">${code}</div>
                            </div>
                            <div class="etf-price-block">
                                <div class="etf-price" data-price>--</div>
                                <div class="etf-change" data-change>--</div>
                            </div>
                        </div>
                        <div class="etf-scores">
                            <span>相关${relevance}·力度${impact}·长度${duration}</span>
                            <span style="font-weight: 700; color: var(--text-dark);">${total}</span>
                        </div>
                        <div class="etf-reason">${escapeHtml(item.reason || "")}</div>
                        <div class="etf-badge">📢 综合</div>
                        ${relatedNews ? `<div class="etf-related">资讯：${escapeHtml(relatedNews)}</div>` : ""}
                    </div>
                `;
            });
            console.log("🎨 HTML 已渲染到页面，长度:", html.length);
            document.getElementById("recContent").innerHTML = html;
            console.log("🎨 推荐列表渲染完成");
        }

        async function updateQuotes() {
            try {
                const codes = Array.from(document.querySelectorAll(".etf-card")).map(c => c.dataset.code).filter(c => c);
                if (codes.length === 0) return;
                
                const params = new URLSearchParams();
                codes.forEach(code => params.append("codes", code));
                
                const response = await fetch("/api/quotes?" + params.toString());
                const data = await response.json();
                
                if (data.success) {
                    document.querySelectorAll(".etf-card").forEach(card => {
                        const code = card.dataset.code;
                        const quote = data.quotes[code];
                        if (quote) {
                            const priceEl = card.querySelector("[data-price]");
                            const changeEl = card.querySelector("[data-change]");
                            if (priceEl) priceEl.textContent = quote.price ? quote.price.toFixed(3) : "--";
                            if (changeEl) {
                                const pct = quote.change_pct;
                                changeEl.textContent = pct ? (pct > 0 ? "+" : "") + pct.toFixed(2) + "%" : "--";
                                changeEl.className = "etf-change " + (pct >= 0 ? "up" : "down");
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("行情更新失败:", e);
            }
        }

        async function handleRefresh() {
            const btn = document.getElementById("refreshBtn");
            btn.disabled = true;
            btn.textContent = "刷新中...";
            try {
                const horizon = appState.investmentHorizon || "短线";
                console.log("开始刷新资讯，当前投资周期:", horizon);

                const response = await fetch("/api/refresh", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ investment_horizon: horizon })
                });
                const data = await response.json();

                if (data.success) {
                    console.log("✅ 刷新完成");
                    // 重新加载财经资讯
                    await loadNews();
                    // 根据当前投资周期重新生成推荐列表
                    await loadRecommendations();
                } else {
                    alert("刷新失败: " + data.message);
                }
            } catch (e) {
                alert("刷新失败: " + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = "刷新资讯";
            }
        }

        function escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }

        function formatTime(value) {
            if (!value) return "--";
            return String(value).replace("T", " ").substring(0, 19);
        }

        loadNews();
        loadRecommendations();

        // 投资周期选择按钮事件处理
        document.querySelectorAll(".horizon-option").forEach(option => {
            option.addEventListener("click", async (e) => {
                e.preventDefault();
                e.stopPropagation();

                const newHorizon = option.dataset.value;
                console.log("✅ 投资周期已改变:", newHorizon);

                // 更新按钮显示（圆点）
                document.querySelectorAll(".horizon-option").forEach(opt => {
                    const dot = opt.querySelector(".horizon-dot");
                    if (opt.dataset.value === newHorizon) {
                        dot.textContent = "●";  // 实心圆
                    } else {
                        dot.textContent = "○";  // 空心圆
                    }
                });

                // 显示加载指示器
                const recContent = document.getElementById("recContent");
                recContent.innerHTML = '<div class="loading">正在重新筛选指数...</div>';

                // 更新应用状态
                if (typeof appState === 'undefined') {
                    window.appState = { investmentHorizon: newHorizon };
                } else {
                    appState.investmentHorizon = newHorizon;
                }

                try {
                    console.log("📊 根据新投资周期刷新数据...");
                    // 第一步：根据新的投资周期刷新数据（进行硬性筛选）
                    const refreshResponse = await fetch("/api/refresh", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ investment_horizon: newHorizon })
                    });
                    const refreshData = await refreshResponse.json();

                    if (!refreshData.success) {
                        console.warn("⚠️  刷新失败:", refreshData.message);
                        // 即使刷新失败，仍然尝试加载推荐
                    }

                    console.log("📊 刷新完成，现在加载推荐列表...");
                    // 第二步：加载新的推荐列表（已根据投资周期硬性筛选）
                    await loadRecommendations();
                    console.log("✅ 推荐列表已更新");
                } catch (error) {
                    console.error("❌ 推荐列表更新失败:", error);
                    recContent.innerHTML = '<div style="padding:1rem; color:red;">刷新失败: ' + error.message + '</div>';
                }
            });
        });
    </script>
</body>
</html>

'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
