"""
美股AI分析系统 - 专业版
支持所有美股实时数据，专家级分析，策略存储
集成 OpenRouter API + Gemini Pro + Claude Opus 4.6
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import time
import random
import hashlib
import json
import os
import requests

# 尝试导入yfinance，如果不存在则使用模拟数据
try:
    import yfinance as yf
    import pandas as pd
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    pd = None
    print("⚠️  yfinance 未安装，使用模拟数据。安装: pip install yfinance")

app = Flask(__name__)

# ============ API 配置 ============
# OpenRouter API - 获取密钥: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 如果未设置环境变量，使用配置的密钥
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "your-openrouter-api-key-here"

# 数据缓存
CACHE = {}
CACHE_EXPIRY = 600  # 缓存10分钟
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 3  # API限流保护

# 持久化缓存文件
FUNDAMENTAL_CACHE_FILE = '/tmp/fundamental_cache.json'
FUNDAMENTAL_CACHE_EXPIRY = 3600  # 基本面数据缓存1小时

# 加载持久化缓存
def load_fundamental_cache():
    """加载基本面数据持久化缓存"""
    try:
        if os.path.exists(FUNDAMENTAL_CACHE_FILE):
            with open(FUNDAMENTAL_CACHE_FILE, 'r') as f:
                cache = json.load(f)
                # 清理过期缓存
                current_time = time.time()
                return {k: v for k, v in cache.items() if current_time - v.get('time', 0) < FUNDAMENTAL_CACHE_EXPIRY}
    except:
        pass
    return {}

# 保存持久化缓存
def save_fundamental_cache(cache):
    """保存基本面数据到持久化缓存"""
    try:
        with open(FUNDAMENTAL_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

# 全局基本面缓存
fundamental_cache = load_fundamental_cache()

# API请求计数器（用于限流）
API_CALL_COUNTS = {}

# 策略存储文件
STRATEGY_FILE = '/tmp/stock_strategies.json'
WATCHLIST_FILE = '/tmp/stock_watchlist.json'


# ============ Yahoo Finance 股票数据获取 ============

def get_fundamental_data_cached(symbol):
    """获取基本面数据（带缓存）"""
    cache_key = f"fund_{symbol}"
    current_time = time.time()

    # 检查内存缓存
    if cache_key in fundamental_cache:
        cached = fundamental_cache[cache_key]
        if current_time - cached.get('time', 0) < FUNDAMENTAL_CACHE_EXPIRY:
            return cached.get('data', {})

    # 从yfinance获取数据
    if HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # 提取关键基本面数据
            data = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_assets': info.get('totalAssets', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
            }

            # 保存到缓存
            fundamental_cache[cache_key] = {
                'data': data,
                'time': current_time
            }
            save_fundamental_cache(fundamental_cache)

            return data
        except Exception as e:
            print(f"⚠️ {symbol} 基本面数据获取失败: {e}")

    return {}


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    if len(prices) < period + 1:
        return 50

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)


def get_yfinance_stock_data(symbol):
    """使用 yfinance 获取真实股票数据"""
    symbol = symbol.upper().strip()
    if not symbol:
        return None

    # 检查缓存
    cache_key = f"stock_{symbol}"
    current_time = time.time()
    if cache_key in CACHE:
        cached_data, cached_time = CACHE[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            return cached_data

    # API限流保护
    rate_limit_pause()

    # 方法1: 尝试使用yfinance
    if HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)

            # 使用快速历史数据获取当前价格（避免频繁调用.info）
            hist = ticker.history(period="5d")
            if hist is not None and len(hist) > 0:
                latest = hist.iloc[-1]
                current_price = float(latest['Close'])
                open_price = float(latest['Open'])
                day_high = float(latest['High'])
                day_low = float(latest['Low'])
                volume = int(latest['Volume'])

                # 获取前一天收盘价计算涨跌
                if len(hist) > 1:
                    prev_close = float(hist.iloc[-2]['Close'])
                else:
                    prev_close = current_price

                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0

                # 计算RSI
                hist_long = ticker.history(period="3mo")
                if hist_long is not None and len(hist_long) > 14:
                    close_prices = hist_long['Close'].tolist()
                    rsi = calculate_rsi(close_prices)
                else:
                    rsi = 50

                # 获取info（可能被限流，所以先尝试）
                info = {}
                try:
                    info = ticker.info
                    name = info.get('longName', info.get('shortName', f'{symbol} Corp'))
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', info.get('forwardPE', 0))
                    eps = info.get('epsTrailingTwelveMonths', 0)
                    dividend_yield = info.get('dividendYield', 0)
                    high_52w = info.get('fiftyTwoWeekHigh', current_price)
                    low_52w = info.get('fiftyTwoWeekLow', current_price)
                except:
                    # info调用失败，使用默认值
                    name = f'{symbol} Corp'
                    market_cap = 0
                    pe_ratio = 0
                    eps = 0
                    dividend_yield = 0
                    high_52w = current_price * 1.3
                    low_52w = current_price * 0.7

                trend = 'up' if change > 0 else 'down' if change < 0 else 'neutral'

                # 保存历史数据用于技术分析
                hist_for_analysis = None
                try:
                    hist_long = ticker.history(period="3mo")
                    if hist_long is not None and len(hist_long) > 20:
                        hist_for_analysis = hist_long
                except:
                    pass

                result = {
                    'symbol': symbol,
                    'name': name,
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2),
                    'open': round(open_price, 2),
                    'high': round(day_high, 2),
                    'low': round(day_low, 2),
                    'volume': volume,
                    'market_cap': int(market_cap) if market_cap else 0,
                    'pe_ratio': round(float(pe_ratio), 2) if pe_ratio else 0,
                    'eps': round(float(eps), 2) if eps else 0,
                    'dividend_yield': round(float(dividend_yield * 100), 2) if dividend_yield else 0,
                    'high_52w': round(float(high_52w), 2) if high_52w else 0,
                    'low_52w': round(float(low_52w), 2) if low_52w else 0,
                    'resistance': round(current_price * 1.03, 2),
                    'support': round(current_price * 0.97, 2),
                    'rsi': rsi,
                    'trend': trend,
                    '_info': info,
                    '_hist_data': hist_for_analysis
                }

                # 缓存数据
                CACHE[cache_key] = (result, current_time)
                return result

        except Exception as e:
            print(f"⚠️ {symbol} yfinance 获取失败: {e}")

    # 方法2: 使用Yahoo Finance API作为备用
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result = data.get('chart', {}).get('result', [])
            if result:
                meta = result[0].get('meta', {})
                current_price = meta.get('regularMarketPrice')
                prev_close = meta.get('previousClose', current_price)

                if current_price:
                    change = current_price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0

                    # 获取历史数据计算RSI
                    indicators = result[0].get('indicators', {})
                    quote = indicators.get('quote', [{}])[0]
                    closes = quote.get('close', [])
                    if closes and len(closes) > 14:
                        rsi = calculate_rsi([c for c in closes if c is not None])
                    else:
                        rsi = 50

                    name = meta.get('longName', f'{symbol} Corp')

                    stock_result = {
                        'symbol': symbol,
                        'name': name,
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2),
                        'open': round(meta.get('regularMarketOpen', current_price), 2),
                        'high': round(meta.get('regularMarketDayHigh', current_price), 2),
                        'low': round(meta.get('regularMarketDayLow', current_price), 2),
                        'volume': int(meta.get('regularMarketVolume', 0)),
                        'market_cap': int(meta.get('marketCap', 0)),
                        'pe_ratio': 0,
                        'eps': 0,
                        'dividend_yield': 0,
                        'high_52w': round(meta.get('fiftyTwoWeekHigh', current_price), 2),
                        'low_52w': round(meta.get('fiftyTwoWeekLow', current_price), 2),
                        'resistance': round(current_price * 1.03, 2),
                        'support': round(current_price * 0.97, 2),
                        'rsi': rsi,
                        'trend': 'up' if change > 0 else 'down' if change < 0 else 'neutral',
                        '_info': {}
                    }

                    CACHE[cache_key] = (stock_result, current_time)
                    return stock_result

    except Exception as e:
        print(f"⚠️ {symbol} Yahoo API 备用获取失败: {e}")

    return None


# ============ AI 模型分析 ============

def call_openrouter_api(model, messages, system_prompt="", max_tokens=500, temperature=0.7):
    """通用的 OpenRouter API 调用函数"""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        return {'error': 'OpenRouter API Key 未配置', 'success': False}

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'http://localhost:5000',
        'X-Title': 'Stock Analysis System'
    }

    # 构建消息列表
    api_messages = []
    if system_prompt:
        api_messages.append({'role': 'system', 'content': system_prompt})
    api_messages.extend(messages)

    payload = {
        'model': model,
        'messages': api_messages,
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return {'content': content, 'success': True}
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
            print(f"⚠️ OpenRouter API 错误: {error_msg}")
            return {'error': error_msg, 'success': False}

    except requests.exceptions.Timeout:
        return {'error': 'API 请求超时', 'success': False}
    except Exception as e:
        print(f"⚠️ OpenRouter API 异常: {e}")
        return {'error': str(e), 'success': False}


def analyze_with_gemini(symbol, stock_data, market_context=""):
    """使用 Gemini Pro 进行 AI 分析"""
    prompt = f"""作为专业股票分析师，请分析以下美股：

股票代码: {symbol}
公司名称: {stock_data.get('name', '')}
当前价格: ${stock_data.get('price', 0)}
涨跌幅: {stock_data.get('change_pct', 0)}%
52周高点: ${stock_data.get('high_52w', 0)}
52周低点: ${stock_data.get('low_52w', 0)}
RSI: {stock_data.get('rsi', 50)}
市盈率: {stock_data.get('pe_ratio', 0)}

{market_context}

请提供：
1. 技术分析（趋势、支撑/阻力位）
2. 风险评估（1-10分）
3. 短期观点（1-3个月）
4. 关键催化剂

请用中文简洁回复，每项不超过2句话。"""

    result = call_openrouter_api(
        model='google/gemini-2.5-pro',
        messages=[{'role': 'user', 'content': prompt}],
        system_prompt='你是专业的美股分析师，擅长技术分析和风险评估。',
        max_tokens=500,
        temperature=0.7
    )

    if result.get('success'):
        return {
            'model': 'Gemini Pro',
            'analysis': result['content'],
            'success': True
        }
    else:
        return {
            'model': 'Gemini Pro',
            'analysis': f"分析不可用: {result.get('error', '未知错误')}",
            'success': False
        }


def analyze_with_claude(symbol, stock_data, market_context=""):
    """使用 Claude Opus 4.6 进行 AI 分析"""
    prompt = f"""作为资深投资策略师，请分析这只股票：

{symbol} - {stock_data.get('name', '')}
价格: ${stock_data.get('price', 0)} ({stock_data.get('change_pct', 0):+.2f}%)
技术指标: RSI={stock_data.get('rsi', 50)}, 趋势={stock_data.get('trend', 'neutral')}
估值: PE={stock_data.get('pe_ratio', 0)}

{market_context}

请给出：
1. 投资建议（买入/持有/卖出）
2. 信心级别（高/中/低）
3. 关键风险因素
4. 潜在催化剂

用中文回复，简洁专业。"""

    result = call_openrouter_api(
        model='anthropic/claude-opus-4.6',
        messages=[{'role': 'user', 'content': prompt}],
        system_prompt='你是经验丰富的投资策略师，擅长基本面和量化分析。',
        max_tokens=500,
        temperature=0.5
    )

    if result.get('success'):
        return {
            'model': 'Claude Opus 4.6',
            'analysis': result['content'],
            'success': True
        }
    else:
        return {
            'model': 'Claude Opus 4.6',
            'analysis': f"分析不可用: {result.get('error', '未知错误')}",
            'success': False
        }


def search_global_market(symbol):
    """全球市场搜索 - 获取最新市场资讯"""
    prompt = f"""请搜索并总结关于 {symbol} 的最新市场资讯和重要事件，包括：
1. 最近24小时的重要新闻
2. 分析师观点和评级变化
3. 行业动态影响
4. 宏观经济因素

请用中文简洁总结，不超过200字。"""

    result = call_openrouter_api(
        model='anthropic/claude-opus-4.6',
        messages=[{'role': 'user', 'content': prompt}],
        system_prompt='你是专业的市场研究分析师，擅长整合全球市场信息。',
        max_tokens=300,
        temperature=0.3
    )

    if result.get('success'):
        return result['content']
    return ""


# ============ 辅助函数 ============

def load_strategies():
    """加载存储的策略"""
    if os.path.exists(STRATEGY_FILE):
        with open(STRATEGY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def delete_strategy(strategy_id):
    """删除策略"""
    strategies = load_strategies()
    strategies = [s for s in strategies if s.get('id') != strategy_id]

    with open(STRATEGY_FILE, 'w', encoding='utf-8') as f:
        json.dump(strategies, f, ensure_ascii=False, indent=2)
    return True


def save_strategy(symbol, strategy_data):
    """保存策略"""
    strategies = load_strategies()
    strategy_data['id'] = f"{symbol}_{int(time.time())}"
    strategy_data['symbol'] = symbol  # 确保symbol被保存
    strategy_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    strategies.insert(0, strategy_data)

    # 只保留最近100条
    if len(strategies) > 100:
        strategies = strategies[:100]

    with open(STRATEGY_FILE, 'w', encoding='utf-8') as f:
        json.dump(strategies, f, ensure_ascii=False, indent=2)
    return strategy_data


def load_watchlist():
    """加载关注列表"""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']


def save_watchlist(watchlist):
    """保存关注列表"""
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)


def get_financial_statements(ticker):
    """获取完整的财务报表数据"""
    try:
        # 使用缓存避免频繁请求
        cache_key = f"financials_{ticker.ticker}"
        if cache_key in CACHE:
            cached_financials, cached_time = CACHE[cache_key]
            if time.time() - cached_time < 3600:  # 缓存1小时
                return cached_financials

        # 获取年度财务报表
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow

        # 获取季度财务报表
        income_stmt_q = ticker.quarterly_financials
        balance_sheet_q = ticker.quarterly_balance_sheet
        cash_flow_q = ticker.quarterly_cashflow

        result = {
            'annual_income': income_stmt,
            'annual_balance': balance_sheet,
            'annual_cashflow': cash_flow,
            'quarterly_income': income_stmt_q,
            'quarterly_balance': balance_sheet_q,
            'quarterly_cashflow': cash_flow_q
        }

        # 缓存财务报表数据
        CACHE[cache_key] = (result, time.time())
        return result
    except Exception as e:
        print(f"⚠️ 获取财务报表失败: {e}")
        return None


def analyze_fundamentals_advanced(symbol, data, info, ticker=None):
    """深入基本面分析 - 使用真实财务数据"""
    # 首先尝试使用ticker获取详细财务报表
    if ticker and HAS_YFINANCE:
        try:
            # 获取财务报表
            financials = get_financial_statements(ticker)
            if financials and financials['annual_income'] is not None:
                annual_income = financials['annual_income']
                annual_balance = financials['annual_balance']
                annual_cashflow = financials['annual_cashflow']

                if annual_income is not None and len(annual_income) > 0:
                    return analyze_with_financial_statements(symbol, data, info, financials)
        except Exception as e:
            print(f"⚠️ 财务报表分析失败: {e}，使用info数据分析")

    # 使用ticker.info数据进行增强分析
    return analyze_fundamentals_enhanced(symbol, data, info)


def analyze_with_financial_statements(symbol, data, info, financials):
    """使用财务报表进行深度分析"""
    annual_income = financials['annual_income']
    annual_balance = financials['annual_balance']
    annual_cashflow = financials['annual_cashflow']
    quarterly_income = financials['quarterly_income']

    details = {}
    scores = {}

    # ========== 1. 营收分析 (20分) ==========
    revenue_score = 10
    revenue_analysis = []

    try:
        if annual_income is not None and len(annual_income) > 0:
            latest_income = annual_income.iloc[:, 0]
            prev_income = annual_income.iloc[:, 1] if len(annual_income) > 1 else None

            # 总收入
            total_revenue = latest_income.get('Total Revenue', 0)
            if pd.isna(total_revenue) or total_revenue == 0:
                total_revenue = latest_income.get('Total Revenue', 0)

            # 营收增长
            revenue_growth_rate = 0
            if prev_income is not None:
                prev_revenue = prev_income.get('Total Revenue', 0)
                if prev_revenue and prev_revenue > 0:
                    revenue_growth_rate = (total_revenue - prev_revenue) / prev_revenue * 100

            # 毛利率
            cost_of_revenue = latest_income.get('Cost Of Revenue', 0)
            if total_revenue and total_revenue > 0 and cost_of_revenue:
                gross_margin = (total_revenue - cost_of_revenue) / total_revenue * 100
            else:
                gross_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0

            # 评分
            if revenue_growth_rate > 20:
                revenue_score += 8
            elif revenue_growth_rate > 10:
                revenue_score += 5
            elif revenue_growth_rate > 0:
                revenue_score += 2

            if gross_margin > 50:
                revenue_score += 2
            elif gross_margin > 30:
                revenue_score += 1

            details['revenue'] = {
                'total_revenue': float(total_revenue) if total_revenue else 0,
                'revenue_growth': round(revenue_growth_rate, 2),
                'gross_margin': round(gross_margin, 2),
                'analysis': f"营收${total_revenue/1e9:.1f}B, 增长{revenue_growth_rate:.1f}%, 毛利{gross_margin:.1f}%"
            }
        else:
            # 回退到info数据
            revenue_growth = info.get('revenueGrowth', 0) or 0
            profit_margin = info.get('profitMargins', 0) or 0
            details['revenue'] = {
                'total_revenue': 0,
                'revenue_growth': round(revenue_growth * 100, 2),
                'gross_margin': round(profit_margin * 100, 2),
                'analysis': f"营收增长{revenue_growth*100:.1f}%, 利润率{profit_margin*100:.1f}%"
            }
    except Exception as e:
        revenue_analysis.append(f"数据分析错误: {str(e)}")

    scores['revenue'] = min(max(revenue_score, 0), 20)

    # 继续其他分析...
    # 为了简化，这里使用info数据补充其他维度
    return analyze_fundamentals_enhanced(symbol, data, info, details, scores)


def analyze_fundamentals_enhanced(symbol, data, info, existing_details=None, existing_scores=None):
    """增强的基本面分析 - 使用info数据和已有数据"""
    details = existing_details or {}
    scores = existing_scores or {}

    # 如果info为空或缺少数据，尝试从缓存获取基本面数据
    if not info or not info.get('market_cap'):
        cached_fund = get_fundamental_data_cached(symbol)
        if cached_fund:
            info = cached_fund

    # ========== 盈利能力 (20分) ==========
    if 'profitability' not in scores:
        profit_score = 10
        profit_analysis = []

        eps = data.get('eps', 0)
        profit_margin = info.get('profitMargins', 0) or info.get('profit_margin', 0) or 0
        roe = info.get('returnOnEquity', 0) or info.get('roe', 0) or 0
        roa = info.get('returnOnAssets', 0) or info.get('roa', 0) or 0

        if eps > 0:
            profit_score += 4
            profit_analysis.append(f"EPS ${eps:.2f}")
        if profit_margin > 0.15:
            profit_score += 5
            profit_analysis.append(f"净利率优异 {profit_margin*100:.1f}%")
        elif profit_margin > 0.05:
            profit_score += 2
            profit_analysis.append(f"净利率 {profit_margin*100:.1f}%")
        if roe > 0.15:
            profit_score += 4
            profit_analysis.append(f"ROE {roe*100:.1f}% 优秀")
        elif roe > 0.08:
            profit_score += 2
            profit_analysis.append(f"ROE {roe*100:.1f}%")

        scores['profitability'] = min(max(profit_score, 0), 20)
        details['profitability'] = {
            'eps': round(eps, 2),
            'profit_margin': round(profit_margin * 100, 2),
            'roe': round(roe * 100, 2),
            'roa': round(roa * 100, 2),
            'analysis': '; '.join(profit_analysis) if profit_analysis else "数据不足"
        }

    # ========== 成长性 (20分) ==========
    if 'growth' not in scores:
        growth_score = 10
        growth_analysis = []

        revenue_growth = info.get('revenueGrowth', 0) or info.get('revenue_growth', 0) or 0
        earnings_growth = info.get('earningsGrowth', 0) or info.get('earnings_growth', 0) or 0
        earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0) or info.get('earnings_quarterly_growth', 0) or 0

        if revenue_growth > 0.2:
            growth_score += 6
        elif revenue_growth > 0.1:
            growth_score += 4
        elif revenue_growth > 0:
            growth_score += 2

        if earnings_growth > 0.2:
            growth_score += 4
        elif earnings_growth > 0.1:
            growth_score += 2
        elif earnings_growth > 0:
            growth_score += 1

        scores['growth'] = min(max(growth_score, 0), 20)
        details['growth'] = {
            'revenue_growth': round(revenue_growth * 100, 2),
            'earnings_growth': round(earnings_growth * 100, 2),
            'quarterly_growth': round(earnings_quarterly_growth * 100, 2),
            'analysis': f"营收增长{revenue_growth*100:.1f}%, 盈利增长{earnings_growth*100:.1f}%"
        }

    # ========== 财务健康 (20分) ==========
    if 'financial_health' not in scores:
        health_score = 10
        health_analysis = []

        debt_to_equity = info.get('debtToEquity', 0) or info.get('debt_to_equity', 0) or 0
        current_ratio = info.get('currentRatio', 0) or info.get('current_ratio', 0) or 0
        quick_ratio = info.get('quickRatio', 0) or info.get('quick_ratio', 0) or 0
        total_cash = info.get('totalCash', 0) or info.get('total_cash', 0) or 0
        total_debt = info.get('totalDebt', 0) or info.get('total_debt', 0) or 0
        total_assets = info.get('totalAssets', 0) or info.get('total_assets', 0) or 0

        if current_ratio > 1.5:
            health_score += 4
        elif current_ratio > 1:
            health_score += 2
        if debt_to_equity < 50:
            health_score += 4
        elif debt_to_equity < 100:
            health_score += 2
        if total_cash > total_debt:
            health_score += 2

        scores['financial_health'] = min(max(health_score, 0), 20)
        details['financial_health'] = {
            'debt_to_equity': round(debt_to_equity, 2),
            'current_ratio': round(current_ratio, 2),
            'quick_ratio': round(quick_ratio, 2),
            'cash': int(total_cash),
            'debt': int(total_debt),
            'assets': int(total_assets),
            'analysis': f"流动比率{current_ratio:.1f}, 负债率{debt_to_equity:.1f}%"
        }

    # ========== 估值 (20分) ==========
    if 'valuation' not in scores:
        value_score = 10
        value_analysis = []

        pe = data.get('pe_ratio', 0) or 0
        pb = info.get('priceToBook', 0) or info.get('pb_ratio', 0) or 0
        ps = info.get('priceToSalesTrailing12Months', 0) or info.get('ps_ratio', 0) or 0
        peg = info.get('pegRatio', 0) or info.get('peg_ratio', 0) or 0
        ev_to_ebitda = info.get('enterpriseToEbitda', 0) or info.get('ev_to_ebitda', 0) or 0
        forward_pe = info.get('forwardPE', 0) or info.get('forward_pe', 0) or 0

        if pe > 0 and pe < 15:
            value_score += 4
        elif pe > 0 and pe < 25:
            value_score += 2
        if peg > 0 and peg < 1:
            value_score += 4
        elif peg > 0 and peg < 1.5:
            value_score += 2
        if pb > 0 and pb < 3:
            value_score += 2
        if ps > 0 and ps < 5:
            value_score += 2

        scores['valuation'] = min(max(value_score, 0), 20)
        details['valuation'] = {
            'pe_ratio': round(pe, 2),
            'forward_pe': round(forward_pe, 2),
            'pb_ratio': round(pb, 2),
            'ps_ratio': round(ps, 2),
            'peg_ratio': round(peg, 2),
            'ev_to_ebitda': round(ev_to_ebitda, 2),
            'analysis': f"PE {pe:.1f}, PEG {peg:.1f}, PB {pb:.1f}"
        }

    # 计算总分
    total_score = sum(scores.values())

    # 生成总结
    summary = generate_fundamental_summary_advanced(details, total_score, scores)

    return {
        'total_score': round(total_score, 1),
        'scores': scores,
        'details': details,
        'summary': summary
    }


def analyze_fundamentals_basic(symbol, data, info):
    """基础基本面分析（当无法获取详细财报时使用）"""
    scores = {}
    details = {}

    # 1. 估值分析 (20分)
    pe = data.get('pe_ratio', 0)
    if pe > 0:
        if pe < 15:
            val_score = 20
            val_level = "低估"
        elif pe < 25:
            val_score = 15
            val_level = "合理"
        elif pe < 40:
            val_score = 10
            val_level = "偏高"
        else:
            val_score = 5
            val_level = "高估"
    else:
        val_score = 10
        val_level = "无数据"

    scores['valuation'] = val_score
    details['valuation'] = {
        'pe_ratio': pe,
        'level': val_level,
        'score': val_score,
        'analysis': f"PE比率 {pe:.1f}，属于{val_level}区间"
    }

    # 2. 盈利能力 (20分)
    eps = data.get('eps', 0)
    profit_margin = info.get('profitMargins', 0) or 0
    roe = info.get('returnOnEquity', 0) or 0

    profit_score = 10
    profit_analysis = []

    if eps > 0:
        profit_score += 5
        profit_analysis.append(f"EPS为正 ${eps:.2f}")
    if profit_margin > 0.15:
        profit_score += 3
        profit_analysis.append(f"净利润率{profit_margin*100:.1f}%优秀")
    elif profit_margin > 0.05:
        profit_score += 1
        profit_analysis.append(f"净利润率{profit_margin*100:.1f}%一般")
    if roe > 0.15:
        profit_score += 2
        profit_analysis.append(f"ROE {roe*100:.1f}%优秀")

    scores['profitability'] = profit_score
    details['profitability'] = {
        'eps': eps,
        'profit_margin': profit_margin * 100 if profit_margin else 0,
        'roe': roe * 100 if roe else 0,
        'score': profit_score,
        'analysis': '; '.join(profit_analysis) if profit_analysis else "数据不足"
    }

    # 3. 成长性 (20分)
    revenue_growth = info.get('revenueGrowth', 0) or 0
    earnings_growth = info.get('earningsGrowth', 0) or 0

    growth_score = 10
    growth_analysis = []

    if revenue_growth > 0.2:
        growth_score += 5
        growth_analysis.append(f"营收增长{revenue_growth*100:.1f}%强劲")
    elif revenue_growth > 0.1:
        growth_score += 3
        growth_analysis.append(f"营收增长{revenue_growth*100:.1f}%稳定")
    elif revenue_growth > 0:
        growth_score += 1
        growth_analysis.append(f"营收增长{revenue_growth*100:.1f}%缓慢")
    else:
        growth_analysis.append("营收增长停滞")

    if earnings_growth > 0.2:
        growth_score += 5
        growth_analysis.append(f"盈利增长{earnings_growth*100:.1f}%强劲")
    elif earnings_growth > 0.1:
        growth_score += 3
        growth_analysis.append(f"盈利增长{earnings_growth*100:.1f}%稳定")

    scores['growth'] = growth_score
    details['growth'] = {
        'revenue_growth': revenue_growth * 100 if revenue_growth else 0,
        'earnings_growth': earnings_growth * 100 if earnings_growth else 0,
        'score': growth_score,
        'analysis': '; '.join(growth_analysis) if growth_analysis else "数据不足"
    }

    # 4. 财务健康 (20分)
    debt_to_equity = info.get('debtToEquity', 0) or 0
    current_ratio = info.get('currentRatio', 0) or 0
    total_cash = info.get('totalCash', 0) or 0
    total_debt = info.get('totalDebt', 0) or 0

    health_score = 10
    health_analysis = []

    if debt_to_equity < 50:
        health_score += 5
        health_analysis.append("负债水平健康")
    elif debt_to_equity < 100:
        health_score += 2
        health_analysis.append("负债水平适中")
    else:
        health_analysis.append("负债水平偏高")

    if current_ratio > 1.5:
        health_score += 3
        health_analysis.append("流动性充裕")
    elif current_ratio > 1:
        health_score += 1
        health_analysis.append("流动性一般")

    if total_cash > total_debt:
        health_score += 2
        health_analysis.append("现金覆盖债务")

    scores['financial_health'] = health_score
    details['financial_health'] = {
        'debt_to_equity': debt_to_equity,
        'current_ratio': current_ratio,
        'cash': total_cash,
        'debt': total_debt,
        'score': health_score,
        'analysis': '; '.join(health_analysis) if health_analysis else "数据不足"
    }

    # 5. 分红回报 (20分)
    dividend_yield = data.get('dividend_yield', 0)
    if dividend_yield > 3:
        div_score = 20
        div_level = "高息"
    elif dividend_yield > 1.5:
        div_score = 15
        div_level = "正常"
    elif dividend_yield > 0:
        div_score = 10
        div_level = "低息"
    else:
        div_score = 5
        div_level = "无分红"

    scores['dividend'] = div_score
    details['dividend'] = {
        'yield': dividend_yield,
        'level': div_level,
        'score': div_score,
        'analysis': f"股息率{dividend_yield:.2f}%，{div_level}股票"
    }

    # 总分计算
    total_fund_score = sum(scores.values()) / 5  # 平均分

    return {
        'total_score': round(total_fund_score, 1),
        'scores': scores,
        'details': details,
        'summary': generate_fundamental_summary(details, total_fund_score)
    }


def analyze_technicals(symbol, data, hist_data=None):
    """技术面分析 - 市场情绪"""
    scores = {}
    details = {}

    price = data['price']
    change_pct = data['change_pct']
    rsi = data['rsi']
    trend = data['trend']

    # 1. 趋势分析 (25分)
    trend_score = 12
    trend_analysis = []

    if trend == 'up':
        trend_score += 8
        trend_analysis.append("上升趋势确立")
    elif trend == 'down':
        trend_score -= 5
        trend_analysis.append("下降趋势中")

    # 5日/20日均线趋势
    if hist_data is not None and hasattr(hist_data, '__len__') and len(hist_data) >= 20:
        ma5 = hist_data[-5:]['Close'].mean()
        ma20 = hist_data[-20:]['Close'].mean()
        if price > ma5 > ma20:
            trend_score += 5
            trend_analysis.append("多头排列")
        elif price < ma5 < ma20:
            trend_score -= 3
            trend_analysis.append("空头排列")

    scores['trend'] = min(max(trend_score, 0), 25)
    details['trend'] = {
        'current_trend': trend,
        'score': scores['trend'],
        'analysis': '; '.join(trend_analysis) if trend_analysis else "趋势不明"
    }

    # 2. 动量分析 (25分)
    momentum_score = 12
    momentum_analysis = []

    if abs(change_pct) > 3:
        momentum_score += 8
        momentum_analysis.append("强势动量")
    elif abs(change_pct) > 1.5:
        momentum_score += 4
        momentum_analysis.append("中等动量")
    else:
        momentum_analysis.append("动量平缓")

    if change_pct > 0:
        momentum_analysis.append("上涨动能")
    else:
        momentum_analysis.append("下跌压力")

    scores['momentum'] = min(max(momentum_score, 0), 25)
    details['momentum'] = {
        'change_pct': change_pct,
        'score': scores['momentum'],
        'analysis': '; '.join(momentum_analysis)
    }

    # 3. 超买超卖 (25分)
    rsi_score = 12
    rsi_analysis = []

    if rsi < 30:
        rsi_score += 10
        rsi_level = "超卖"
        rsi_analysis.append("RSI超卖，反弹机会")
    elif rsi < 40:
        rsi_score += 5
        rsi_level = "偏低"
        rsi_analysis.append("RSI偏低")
    elif rsi < 60:
        rsi_level = "中性"
        rsi_analysis.append("RSI中性区间")
    elif rsi < 70:
        rsi_score -= 3
        rsi_level = "偏高"
        rsi_analysis.append("RSI偏高")
    else:
        rsi_score -= 8
        rsi_level = "超买"
        rsi_analysis.append("RSI超买，回调风险")

    scores['rsi'] = min(max(rsi_score, 0), 25)
    details['rsi'] = {
        'value': rsi,
        'level': rsi_level,
        'score': scores['rsi'],
        'analysis': '; '.join(rsi_analysis)
    }

    # 4. 支撑阻力 (25分)
    support_score = 12
    resistance_score = 12
    level_analysis = []

    # 距离支撑位
    support = data['support']
    resistance = data['resistance']
    dist_to_support = ((price - support) / price * 100)
    dist_to_resistance = ((resistance - price) / price * 100)

    if dist_to_support < 2:
        support_score += 8
        level_analysis.append("接近支撑位")
    elif dist_to_support < 5:
        support_score += 4
        level_analysis.append("支撑位附近")

    if dist_to_resistance > 5:
        resistance_score += 8
        level_analysis.append("距阻力位较远")
    elif dist_to_resistance > 2:
        resistance_score += 4
        level_analysis.append("距阻力位适中")

    # 52周位置
    high_52w = data['high_52w']
    low_52w = data['low_52w']
    if high_52w > 0:
        position_52w = (price - low_52w) / (high_52w - low_52w) * 100
        if position_52w < 30:
            level_analysis.append("接近52周低位")
            support_score += 3
        elif position_52w > 70:
            level_analysis.append("接近52周高位")
            support_score -= 2

    scores['levels'] = min(max((support_score + resistance_score) / 2, 0), 25)
    details['levels'] = {
        'support': support,
        'resistance': resistance,
        'position_52w': position_52w if high_52w > 0 else 50,
        'score': scores['levels'],
        'analysis': '; '.join(level_analysis) if level_analysis else "关键点位中性"
    }

    # 总分
    total_tech_score = sum(scores.values())

    return {
        'total_score': round(total_tech_score, 1),
        'scores': scores,
        'details': details,
        'summary': generate_technical_summary(details, total_tech_score)
    }


def generate_fundamental_summary(details, score):
    """生成基本面分析总结"""
    val = details['valuation']['level']
    growth = details['growth']['score']

    if score >= 17:
        return f"基本面优秀。估值{val}，成长性强，盈利能力突出。"
    elif score >= 14:
        return f"基本面良好。估值{val}，财务状况稳健，具有投资价值。"
    elif score >= 10:
        return f"基本面一般。估值{val}，需关注成长性和盈利能力。"
    else:
        return f"基本面偏弱。估值可能{val}，财务或成长性存在隐忧。"


def generate_technical_summary(details, score):
    """生成技术面分析总结"""
    trend = details['trend']['current_trend']
    rsi = details['rsi']['level']

    if score >= 80:
        return f"技术面强势。{trend}趋势，{rsi}区域，多头占优。"
    elif score >= 60:
        return f"技术面积极。{trend}趋势，{rsi}，看涨信号。"
    elif score >= 40:
        return f"技术面中性。趋势不明，{rsi}，等待方向确认。"
    else:
        return f"技术面疲弱。{trend}趋势，{rsi}，短期承压。"


def generate_comprehensive_decision(fundamental, technical, data):
    """综合决策分析"""
    fund_score = fundamental['total_score']
    tech_score = technical['total_score']

    # 加权综合评分 (基本面50% + 技术面50%)
    comprehensive_score = (fund_score * 0.5 + tech_score * 0.5)

    # 决策矩阵
    if fund_score >= 16 and tech_score >= 70:
        action = "强力买入"
        action_en = "STRONG BUY"
        action_color = "#00C853"
        confidence = "极高"
        position = "核心仓位 15-20%"
    elif fund_score >= 14 and tech_score >= 60:
        action = "买入"
        action_en = "BUY"
        action_color = "#64DD17"
        confidence = "高"
        position = "标准仓位 10-15%"
    elif fund_score >= 12 or tech_score >= 50:
        action = "谨慎买入"
        action_en = "CAUTIOUS BUY"
        action_color = "#9CCC65"
        confidence = "中高"
        position = "轻仓 5-10%"
    elif fund_score >= 10 and tech_score >= 40:
        action = "持有"
        action_en = "HOLD"
        action_color = "#FFD600"
        confidence = "中"
        position = "维持现有仓位"
    elif fund_score < 10 or tech_score < 30:
        action = "考虑减仓"
        action_en = "REDUCE"
        action_color = "#FF6D00"
        confidence = "中高"
        position = "降低仓位至 3-5%"
    else:
        action = "观望/规避"
        action_en = "WATCH/AVOID"
        action_color = "#D50000"
        confidence = "高"
        position = "观望或空仓"

    # 价格目标
    price = data['price']
    support = data['support']
    resistance = data['resistance']

    if action in ["强力买入", "买入", "谨慎买入"]:
        entry_price = price * 0.98 if tech_score < 60 else price
        stop_loss = support * 0.95
        target_price = resistance * 1.15 if comprehensive_score > 60 else resistance * 1.08
        time_horizon = "3-6个月"
    elif action == "持有":
        entry_price = price
        stop_loss = support * 0.95
        target_price = resistance * 1.10
        time_horizon = "6-12个月"
    else:
        entry_price = support
        stop_loss = support * 0.90
        target_price = resistance
        time_horizon = "观察期"

    # 风险评估
    risk_factors = []
    risk_level = "中等"

    if fund_score < 12:
        risk_factors.append("基本面偏弱")
    if technical['details']['rsi']['value'] > 70:
        risk_factors.append("技术面超买")
    if technical['details']['rsi']['value'] < 30:
        risk_factors.append("技术面超卖反弹")
    if data.get('pe_ratio', 0) > 40:
        risk_factors.append("估值偏高风险")
    if abs(data['change_pct']) > 5:
        risk_factors.append("短期波动较大")
        risk_level = "高"

    if not risk_factors:
        risk_factors.append("短期风险可控")
        risk_level = "低"

    # 关键催化剂
    catalysts = []
    growth_rev = fundamental['details'].get('growth', {}).get('revenue_growth', 0)
    if growth_rev > 15:
        catalysts.append("强劲营收增长")

    profit_roe = fundamental['details'].get('profitability', {}).get('roe', 0)
    if profit_roe > 15:
        catalysts.append("优秀ROE表现")
    if technical['details']['trend']['score'] > 18:
        catalysts.append("上升趋势确立")
    if technical['details']['rsi']['value'] < 35:
        catalysts.append("超卖反弹机会")
    if fundamental['scores'].get('valuation', 0) >= 18:
        catalysts.append("估值优势明显")

    if not catalysts:
        catalysts.append("等待明确信号")

    return {
        'action': action,
        'action_en': action_en,
        'action_color': action_color,
        'confidence': confidence,
        'position': position,
        'comprehensive_score': round(comprehensive_score, 1),
        'fundamental_score': fund_score,
        'technical_score': tech_score,
        'entry_price': round(entry_price, 2),
        'stop_loss': round(stop_loss, 2),
        'target_price': round(target_price, 2),
        'time_horizon': time_horizon,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'catalysts': catalysts,
        'support': support,
        'resistance': resistance,
        'current_price': price
    }


def generate_fundamental_summary_advanced(details, score, scores):
    """生成深入基本面分析总结 - 包含具体数据"""
    revenue_score = scores.get('revenue', 0)
    profit_score = scores.get('profitability', 0)
    health_score = scores.get('financial_health', 0)
    growth_score = scores.get('growth', 0)
    value_score = scores.get('valuation', 0)

    summary_parts = []

    # 营收分析 - 显示具体数据
    revenue = details.get('revenue', {})
    if revenue:
        rev_growth = revenue.get('revenue_growth', 0)
        if abs(rev_growth) > 0:
            summary_parts.append(f"营收增长{rev_growth:.1f}%")
        else:
            summary_parts.append("营收稳定")

    # 盈利能力 - 显示具体数据
    profit = details.get('profitability', {})
    if profit:
        profit_margin = profit.get('profit_margin', 0)
        roe = profit.get('roe', 0)
        if profit_margin > 0:
            summary_parts.append(f"净利率{profit_margin:.1f}%")
        if roe > 0:
            summary_parts.append(f"ROE {roe:.1f}%")

    # 财务健康 - 显示具体数据
    health = details.get('financial_health', {})
    if health:
        debt_ratio = health.get('debt_to_equity', 0)
        if debt_ratio < 50:
            summary_parts.append("低负债")
        elif debt_ratio < 100:
            summary_parts.append(f"负债率{debt_ratio:.0f}%")

    # 成长性 - 显示具体数据
    growth = details.get('growth', {})
    if growth:
        rev_growth = growth.get('revenue_growth', 0)
        earn_growth = growth.get('earnings_growth', 0)
        if rev_growth != 0 or earn_growth != 0:
            growth_text = []
            if rev_growth != 0:
                growth_text.append(f"营收{rev_growth:.0f}%")
            if earn_growth != 0:
                growth_text.append(f"盈利{earn_growth:.0f}%")
            if growth_text:
                summary_parts.append("增长" + "/".join(growth_text))

    # 估值 - 显示具体数据
    valuation = details.get('valuation', {})
    if valuation:
        pe = valuation.get('pe_ratio', 0)
        if pe > 0:
            summary_parts.append(f"PE {pe:.0f}")

    # 如果没有收集到任何具体数据，使用默认描述
    if not summary_parts:
        if health_score >= 12:
            summary_parts.append("财务状况稳健")
        else:
            summary_parts.append("关注财务指标")

    # 根据总分生成评估
    if score >= 80:
        rating = "基本面优秀"
    elif score >= 60:
        rating = "基本面良好"
    elif score >= 40:
        rating = "基本面一般"
    else:
        rating = "基本面偏弱"

    return f"{rating}。{'；'.join(summary_parts)}。"


def generate_expert_analysis(symbol, data, info, hist_data=None, ticker=None):
    """生成专家级分析报告 - 使用深入基本面分析"""
    # 深入基本面分析（使用真实财报数据）
    fundamental = analyze_fundamentals_advanced(symbol, data, info, ticker)

    # 技术面分析
    technical = analyze_technicals(symbol, data, hist_data)

    # 综合决策
    decision = generate_comprehensive_decision(fundamental, technical, data)

    return {
        'fundamental': fundamental,
        'technical': technical,
        'decision': decision
    }


def generate_mock_data(symbol):
    """为任意股票代码生成逼真的模拟数据"""
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    random.seed(seed)

    base_price = random.uniform(20, 300)
    change_pct = random.uniform(-3, 3)
    change = base_price * (change_pct / 100)

    trend = 'up' if change_pct > 1 else 'down' if change_pct < -1 else 'neutral'
    rsi = random.uniform(30, 70)

    data = {
        'symbol': symbol,
        'name': f'{symbol} Corp',
        'price': round(base_price, 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 2),
        'open': round(base_price - change, 2),
        'high': round(base_price * 1.02, 2),
        'low': round(base_price * 0.98, 2),
        'volume': random.randint(5000000, 50000000),
        'market_cap': random.randint(10000000000, 2000000000000),
        'pe_ratio': random.uniform(15, 50),
        'eps': round(base_price / random.uniform(15, 50), 2),
        'dividend_yield': random.uniform(0, 2),
        'high_52w': round(base_price * 1.3, 2),
        'low_52w': round(base_price * 0.7, 2),
        'resistance': round(base_price * 1.05, 2),
        'support': round(base_price * 0.95, 2),
        'rsi': round(rsi, 1),
        'trend': trend,
        'is_mock': True,
        '_info': {}
    }

    random.seed()
    return data


def rate_limit_pause():
    """处理API限流"""
    global LAST_REQUEST_TIME
    elapsed = time.time() - LAST_REQUEST_TIME
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    LAST_REQUEST_TIME = time.time()


# ============ 路由 ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/advanced')
def advanced():
    return render_template('advanced.html')


@app.route('/watchlist')
def watchlist():
    """关注列表对比页面"""
    return render_template('watchlist.html')


@app.route('/strategies')
def strategies():
    """策略历史页面"""
    return render_template('strategies.html')


def generate_market_outlook():
    """生成市场前瞻分析 - 宏观微观 + 理性情绪 + 趋势预测"""

    # 获取市场基准数据 (使用SPY或多个大盘股的平均)
    benchmark_symbols = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
    market_data = []
    total_sentiment = 0
    total_technical = 0

    for symbol in benchmark_symbols:
        data = get_yfinance_stock_data(symbol)
        if data and 'error' not in data:
            is_mock = data.pop('is_mock', False)
            info = data.pop('_info', {})
            hist_data = data.pop('_hist_data', None)

            expert = generate_expert_analysis(symbol, data, info, hist_data)
            market_data.append({
                'symbol': symbol,
                'price': data.get('price', 0),
                'change_pct': data.get('change_pct', 0),
                'fundamental': expert.get('fundamental', {}),
                'technical': expert.get('technical', {}),
                'decision': expert.get('decision', {})
            })
            total_technical += expert.get('technical', {}).get('total_score', 50)
            total_sentiment += data.get('change_pct', 0)

    n_symbols = len(market_data)
    if n_symbols == 0:
        n_symbols = 1

    avg_technical = total_technical / n_symbols
    avg_change_pct = total_sentiment / n_symbols

    # ========== 1. 宏观经济环境分析 ==========
    macro = analyze_macro_environment(market_data)

    # ========== 2. 市场情绪分析 ==========
    sentiment = analyze_market_sentiment(market_data, avg_change_pct)

    # ========== 3. 理性分析 (基本面驱动) ==========
    rational = analyze_rational_factors(market_data, macro)

    # ========== 4. 情绪分析 (心理驱动) ==========
    emotional = analyze_emotional_factors(market_data, sentiment)

    # ========== 5. 趋势预测 ==========
    predictions = generate_trend_predictions(market_data, rational, emotional, sentiment)

    # ========== 6. 综合建议 ==========
    recommendation = generate_comprehensive_recommendation(rational, emotional, predictions)

    return {
        'macro': macro,
        'sentiment': sentiment,
        'rational': rational,
        'emotional': emotional,
        'predictions': predictions,
        'recommendation': recommendation,
        'market_data_summary': {
            'symbols_analyzed': n_symbols,
            'avg_technical_score': round(avg_technical, 1),
            'avg_change_pct': round(avg_change_pct, 2)
        }
    }


def analyze_macro_environment(market_data):
    """分析宏观经济环境"""

    # 基于市场表现推断宏观环境
    avg_change = sum(d['change_pct'] for d in market_data) / len(market_data) if market_data else 0
    avg_fundamental = sum(d['fundamental'].get('total_score', 50) for d in market_data) / len(market_data) if market_data else 50

    # 利率环境推断
    if avg_fundamental > 60 and avg_change > 1:
        interest_rate = "宽松"
        interest_rate_impact = "positive"
    elif avg_fundamental < 40 or avg_change < -1:
        interest_rate = "紧缩"
        interest_rate_impact = "negative"
    else:
        interest_rate = "中性"
        interest_rate_impact = "neutral"

    # 通胀水平推断
    positive_momentum = sum(1 for d in market_data if d['change_pct'] > 0)
    if positive_momentum > len(market_data) * 0.7:
        inflation = "温和上涨"
        inflation_impact = "neutral"
    elif positive_momentum < len(market_data) * 0.3:
        inflation = "压力较大"
        inflation_impact = "negative"
    else:
        inflation = "稳定"
        inflation_impact = "neutral"

    # GDP增长推断
    high_growth_stocks = sum(1 for d in market_data if d['fundamental'].get('details', {}).get('growth', {}).get('revenue_growth', 0) > 10)
    if high_growth_stocks > len(market_data) * 0.5:
        gdp = "稳健增长"
        gdp_impact = "positive"
    elif high_growth_stocks < len(market_data) * 0.2:
        gdp = "增长放缓"
        gdp_impact = "negative"
    else:
        gdp = "正常水平"
        gdp_impact = "neutral"

    # 失业率 (通过公司盈利能力间接反映)
    avg_profitability = sum(d['fundamental'].get('scores', {}).get('profitability', 50) for d in market_data) / len(market_data) if market_data else 50
    if avg_profitability > 60:
        unemployment = "低就业"
        unemployment_impact = "positive"
    elif avg_profitability < 40:
        unemployment = "就业压力"
        unemployment_impact = "negative"
    else:
        unemployment = "稳定"
        unemployment_impact = "neutral"

    # 政策环境
    if avg_change > 2:
        policy = "积极支持"
        policy_impact = "positive"
    elif avg_change < -2:
        policy = "收紧调控"
        policy_impact = "negative"
    else:
        policy = "中性观望"
        policy_impact = "neutral"

    # 地缘政治
    volatility = sum(abs(d['change_pct']) for d in market_data) / len(market_data) if market_data else 0
    if volatility > 3:
        geopolitical = "波动较大"
        geopolitical_impact = "negative"
    elif volatility < 1:
        geopolitical = "相对稳定"
        geopolitical_impact = "positive"
    else:
        geopolitical = "正常"
        geopolitical_impact = "neutral"

    return {
        'interest_rate': interest_rate,
        'interest_rate_impact': interest_rate_impact,
        'inflation': inflation,
        'inflation_impact': inflation_impact,
        'gdp': gdp,
        'gdp_impact': gdp_impact,
        'unemployment': unemployment,
        'unemployment_impact': unemployment_impact,
        'policy': policy,
        'policy_impact': policy_impact,
        'geopolitical': geopolitical,
        'geopolitical_impact': geopolitical_impact
    }


def analyze_market_sentiment(market_data, avg_change_pct):
    """分析市场情绪"""

    # VIX恐慌指数 (通过波动率推断)
    volatility = sum(abs(d['change_pct']) for d in market_data) / len(market_data) if market_data else 0
    if volatility > 3:
        vix = "高恐慌"
        vix_impact = "negative"
    elif volatility < 1:
        vix = "低恐慌"
        vix_impact = "positive"
    else:
        vix = "正常"
        vix_impact = "neutral"

    # Put/Call比率 (通过涨跌分布推断)
    gainers = sum(1 for d in market_data if d['change_pct'] > 0)
    ratio = len(market_data) - gainers if gainers < len(market_data) else gainers
    put_call_ratio = f"{ratio / max(len(market_data), 1):.1f}"
    if gainers > len(market_data) * 0.6:
        put_call_impact = "positive"
    elif gainers < len(market_data) * 0.4:
        put_call_impact = "negative"
    else:
        put_call_impact = "neutral"

    # 资金流向
    if avg_change_pct > 1:
        flow = "净流入"
        flow_impact = "positive"
    elif avg_change_pct < -1:
        flow = "净流出"
        flow_impact = "negative"
    else:
        flow = "平衡"
        flow_impact = "neutral"

    # 市场广度
    strong_stocks = sum(1 for d in market_data if d['change_pct'] > 1)
    breadth_ratio = strong_stocks / len(market_data) if market_data else 0
    if breadth_ratio > 0.5:
        breadth = "良好"
        breadth_impact = "positive"
    elif breadth_ratio < 0.2:
        breadth = "疲弱"
        breadth_impact = "negative"
    else:
        breadth = "一般"
        breadth_impact = "neutral"

    return {
        'vix': vix,
        'vix_impact': vix_impact,
        'put_call_ratio': put_call_ratio,
        'put_call_impact': put_call_impact,
        'flow': flow,
        'flow_impact': flow_impact,
        'breadth': breadth,
        'breadth_impact': breadth_impact
    }


def analyze_rational_factors(market_data, macro):
    """理性分析 - 基于基本面和宏观"""

    factors = []
    scores = []

    # 1. 宏观经济评分
    positive_macros = sum([
        1 if macro.get('interest_rate_impact') == 'positive' else 0,
        1 if macro.get('inflation_impact') == 'positive' else 0,
        1 if macro.get('gdp_impact') == 'positive' else 0,
        1 if macro.get('policy_impact') == 'positive' else 0
    ])
    macro_score = positive_macros * 25  # 0-100
    scores.append({'name': '宏观环境', 'value': f"{positive_macros}/4 积极因素"})
    factors.append(f"宏观环境: {['极度不利', '不利', '中性偏空', '中性偏多', '有利', '极度有利'][positive_macros]}")

    # 2. 企业盈利评分
    avg_profit = sum(d['fundamental'].get('scores', {}).get('profitability', 50) for d in market_data) / len(market_data) if market_data else 50
    scores.append({'name': '企业盈利', 'value': f"{avg_profit:.0f}/100"})
    factors.append(f"企业盈利能力: {'强劲' if avg_profit > 60 else '一般' if avg_profit > 40 else '偏弱'}")

    # 3. 估值水平
    avg_valuation = sum(d['fundamental'].get('scores', {}).get('valuation', 50) for d in market_data) / len(market_data) if market_data else 50
    scores.append({'name': '估值水平', 'value': f"{avg_valuation:.0f}/100"})
    factors.append(f"估值吸引力: {'高' if avg_valuation > 60 else '中等' if avg_valuation > 40 else '低'}")

    # 4. 成长性
    avg_growth = sum(d['fundamental'].get('scores', {}).get('growth', 50) for d in market_data) / len(market_data) if market_data else 50
    scores.append({'name': '成长动能', 'value': f"{avg_growth:.0f}/100"})
    factors.append(f"成长动能: {'强劲' if avg_growth > 60 else '稳定' if avg_growth > 40 else '放缓'}")

    overall_score = (macro_score + avg_profit + avg_valuation + avg_growth) / 4

    return {
        'score': round(overall_score, 0),
        'factors': scores,
        'analysis': factors
    }


def analyze_emotional_factors(market_data, sentiment):
    """情绪分析 - 基于市场心理"""

    factors = []
    scores = []

    # 1. 恐慌/贪婪指数
    fear_greed = 50  # 基准
    if sentiment.get('vix_impact') == 'positive':
        fear_greed += 15
    elif sentiment.get('vix_impact') == 'negative':
        fear_greed -= 15
    if sentiment.get('flow_impact') == 'positive':
        fear_greed += 10
    elif sentiment.get('flow_impact') == 'negative':
        fear_greed -= 10

    fear_greed = max(0, min(100, fear_greed))
    scores.append({'name': '市场情绪', 'value': f"{fear_greed:.0f}/100 {'贪婪' if fear_greed > 60 else '恐惧' if fear_greed < 40 else '中性'}"})
    factors.append(f"投资者情绪: {'贪婪' if fear_greed > 60 else '恐惧' if fear_greed < 40 else '中性'}")

    # 2. 动量指标
    avg_momentum = sum(d['technical'].get('scores', {}).get('momentum', 50) for d in market_data) / len(market_data) if market_data else 50
    scores.append({'name': '价格动量', 'value': f"{avg_momentum:.0f}/100"})
    factors.append(f"价格动量: {'强劲' if avg_momentum > 60 else '中性' if avg_momentum > 40 else '疲弱'}")

    # 3. 趋势强度
    avg_trend = sum(d['technical'].get('scores', {}).get('trend', 50) for d in market_data) / len(market_data) if market_data else 50
    scores.append({'name': '趋势强度', 'value': f"{avg_trend:.0f}/100"})
    factors.append(f"趋势强度: {'上升' if avg_trend > 60 else '震荡' if avg_trend > 40 else '下降'}")

    # 4. 资金态度
    flow_sentiment = 50
    if sentiment.get('breadth_impact') == 'positive':
        flow_sentiment += 20
    elif sentiment.get('breadth_impact') == 'negative':
        flow_sentiment -= 20
    if sentiment.get('put_call_impact') == 'positive':
        flow_sentiment += 15
    elif sentiment.get('put_call_impact') == 'negative':
        flow_sentiment -= 15

    flow_sentiment = max(0, min(100, flow_sentiment))
    scores.append({'name': '资金态度', 'value': f"{flow_sentiment:.0f}/100"})
    factors.append(f"资金态度: {'积极' if flow_sentiment > 60 else '谨慎' if flow_sentiment > 40 else '消极'}")

    overall_score = (fear_greed + avg_momentum + avg_trend + flow_sentiment) / 4

    return {
        'score': round(overall_score, 0),
        'factors': scores,
        'analysis': factors
    }


def generate_trend_predictions(market_data, rational, emotional, sentiment):
    """生成趋势预测 - 1天、3天、1周"""

    # 综合评分
    rational_weight = 0.4
    emotional_weight = 0.6  # 短期情绪影响更大

    composite_score = (rational['score'] * rational_weight + emotional['score'] * emotional_weight)

    # ========== 1天预测 ==========
    one_day_trend, one_day_confidence, one_day_factors = predict_short_term(
        market_data, emotional, sentiment, '1d'
    )

    # ========== 3天预测 ==========
    three_days_trend, three_days_confidence, three_days_factors = predict_short_term(
        market_data, emotional, sentiment, '3d'
    )

    # ========== 1周预测 ==========
    one_week_trend, one_week_confidence, one_week_factors = predict_medium_term(
        market_data, rational, emotional
    )

    return {
        'one_day': {
            'trend': one_day_trend,
            'confidence': one_day_confidence,
            'factors': one_day_factors
        },
        'three_days': {
            'trend': three_days_trend,
            'confidence': three_days_confidence,
            'factors': three_days_factors
        },
        'one_week': {
            'trend': one_week_trend,
            'confidence': one_week_confidence,
            'factors': one_week_factors
        }
    }


def predict_short_term(market_data, emotional, sentiment, timeframe):
    """短期预测 (1-3天) - 主要基于情绪"""

    # 短期主要看情绪和动量
    avg_momentum = sum(d['technical'].get('scores', {}).get('momentum', 50) for d in market_data) / len(market_data) if market_data else 50
    avg_trend = sum(d['technical'].get('scores', {}).get('trend', 50) for d in market_data) / len(market_data) if market_data else 50

    # 短期反转信号 (RSI超买超卖)
    avg_rsi = sum(d.get('rsi', 50) for d in market_data) / len(market_data) if market_data else 50

    score = (emotional['score'] + avg_momentum + avg_trend) / 3

    # RSI调整
    if avg_rsi > 70:
        score -= 15  # 超买，短期回调概率大
    elif avg_rsi < 30:
        score += 15  # 超卖，短期反弹概率大

    factors = []
    if avg_momentum > 60:
        factors.append("短期动量强劲")
    elif avg_momentum < 40:
        factors.append("短期动量疲弱")

    if avg_trend > 60:
        factors.append("上升趋势确立")
    elif avg_trend < 40:
        factors.append("下降趋势中")

    if avg_rsi > 70:
        factors.append("技术超买，警惕回调")
    elif avg_rsi < 30:
        factors.append("技术超卖，可能反弹")

    if sentiment.get('flow_impact') == 'positive':
        factors.append("资金净流入支撑")
    elif sentiment.get('flow_impact') == 'negative':
        factors.append("资金流出压力")

    if score > 60:
        trend = 'up'
        confidence = '高' if score > 70 else '中高'
    elif score < 40:
        trend = 'down'
        confidence = '高' if score < 30 else '中高'
    else:
        trend = 'neutral'
        confidence = '中等'

    # 确保至少有3个因素
    while len(factors) < 3:
        factors.append("市场震荡整理")

    return trend, confidence, factors[:4]


def predict_medium_term(market_data, rational, emotional):
    """中期预测 (1周) - 综合理性和情绪"""

    # 中期看基本面和趋势结合
    score = (rational['score'] * 0.5 + emotional['score'] * 0.5)

    # 加入趋势延续性
    avg_trend = sum(d['technical'].get('scores', {}).get('trend', 50) for d in market_data) / len(market_data) if market_data else 50

    factors = []

    # 基本面因素
    if rational['score'] > 60:
        factors.append("基本面支撑强劲")
    elif rational['score'] < 40:
        factors.append("基本面压力较大")

    # 情绪因素
    if emotional['score'] > 60:
        factors.append("市场情绪积极")
    elif emotional['score'] < 40:
        factors.append("市场情绪悲观")

    # 趋势因素
    if avg_trend > 60:
        factors.append("中期趋势向上")
    elif avg_trend < 40:
        factors.append("中期趋势向下")

    # 波动性考虑
    volatility = sum(abs(d['change_pct']) for d in market_data) / len(market_data) if market_data else 0
    if volatility < 1.5:
        factors.append("市场波动较小，趋势稳定")
    elif volatility > 3:
        factors.append("市场波动较大，需谨慎")

    score = (score + avg_trend) / 2

    if score > 60:
        trend = 'up'
        confidence = '高' if score > 70 else '中高'
    elif score < 40:
        trend = 'down'
        confidence = '高' if score < 30 else '中高'
    else:
        trend = 'neutral'
        confidence = '中等'

    # 确保至少有3个因素
    while len(factors) < 3:
        factors.append("关注市场变化")

    return trend, confidence, factors[:4]


def generate_comprehensive_recommendation(rational, emotional, predictions):
    """生成综合操作建议"""

    week_trend = predictions['one_week']['trend']
    rational_score = rational['score']
    emotional_score = emotional['score']

    if week_trend == 'up' and rational_score >= 50:
        if emotional_score > 60:
            return "市场环境向好，建议积极布局优质标的，控制仓位在70%左右，分批买入。"
        else:
            return "基本面支撑较强，可逢低吸纳，关注回调机会，建议仓位50-60%。"
    elif week_trend == 'up' and rational_score < 50:
        return "技术面反弹但基本面偏弱，建议短线参与，快进快出，仓位控制在30%以内。"
    elif week_trend == 'down' and rational_score >= 50:
        return "基本面良好但短期调整，建议耐心等待更好入场点，可考虑定投方式分批建仓。"
    elif week_trend == 'down' and rational_score < 50:
        return "基本面和技术面均偏弱，建议以观望为主，控制仓位或适度对冲风险。"
    else:
        return "市场方向不明，建议保持中性仓位，关注结构性机会，做好风险管理。"


@app.route('/ai_stocks')
def ai_stocks():
    """AI领域投资分析页面"""
    return render_template('ai_stocks.html')


@app.route('/market_outlook')
def market_outlook_page():
    """市场前瞻分析页面"""
    return render_template('market_outlook.html')


@app.route('/api/market_outlook')
def get_market_outlook():
    """API: 获取市场前瞻分析"""
    outlook = generate_market_outlook()
    return jsonify({
        'success': True,
        'outlook': outlook
    })


@app.route('/debug')
def debug():
    """调试页面"""
    return render_template('debug.html')


@app.route('/test')
def test():
    """简单测试页面"""
    return render_template('test_simple.html')


@app.route('/get_all_data', methods=['POST'])
def get_all_data():
    """获取股票数据 + 双体系深入分析"""
    symbol = request.json.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': '请输入股票代码'})

    # 获取股票数据
    data = get_yfinance_stock_data(symbol)

    if data is None:
        return jsonify({'error': f'无法获取 {symbol} 的数据'})

    is_mock = data.pop('is_mock', False)
    info = data.pop('_info', {})
    hist_data = data.pop('_hist_data', None)

    # 创建ticker对象用于深入基本面分析
    ticker = None
    if HAS_YFINANCE and not is_mock:
        try:
            ticker = yf.Ticker(symbol)
        except:
            pass

    # 生成双体系深入分析（真实财报数据 + 技术面）
    expert_analysis = generate_expert_analysis(symbol, data, info, hist_data, ticker)

    return jsonify({
        'success': True,
        'data': data,
        'expert': expert_analysis,
        'is_mock_data': is_mock
    })


@app.route('/ai_analysis', methods=['POST'])
def ai_analysis():
    """AI 分析 - 基于双体系分析生成详细报告"""
    symbol = request.json.get('symbol', '').upper()
    analysis_type = request.json.get('type', 'overview')  # overview, technical, news, target

    if not symbol:
        return jsonify({'error': '请输入股票代码'})

    # 获取股票数据
    data = get_yfinance_stock_data(symbol)

    if data is None:
        return jsonify({'error': f'无法获取 {symbol} 的数据'})

    is_mock = data.pop('is_mock', False)
    info = data.pop('_info', {})
    hist_data = data.pop('_hist_data', None)

    # 生成双体系分析
    expert_analysis = generate_expert_analysis(symbol, data, info, hist_data)

    # 根据类型生成不同的分析报告
    analysis_text = format_ai_analysis(symbol, data, expert_analysis, analysis_type)

    return jsonify({
        'success': True,
        'analysis': analysis_text,
        'is_mock_data': is_mock
    })


def format_ai_analysis(symbol, data, expert, analysis_type):
    """格式化AI分析报告为可读文本"""
    decision = expert.get('decision', {})
    fundamental = expert.get('fundamental', {})
    technical = expert.get('technical', {})

    # 计算收益数据
    current_price = data.get('price', 0)
    entry_price = decision.get('entry_price', current_price)
    target_price = decision.get('target_price', current_price)
    stop_loss = decision.get('stop_loss', current_price)

    target_return = ((target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

    # 计算风险收益比
    if entry_price > stop_loss and entry_price > 0:
        risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
    else:
        risk_reward = 0

    if analysis_type == 'overview':
        return f"""📊 {symbol} 全面分析报告

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 投资建议: {decision.get('action', 'N/A')}
📈 综合评分: {decision.get('comprehensive_score', 0)}/100
💪 基本面: {fundamental.get('total_score', 0)}/100
📊 技术面: {technical.get('total_score', 0)}/100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 基本信息
  当前价格: ${data.get('price', 0):.2f}
  日涨跌: {data.get('change', 0):+.2f} ({data.get('change_pct', 0):+.2f}%)
  市值: {data.get('market_cap', 0)/1e9:.1f}B
  成交量: {data.get('volume', 0)/1e6:.1f}M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 基本面分析
{fundamental.get('summary', '暂无分析')}

{format_fundamental_details(fundamental.get('details', {}))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 技术面分析
{technical.get('summary', '暂无分析')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 交易建议
  入场价位: ${decision.get('entry_price', 0):.2f}
  目标价位: ${decision.get('target_price', 0):.2f}
  止损价位: ${decision.get('stop_loss', 0):.2f}
  持仓周期: {decision.get('time_horizon', 'N/A')}
  建议仓位: {decision.get('position', 'N/A')}

风险等级: {decision.get('risk_level', 'N/A')}
信心指数: {decision.get('confidence', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 催化剂因素
{format_catalysts(decision.get('catalysts', []))}
"""

    elif analysis_type == 'technical':
        tech_details = technical.get('details', {})
        return f"""📈 {symbol} 技术面深度分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 技术面评分: {technical.get('total_score', 0)}/100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 趋势分析
  {tech_details.get('trend', {}).get('analysis', '暂无分析')}
  评分: {tech_details.get('trend', {}).get('score', 0)}/25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ 动量分析
  {tech_details.get('momentum', {}).get('analysis', '暂无分析')}
  评分: {tech_details.get('momentum', {}).get('score', 0)}/25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📉 RSI指标
  {tech_details.get('rsi', {}).get('analysis', '暂无分析')}
  RSI值: {tech_details.get('rsi', {}).get('value', 'N/A')}
  评分: {tech_details.get('rsi', {}).get('score', 0)}/25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 关键价位
  {tech_details.get('levels', {}).get('analysis', '暂无分析')}
  支撑位: ${decision.get('support', 0):.2f}
  阻力位: ${decision.get('resistance', 0):.2f}
  评分: {tech_details.get('levels', {}).get('score', 0)}/25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
技术面总结: {technical.get('summary', '')}
"""

    elif analysis_type == 'news':
        # 计算距离百分比
        dist_to_entry = ((current_price - entry_price) / current_price * 100) if current_price > 0 else 0
        dist_to_target = ((current_price - target_price) / current_price * 100) if current_price > 0 else 0

        return f"""📰 {symbol} 市场动态

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ 近期催化剂
{format_catalysts(decision.get('catalysts', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 风险因素
{format_risk_factors(decision.get('risk_factors', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 技术信号
  趋势状态: {technical.get('details', {}).get('trend', {}).get('current_trend', 'N/A').upper()}
  RSI水平: {technical.get('details', {}).get('rsi', {}).get('level', 'N/A')}
  动量状态: {technical.get('details', {}).get('momentum', {}).get('analysis', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 操作建议
  当前价格: ${current_price:.2f}
  距离入场价: {dist_to_entry:+.1f}%
  距离目标价: {dist_to_target:+.1f}%

建议操作: {decision.get('action', 'N/A')}
"""

    elif analysis_type == 'target':
        template = """🎯 {symbol} 目标价分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 价格目标

  当前价格: ${current_price:.2f}
  ────────────────────────────────
  入场价位: ${entry_price:.2f}
  目标价位: ${target_price:.2f}
  止损价位: ${stop_loss:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 潜在收益

  目标收益: {target_return:+.1f}%
  风险收益比: 1:{risk_reward:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏱️ 时间框架

  建议持仓周期: {time_horizon}
  建议仓位: {position}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 分析依据

  综合评分: {comprehensive_score}/100
  基本面评分: {fundamental_score}/100
  技术面评分: {technical_score}/100

  信心指数: {confidence}
  风险等级: {risk_level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 操作建议
{reasoning}
"""
        return template.format(
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            target_return=target_return,
            risk_reward=risk_reward,
            time_horizon=decision.get('time_horizon', 'N/A'),
            position=decision.get('position', 'N/A'),
            comprehensive_score=decision.get('comprehensive_score', 0),
            fundamental_score=fundamental.get('total_score', 0),
            technical_score=technical.get('total_score', 0),
            confidence=decision.get('confidence', 'N/A'),
            risk_level=decision.get('risk_level', 'N/A'),
            reasoning=decision.get('reasoning', '基于当前技术面和基本面综合分析，建议' + decision.get('action', '谨慎操作'))
        )

    return "暂无此类型分析"


def format_catalysts(catalysts):
    if not catalysts:
        return "  暂无明确催化剂"
    return "\n".join([f"  • {c}" for c in catalysts])


def format_risk_factors(risks):
    if not risks:
        return "  暂无特定风险"
    return "\n".join([f"  • {r}" for r in risks])


def format_fundamental_details(details):
    """格式化基本面详细数据"""
    lines = []

    # 盈利能力
    profit = details.get('profitability', {})
    if profit and profit.get('analysis') and profit['analysis'] != '数据不足':
        lines.append(f"  盈利: {profit.get('analysis', '')}")

    # 成长性
    growth = details.get('growth', {})
    if growth and growth.get('analysis'):
        lines.append(f"  成长: {growth.get('analysis', '')}")

    # 财务健康
    health = details.get('financial_health', {})
    if health and health.get('analysis'):
        lines.append(f"  财务: {health.get('analysis', '')}")

    # 估值
    valuation = details.get('valuation', {})
    if valuation and valuation.get('analysis'):
        lines.append(f"  估值: {valuation.get('analysis', '')}")

    if lines:
        return "\n".join(lines)
    return "  详细数据获取中..."


@app.route('/get_watchlist_data', methods=['POST'])
def get_watchlist_data():
    """获取关注列表的对比数据"""
    watchlist = load_watchlist()

    results = []
    for symbol in watchlist:
        data = get_yfinance_stock_data(symbol)
        if data and 'error' not in data:
            is_mock = data.pop('is_mock', False)
            info = data.pop('_info', {})
            hist_data = data.pop('_hist_data', None)

            expert = generate_expert_analysis(symbol, data, info, hist_data)

            # 获取各个分析模块的数据
            decision = expert.get('decision', {})
            fundamental = expert.get('fundamental', {})
            technical = expert.get('technical', {})

            # 操作建议映射
            action = decision.get('action', 'N/A')
            action_en_map = {
                '强烈买入': 'STRONG BUY',
                '买入': 'BUY',
                '谨慎买入': 'CAUTIOUS BUY',
                '持有': 'HOLD',
                '减持': 'REDUCE',
                '规避': 'AVOID'
            }
            action_en = action_en_map.get(action, 'N/A')

            # 操作颜色映射
            action_color_map = {
                '强烈买入': '#00c853',
                '买入': '#64dd17',
                '谨慎买入': '#ffc107',
                '持有': '#9e9e9e',
                '减持': '#ff9800',
                '规避': '#f44336'
            }
            action_color = action_color_map.get(action, '#888')

            result = {
                'symbol': symbol,
                'name': data.get('name', f'{symbol} Corp'),
                'current_price': data.get('price', 0),
                'action': action,
                'action_en': action_en,
                'action_color': action_color,
                'total_score': decision.get('comprehensive_score', 0),
                'tech_score': technical.get('total_score', 0),
                'fund_score': fundamental.get('total_score', 0),
                'entry_price': decision.get('entry_price', data.get('price', 0)),
                'stop_loss': decision.get('stop_loss', 0),
                'target_short': decision.get('target_price', 0),
                'target_medium': decision.get('target_price', 0),
                'target_long': decision.get('target_price', 0),
                'position': decision.get('position_suggestion', 'N/A'),
                'entry_msg': decision.get('reasoning', decision.get('risk_level', 'N/A')),
                'confidence': decision.get('confidence', 'N/A'),
                'is_mock': is_mock,
                # 保留完整的expert数据
                'expert': expert
            }
            results.append(result)

    # 按评分排序
    results.sort(key=lambda x: x['total_score'], reverse=True)

    return jsonify({
        'success': True,
        'watchlist': watchlist,
        'stocks': results
    })


@app.route('/update_watchlist', methods=['POST'])
def update_watchlist():
    """更新关注列表"""
    action = request.json.get('action', '')
    symbols = request.json.get('symbols', [])

    if action == 'add':
        watchlist = load_watchlist()
        for symbol in symbols:
            s = symbol.upper()
            if s not in watchlist:
                watchlist.append(s)
        save_watchlist(watchlist)
    elif action == 'remove':
        watchlist = load_watchlist()
        for symbol in symbols:
            s = symbol.upper()
            if s in watchlist:
                watchlist.remove(s)
        save_watchlist(watchlist)
    elif action == 'set':
        save_watchlist([s.upper() for s in symbols])

    return jsonify({
        'success': True,
        'watchlist': load_watchlist()
    })


@app.route('/get_watchlist', methods=['GET'])
def get_watchlist():
    """获取关注列表"""
    return jsonify({
        'success': True,
        'watchlist': load_watchlist()
    })


@app.route('/save_strategy', methods=['POST'])
def save_strategy_route():
    """保存策略"""
    symbol = request.json.get('symbol', '').upper()
    strategy_data = request.json.get('strategy', {})

    if not symbol or not strategy_data:
        return jsonify({'error': '参数不完整'})

    saved = save_strategy(symbol, strategy_data)

    return jsonify({
        'success': True,
        'strategy': saved
    })


@app.route('/get_strategies', methods=['GET'])
def get_strategies_route():
    """获取所有保存的策略"""
    strategies = load_strategies()

    # 按符号分组
    by_symbol = {}
    for s in strategies:
        # 跳过没有symbol字段的旧数据
        if 'symbol' not in s:
            continue
        sym = s['symbol']
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(s)

    return jsonify({
        'success': True,
        'strategies': strategies,
        'by_symbol': by_symbol
    })


@app.route('/delete_strategy', methods=['POST'])
def delete_strategy_route():
    """删除策略"""
    strategy_id = request.json.get('id', '')

    if not strategy_id:
        return jsonify({'error': '缺少策略ID'})

    delete_strategy(strategy_id)

    return jsonify({'success': True})


@app.route('/get_ai_stocks_data', methods=['POST'])
def get_ai_stocks_data():
    """获取AI概念股数据"""
    symbols = request.json.get('symbols', [])

    if not symbols:
        return jsonify({'error': '缺少股票代码列表'})

    results = []
    for symbol in symbols:
        data = get_yfinance_stock_data(symbol)
        if data and 'error' not in data:
            is_mock = data.pop('is_mock', False)
            info = data.pop('_info', {})
            results.append({
                'symbol': symbol,
                'name': data.get('name', f'{symbol} Corp'),
                'price': data.get('price', 0),
                'change': data.get('change', 0),
                'change_pct': data.get('change_pct', 0),
                'volume': data.get('volume', 0),
                'market_cap': data.get('market_cap', 0),
                'pe_ratio': data.get('pe_ratio', 0),
                'rsi': data.get('rsi', 50),
                'trend': data.get('trend', 'neutral'),
                'is_mock': is_mock
            })

    return jsonify({
        'success': True,
        'stocks': results
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 美股AI分析系统 - 专业版")
    print("   集成 OpenRouter API + Gemini Pro + Claude Opus 4.6")
    print("=" * 70)

    # 检查API密钥配置
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        print("⚠️  警告: OpenRouter API Key 未配置!")
        print("   AI分析功能将使用模拟数据")
        print("   请设置环境变量: export OPENROUTER_API_KEY=your_key")
        print("   或在代码中配置 API_KEY")
        print("")
    else:
        print("✓ OpenRouter API 已配置")
        print("")

    print("正在启动服务器...")
    print("  http://localhost:5000/          - 主页")
    print("  http://localhost:5000/advanced  - 高级分析")
    print("  http://localhost:5000/watchlist - 关注列表")
    print("  http://localhost:5000/strategies - 策略历史")
    print("  http://localhost:5000/ai_stocks - AI领域投资")
    print("=" * 70)

    app.run(host='127.0.0.1', port=5000, debug=False)
