# 股票数据API配置指南

## 📌 当前状态
系统正在使用**模拟数据**运行。要获取真实股票数据，请配置以下任一API。

---

## 🔑 推荐方案：Finnhub API（最简单）

### 为什么选择Finnhub？
- ✅ **完全免费**，无需信用卡
- ✅ **60次/分钟** 请求限制，足够个人使用
- ✅ **数据准确**，涵盖美股、港股、A股
- ✅ **API稳定**，不会频繁限流

### 注册步骤：
1. 访问：https://finnhub.io/register
2. 填写邮箱和密码（无需信用卡）
3. 登录后获取 API Key
4. 复制您的 API Key

### 配置方法：
```bash
# 方法1：环境变量（推荐）
export FINNHUB_API_KEY="你的API密钥"
python3 stock_api_server.py

# 方法2：在代码中配置
# 编辑 stock_api_server.py 第38行：
FINNHUB_API_KEY = "你的API密钥"
```

---

## 🔄 备用方案：yfinance

### 特点：
- ✅ 无需注册
- ⚠️ 容易被限流
- ⚠️ 数据获取不稳定

### 使用方法：
```bash
pip install yfinance
```

系统已自动集成，无需额外配置。

---

## 📊 其他API选择

### TwelveData
- 注册：https://twelvedata.com/
- 免费额度：800次/天
- 配置：`export TWELVEDATA_API_KEY="你的密钥"`

### Polygon.io
- 注册：https://polygon.io/
- 免费额度：有限但数据质量高
- 配置：`export POLYGON_API_KEY="你的密钥"`

---

## 🚀 快速开始

### 1. 获取Finnhub API密钥（2分钟）
```bash
# 访问 https://finnhub.io/register
# 注册后复制API Key
```

### 2. 设置环境变量
```bash
export FINNHUB_API_KEY="你的API密钥"
```

### 3. 启动服务器
```bash
python3 stock_api_server.py
```

### 4. 访问系统
- 主页：http://localhost:5000
- 关注列表：http://localhost:5000/watchlist
- 市场前瞻：http://localhost:5000/market_outlook

---

## 💡 系统特点

### 多数据源自动切换
1. **优先使用** Finnhub API（稳定）
2. **自动降级** yfinance（备用）
3. **最终兜底** 模拟数据（保证可用）

### 智能缓存
- 10分钟数据缓存
- 减少API调用
- 提升响应速度

### 错误处理
- API失败自动切换数据源
- 使用模拟数据保证系统可用
- 清晰标注数据来源

---

## ❓ 常见问题

**Q: 不配置API能使用吗？**
A: 可以！系统会使用模拟数据，功能完整可用。

**Q: API会超限吗？**
A: Finnhub免费版60次/分钟，个人使用足够。

**Q: 数据准确吗？**
A: Finnhub使用官方数据源，准确可靠。

**Q: 需要付费吗？**
A: Finnhub个人使用完全免费。

---

## 📞 技术支持

如遇问题，请检查：
1. API密钥是否正确配置
2. 网络连接是否正常
3. 服务器日志（server.log）
