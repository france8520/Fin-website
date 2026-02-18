"""
Risk Analysis Module — Enhanced AI Engine
Advanced stock risk analysis with Beta, Sortino, CVAR, RSI, trend analysis, 
AI risk scoring, and recommendation engine.
"""

import yfinance as yf
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class RiskMetrics:
    """Data class storing comprehensive risk analysis results"""
    ticker: str
    current_price: float
    volatility: float
    var_95: float
    var_99: float
    max_drawdown: float
    sharpe_ratio: float
    risk_level: str
    risk_color: str
    # New advanced metrics
    beta: float
    sortino_ratio: float
    cvar_95: float
    rsi: float
    trend: str
    sma_50: float
    sma_200: float
    risk_score: int  # 0-100 composite score
    recommendation: str  # "Strong Buy" to "Strong Sell"
    recommendation_reason: str


class StockRiskAnalyzer:
    """Main class for comprehensive stock risk analysis"""

    def __init__(self):
        self.risk_thresholds = {
            'high': 0.4,
            'medium': 0.2
        }

    def analyze_stock(self, ticker: str, period: str = "1y") -> Optional[RiskMetrics]:
        """
        Analyze stock risk metrics for given ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period for analysis (default: 1y)

        Returns:
            RiskMetrics with all computed values, or None if error

        Raises:
            ValueError: If ticker is invalid or insufficient data
            Exception: For other data fetching errors
        """
        try:
            data = self._fetch_stock_data(ticker, period)
            returns = self._calculate_returns(data)
            metrics = self._calculate_risk_metrics(ticker, data, returns)
            return metrics
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

    def _fetch_stock_data(self, ticker: str, period: str) -> Any:
        """Fetch stock data from Yahoo Finance"""
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("No data found for this ticker")
        return data

    def _fetch_benchmark_data(self, period: str = "1y") -> Any:
        """Fetch S&P 500 data for Beta calculation"""
        try:
            data = yf.download("SPY", period=period, progress=False, auto_adjust=True)
            if data.empty:
                return None
            return data
        except Exception:
            return None

    def _calculate_returns(self, data: Any) -> np.ndarray:
        """Calculate daily returns from price data"""
        returns = data["Close"].pct_change().dropna()
        if len(returns) < 30:
            raise ValueError("Insufficient data for analysis (need at least 30 days)")
        return returns

    def _calculate_beta(self, stock_returns, benchmark_returns) -> float:
        """Calculate Beta relative to S&P 500"""
        try:
            # Align the two series
            aligned = stock_returns.align(benchmark_returns, join='inner')
            stock_r = aligned[0].values.flatten()
            bench_r = aligned[1].values.flatten()

            if len(stock_r) < 30:
                return 1.0

            covariance = np.cov(stock_r, bench_r)[0][1]
            benchmark_variance = np.var(bench_r)

            if benchmark_variance == 0:
                return 1.0

            return float(covariance / benchmark_variance)
        except Exception:
            return 1.0

    def _calculate_sortino(self, returns) -> float:
        """Calculate Sortino Ratio (downside risk-adjusted return)"""
        try:
            annual_return = float((returns.mean() * 252).item() if hasattr(returns.mean() * 252, 'item') else returns.mean() * 252)
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                return 0.0

            downside_std = float(downside_returns.std().item() if hasattr(downside_returns.std(), 'item') else downside_returns.std())
            downside_deviation = downside_std * np.sqrt(252)

            if downside_deviation == 0:
                return 0.0

            return annual_return / downside_deviation
        except Exception:
            return 0.0

    def _calculate_cvar(self, returns, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            var_threshold = float(np.percentile(returns, confidence * 100))
            tail_losses = returns[returns <= var_threshold]

            if len(tail_losses) == 0:
                return var_threshold

            return float(tail_losses.mean().item() if hasattr(tail_losses.mean(), 'item') else tail_losses.mean())
        except Exception:
            return 0.0

    def _calculate_rsi(self, data, period: int = 14) -> float:
        """Calculate Relative Strength Index (14-day)"""
        try:
            close = data["Close"].values.flatten()
            deltas = np.diff(close)

            if len(deltas) < period:
                return 50.0

            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except Exception:
            return 50.0

    def _calculate_trend(self, data) -> tuple:
        """Analyze trend using 50-day and 200-day SMA"""
        try:
            close = data["Close"].values.flatten()

            sma_50 = float(np.mean(close[-50:])) if len(close) >= 50 else float(np.mean(close))
            sma_200 = float(np.mean(close[-200:])) if len(close) >= 200 else float(np.mean(close))

            if sma_50 > sma_200 * 1.02:
                trend = "Bullish"
            elif sma_50 < sma_200 * 0.98:
                trend = "Bearish"
            else:
                trend = "Neutral"

            return trend, sma_50, sma_200
        except Exception:
            return "Neutral", 0.0, 0.0

    def _calculate_risk_score(self, volatility, beta, max_drawdown, var_95, rsi, sharpe) -> int:
        """Calculate composite risk score from 0 (safest) to 100 (riskiest)"""
        # Volatility component (0-30 points)
        vol_score = min(30, volatility * 50)

        # Beta component (0-20 points)
        beta_score = min(20, max(0, (abs(beta) - 0.5) * 20))

        # Max drawdown component (0-20 points)
        dd_score = min(20, max_drawdown * 50)

        # VaR component (0-15 points)
        var_score = min(15, abs(var_95) * 300)

        # RSI extreme component (0-15 points) - penalize extremes
        rsi_score = 0
        if rsi > 70:
            rsi_score = min(15, (rsi - 70) * 0.5)
        elif rsi < 30:
            rsi_score = min(15, (30 - rsi) * 0.5)

        total = vol_score + beta_score + dd_score + var_score + rsi_score
        return max(0, min(100, int(total)))

    def _generate_recommendation(self, metrics_dict: dict) -> tuple:
        """Generate AI recommendation based on aggregated metrics"""
        score = metrics_dict['risk_score']
        sharpe = metrics_dict['sharpe_ratio']
        sortino = metrics_dict['sortino_ratio']
        rsi = metrics_dict['rsi']
        trend = metrics_dict['trend']
        volatility = metrics_dict['volatility']

        # Score-based base recommendation
        buy_signals = 0
        sell_signals = 0
        reasons = []

        # Sharpe analysis
        if sharpe > 1.5:
            buy_signals += 2
            reasons.append("excellent risk-adjusted returns")
        elif sharpe > 1.0:
            buy_signals += 1
            reasons.append("good risk-adjusted returns")
        elif sharpe < 0.3:
            sell_signals += 1
            reasons.append("poor risk-adjusted returns")

        # Sortino analysis
        if sortino > 2.0:
            buy_signals += 1
            reasons.append("strong downside protection")
        elif sortino < 0.5:
            sell_signals += 1
            reasons.append("weak downside protection")

        # RSI analysis
        if rsi > 75:
            sell_signals += 1
            reasons.append("overbought on RSI")
        elif rsi < 25:
            buy_signals += 1
            reasons.append("oversold on RSI")
        elif 40 <= rsi <= 60:
            reasons.append("neutral momentum")

        # Trend analysis
        if trend == "Bullish":
            buy_signals += 1
            reasons.append("bullish trend (SMA 50 > 200)")
        elif trend == "Bearish":
            sell_signals += 1
            reasons.append("bearish trend (SMA 50 < 200)")

        # Risk score impact
        if score > 70:
            sell_signals += 1
            reasons.append("high composite risk score")
        elif score < 25:
            buy_signals += 1
            reasons.append("low composite risk score")

        # Determine recommendation
        net = buy_signals - sell_signals

        if net >= 3:
            rec = "Strong Buy"
        elif net >= 1:
            rec = "Buy"
        elif net <= -3:
            rec = "Strong Sell"
        elif net <= -1:
            rec = "Sell"
        else:
            rec = "Hold"

        # Build reason string — pick top 2-3 reasons
        top_reasons = reasons[:3]
        reason_text = f"{rec} — {', '.join(top_reasons).capitalize()}"

        return rec, reason_text

    def _calculate_risk_metrics(self, ticker: str, data: Any, returns: np.ndarray) -> RiskMetrics:
        """Calculate all risk metrics including new advanced ones"""
        # Basic metrics
        current_price = data["Close"].iloc[-1]
        if hasattr(current_price, 'item'):
            current_price = current_price.item()

        volatility = float((returns.std() * np.sqrt(252)).item())

        # Value at Risk
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))

        # Maximum drawdown
        cumulative_returns = (returns + 1).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = float(drawdown.max().item())

        # Sharpe ratio
        annual_return = float((returns.mean() * 252).item())
        annual_volatility = float((returns.std() * np.sqrt(252)).item())
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Risk level
        risk_level, risk_color = self._determine_risk_level(volatility)

        # === NEW ADVANCED METRICS ===

        # Beta
        benchmark_data = self._fetch_benchmark_data()
        if benchmark_data is not None:
            benchmark_returns = self._calculate_returns(benchmark_data)
            beta = self._calculate_beta(returns, benchmark_returns)
        else:
            beta = 1.0

        # Sortino
        sortino_ratio = self._calculate_sortino(returns)

        # CVAR
        cvar_95 = self._calculate_cvar(returns, 0.05)

        # RSI
        rsi = self._calculate_rsi(data)

        # Trend
        trend, sma_50, sma_200 = self._calculate_trend(data)

        # Composite risk score
        risk_score = self._calculate_risk_score(volatility, beta, max_drawdown, var_95, rsi, sharpe_ratio)

        # AI Recommendation
        metrics_dict = {
            'risk_score': risk_score,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'rsi': rsi,
            'trend': trend,
            'volatility': volatility,
        }
        recommendation, recommendation_reason = self._generate_recommendation(metrics_dict)

        return RiskMetrics(
            ticker=ticker.upper(),
            current_price=current_price,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            risk_level=risk_level,
            risk_color=risk_color,
            beta=beta,
            sortino_ratio=sortino_ratio,
            cvar_95=cvar_95,
            rsi=rsi,
            trend=trend,
            sma_50=sma_50,
            sma_200=sma_200,
            risk_score=risk_score,
            recommendation=recommendation,
            recommendation_reason=recommendation_reason,
        )

    def _determine_risk_level(self, volatility: float) -> tuple:
        """Determine risk level based on volatility"""
        if volatility > self.risk_thresholds['high']:
            return "HIGH", "high"
        elif volatility > self.risk_thresholds['medium']:
            return "MEDIUM", "medium"
        else:
            return "LOW", "low"

    def format_results(self, metrics):
        """Format risk metrics with all advanced fields"""
        volatility = metrics.volatility * 100
        risk_style = "risk-high" if volatility > 50 else "risk-medium" if volatility > 30 else "risk-low"

        result_text = (
            f"AI RISK SCORE: {metrics.risk_score}/100\n"
            f"Recommendation: {metrics.recommendation}\n\n"
            f"CORE METRICS:\n"
            f"• Annual Volatility: {volatility:.1f}%\n"
            f"• Beta (vs S&P 500): {metrics.beta:.2f}\n"
            f"• Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
            f"• Sortino Ratio: {metrics.sortino_ratio:.2f}\n\n"
            f"RISK METRICS:\n"
            f"• VaR (95%): {metrics.var_95 * 100:.2f}% daily\n"
            f"• VaR (99%): {metrics.var_99 * 100:.2f}% daily\n"
            f"• CVaR (95%): {metrics.cvar_95 * 100:.2f}% daily\n"
            f"• Max Drawdown: {metrics.max_drawdown * 100:.1f}%\n\n"
            f"MOMENTUM:\n"
            f"• RSI (14d): {metrics.rsi:.1f}\n"
            f"• Trend: {metrics.trend}\n"
            f"• SMA 50: ${metrics.sma_50:.2f}\n"
            f"• SMA 200: ${metrics.sma_200:.2f}\n\n"
            f"Risk Level: {metrics.risk_level}"
        )
        return result_text, risk_style

    def format_results_detailed(self, metrics: RiskMetrics) -> str:
        """Format risk metrics with detailed explanations"""
        return f"""DETAILED AI ANALYSIS FOR {metrics.ticker}

Current Price: ${metrics.current_price:.2f}
AI Risk Score: {metrics.risk_score}/100
Recommendation: {metrics.recommendation_reason}

═══ CORE METRICS ═══

Annual Volatility: {metrics.volatility:.1%}
   Measures how much the stock price fluctuates annually

Beta (vs S&P 500): {metrics.beta:.2f}
   Stock sensitivity to market movements (1.0 = moves with market)

Sharpe Ratio: {metrics.sharpe_ratio:.2f}
   Risk-adjusted return (higher is better, >1 is good)

Sortino Ratio: {metrics.sortino_ratio:.2f}
   Downside risk-adjusted return (ignores upside volatility)

═══ RISK METRICS ═══

Value at Risk (95%): {metrics.var_95:.2%} daily
   Maximum expected daily loss 95% of the time

Value at Risk (99%): {metrics.var_99:.2%} daily
   Maximum expected daily loss 99% of the time

CVaR / Expected Shortfall (95%): {metrics.cvar_95:.2%} daily
   Average loss beyond the VaR threshold (tail risk)

Maximum Drawdown: {metrics.max_drawdown:.1%}
   Largest peak-to-trough decline in the period

═══ MOMENTUM & TREND ═══

RSI (14-day): {metrics.rsi:.1f}
   <30 = oversold, >70 = overbought, 30-70 = neutral

Trend: {metrics.trend}
   Based on 50-day vs 200-day Simple Moving Average

SMA 50: ${metrics.sma_50:.2f} | SMA 200: ${metrics.sma_200:.2f}

Risk Level: {metrics.risk_level}

{self._get_risk_interpretation(metrics.risk_level)}"""

    def _get_risk_interpretation(self, risk_level: str) -> str:
        """Get risk level interpretation"""
        interpretations = {
            "LOW": "✅ This stock has relatively low volatility and is considered less risky. Suitable for most investors.",
            "MEDIUM": "⚠️ This stock has moderate volatility. Consider your risk tolerance before investing.",
            "HIGH": "🔴 This stock is highly volatile and risky. Only suitable for risk-tolerant investors with a long time horizon."
        }
        return interpretations.get(risk_level, "Risk level assessment unavailable.")


# ── Convenience functions ──

def analyze_stock_risk(ticker: str) -> Optional[RiskMetrics]:
    """Quick function to analyze stock risk"""
    analyzer = StockRiskAnalyzer()
    return analyzer.analyze_stock(ticker)


def get_formatted_analysis(ticker: str, detailed: bool = False) -> str:
    """Get formatted analysis results"""
    try:
        analyzer = StockRiskAnalyzer()
        metrics = analyzer.analyze_stock(ticker)

        if detailed:
            return analyzer.format_results_detailed(metrics)
        else:
            return analyzer.format_results(metrics)
    except Exception as e:
        return f"ERROR ANALYZING {ticker.upper()}\n\n{str(e)}\n\nPlease check the ticker symbol and try again."


def get_risk_summary(ticker: str) -> str:
    """Get a quick risk summary"""
    try:
        analyzer = StockRiskAnalyzer()
        metrics = analyzer.analyze_stock(ticker)

        return f"""Quick Summary for {metrics.ticker}:
Price: ${metrics.current_price:.2f}
Risk Score: {metrics.risk_score}/100
Volatility: {metrics.volatility:.1%}
Beta: {metrics.beta:.2f}
Risk: {metrics.risk_level}
AI Rec: {metrics.recommendation}"""
    except Exception as e:
        return f"Unable to analyze {ticker}: {str(e)}"


def get_top_low_risk_stocks(limit: int = 5) -> list:
    """Get top low-risk stocks sorted by risk score (lowest first)"""
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'PG', 'KO', 'PEP', 'WMT',
        'HD', 'VZ', 'INTC', 'PFE', 'MRK', 'UNH', 'JPM', 'V', 'MA', 'DIS',
        'NFLX', 'TSLA', 'NVDA', 'AMD', 'CRM', 'ORCL', 'IBM', 'CSCO', 'ADBE', 'PYPL', 'QCOM',
        'HG=F', 'GC=F', 'SI=F', 'CL=F', 'BTC-USD', 'ETH-USD'
    ]

    analyzer = StockRiskAnalyzer()
    results = []

    for ticker in popular_stocks:
        try:
            metrics = analyzer.analyze_stock(ticker)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    # Sort by risk score ascending (lowest risk first)
    results.sort(key=lambda x: x.risk_score)

    return results[:limit]
