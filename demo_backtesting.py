from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.strategies.atr_rsi_strategy import (
    AtrRsiStrategy,
)


from vnpy_ctastrategy.strategies.double_ma_strategy import (
    DoubleMaStrategy,
)
from datetime import datetime

engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="BTCUSDT.BINANCE",
    interval="1m",
    start=datetime(2024, 2, 1),
    end=datetime(2024, 2, 3),
    rate=0.3/10000,
    slippage=0.2,
    size=300,
    pricetick=0.1,
    capital=1_000_000,
)
engine.add_strategy(DoubleMaStrategy, {})

engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()

# # 优化（如有）
# setting = OptimizationSetting()
# setting.set_target("sharpe_ratio")
# setting.add_parameter("atr_length", 25, 27, 1)
# setting.add_parameter("atr_ma_length", 10, 30, 10)

# engine.run_ga_optimization(setting)
# engine.run_bf_optimization(setting)