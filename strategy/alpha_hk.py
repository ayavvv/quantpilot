"""
Alpha158 扩展因子：基本面 + 宏观。

Alpha158HK：Alpha158 + fundamental + macro（港股专用，含宏观因子）
Alpha158A：Alpha158 + fundamental + macro（全 A 股专用）
Alpha158Fund：Alpha158 + fundamental only（无宏观，向后兼容）
"""

from qlib.contrib.data.handler import Alpha158


class Alpha158HK(Alpha158):
    """Alpha158 + fundamental + macro + short sell + industry factors for HK stocks."""

    EXTRA_FIELDS = [
        # 基本面 - 估值
        ("$pe_ratio", "PE"),
        ("1/($pe_ratio+1e-9)", "INPE"),
        ("$pb_ratio", "PB"),
        ("1/($pb_ratio+1e-9)", "INPB"),
        ("$turnover_rate", "TURNRATE"),
        ("$turnover_rate/(Mean($turnover_rate,5)+1e-9)", "RELTURN5"),
        ("$turnover_rate/(Mean($turnover_rate,20)+1e-9)", "RELTURN20"),
        # 基本面 - 盈利与分红
        ("$dividend_ttm", "DIV_TTM"),
        ("$return_on_equity", "ROE"),
        ("$net_profit_growth_rate", "NP_GROWTH"),
        # 卖空情绪
        ("$short_sell_ratio", "SS_RATIO"),
        ("$short_sell_ratio-Ref($short_sell_ratio,5)", "SS_RATIO_D5"),
        ("Mean($short_sell_ratio,5)", "SS_RATIO_MA5"),
        # 行业
        ("$industry_id", "IND_ID"),
        # 宏观 - VIX
        ("$vix", "VIX"),
        ("$vix/Ref($vix,5)-1", "VIX_CHG5"),
        # 宏观 - 美元指数
        ("$dxy", "DXY"),
        ("$dxy/Ref($dxy,5)-1", "DXY_CHG5"),
        # 宏观 - 美债收益率
        ("$tnx", "TNX"),
        ("$tnx-Ref($tnx,5)", "TNX_CHG5"),
        # 宏观 - SPY 动量
        ("$spy/Ref($spy,5)-1", "SPY_MOM5"),
        ("$spy/Ref($spy,20)-1", "SPY_MOM20"),
    ]

    def get_feature_config(self):
        base_exprs, base_names = super().get_feature_config()
        extra_exprs = [e for e, _ in self.EXTRA_FIELDS]
        extra_names = [n for _, n in self.EXTRA_FIELDS]
        return base_exprs + extra_exprs, base_names + extra_names


class Alpha158A(Alpha158):
    """Alpha158 + fundamental + macro factors for A-shares (全 A 股).

    和 Alpha158HK 相同的因子集，专用于 A 股选股。
    """

    EXTRA_FIELDS = [
        # 基本面
        ("$pe_ratio", "PE"),
        ("1/($pe_ratio+1e-9)", "INPE"),
        ("$pb_ratio", "PB"),
        ("1/($pb_ratio+1e-9)", "INPB"),
        ("$turnover_rate", "TURNRATE"),
        ("$turnover_rate/(Mean($turnover_rate,5)+1e-9)", "RELTURN5"),
        ("$turnover_rate/(Mean($turnover_rate,20)+1e-9)", "RELTURN20"),
        # 宏观 - VIX
        ("$vix", "VIX"),
        ("$vix/Ref($vix,5)-1", "VIX_CHG5"),
        # 宏观 - 美元指数
        ("$dxy", "DXY"),
        ("$dxy/Ref($dxy,5)-1", "DXY_CHG5"),
        # 宏观 - 美债收益率
        ("$tnx", "TNX"),
        ("$tnx-Ref($tnx,5)", "TNX_CHG5"),
        # 宏观 - SPY 动量
        ("$spy/Ref($spy,5)-1", "SPY_MOM5"),
        ("$spy/Ref($spy,20)-1", "SPY_MOM20"),
    ]

    def get_feature_config(self):
        base_exprs, base_names = super().get_feature_config()
        extra_exprs = [e for e, _ in self.EXTRA_FIELDS]
        extra_names = [n for _, n in self.EXTRA_FIELDS]
        return base_exprs + extra_exprs, base_names + extra_names


class Alpha158Fund(Alpha158):
    """Alpha158 + fundamental factors only (no macro). 向后兼容。"""

    EXTRA_FIELDS = [
        ("$pe_ratio", "PE"),
        ("1/($pe_ratio+1e-9)", "INPE"),
        ("$pb_ratio", "PB"),
        ("1/($pb_ratio+1e-9)", "INPB"),
        ("$turnover_rate", "TURNRATE"),
        ("$turnover_rate/(Mean($turnover_rate,5)+1e-9)", "RELTURN5"),
        ("$turnover_rate/(Mean($turnover_rate,20)+1e-9)", "RELTURN20"),
    ]

    def get_feature_config(self):
        base_exprs, base_names = super().get_feature_config()
        extra_exprs = [e for e, _ in self.EXTRA_FIELDS]
        extra_names = [n for _, n in self.EXTRA_FIELDS]
        return base_exprs + extra_exprs, base_names + extra_names
