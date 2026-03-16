"""
自定义 Handler：在 Alpha158 基础上加入 PE 和换手率衍生特征。

需要 converter 已将 pe_ratio, turnover_rate 导出为 qlib bin 文件。
"""

from qlib.contrib.data.handler import Alpha158


class Alpha158Fund(Alpha158):
    """Alpha158 + 基本面/流动性衍生特征（~170 个特征）。"""

    def get_feature_config(self):
        fields, names = super().get_feature_config()

        # --- PE 相关特征 ---
        fund_fields = [
            "$pe_ratio",                                            # 原始 PE
            "$pe_ratio / Mean($pe_ratio, 20) - 1",                 # PE 相对 20 日均值偏离
            "$pe_ratio / Mean($pe_ratio, 60) - 1",                 # PE 相对 60 日均值偏离
            "($pe_ratio - Mean($pe_ratio, 60)) / (Std($pe_ratio, 60) + 1e-12)",  # PE 60 日 z-score
        ]
        fund_names = [
            "PE",
            "PE_REL20",
            "PE_REL60",
            "PE_ZSCORE60",
        ]

        # --- 换手率相关特征 ---
        fund_fields += [
            "$turnover_rate",                                       # 原始换手率
            "$turnover_rate / Mean($turnover_rate, 5) - 1",        # 换手率 5 日相对
            "$turnover_rate / Mean($turnover_rate, 20) - 1",       # 换手率 20 日相对
            "Mean($turnover_rate, 5) / (Mean($turnover_rate, 20) + 1e-12)",  # 短期/长期换手率比
        ]
        fund_names += [
            "TURN",
            "TURN_REL5",
            "TURN_REL20",
            "TURN_RATIO",
        ]

        fields += fund_fields
        names += fund_names
        return fields, names
