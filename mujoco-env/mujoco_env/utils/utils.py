"""
通用工具函数

提供数学和数据处理的实用函数

作者: Liu Gang (原作者)
日期: 2025-12-20
修订: 添加详细注释
"""

import numpy as np


def symlog(x: np.ndarray) -> np.ndarray:
    """
    对称对数变换（Symmetric Logarithm）
    
    对正值和负值都应用对数变换，保持符号不变。
    对于接近0的值，变换是线性的，避免了log(0)的问题。
    
    公式: symlog(x) = sign(x) * log(1 + |x|)
    
    Args:
        x: 输入数据
        
    Returns:
        对称对数变换后的数据
        
    特点:
        - 保持符号不变
        - 压缩大值
        - 对小值近似线性
        - 处理正负值
        
    应用:
        - 强化学习中的奖励缩放
        - 数据可视化
        - 异常值处理
        
    示例:
        >>> x = np.array([-100, -1, 0, 1, 100])
        >>> y = symlog(x)
        >>> print(y)  # [-4.615, -0.693, 0, 0.693, 4.615]
    """
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x: np.ndarray) -> np.ndarray:
    """
    对称指数变换（Symmetric Exponential）
    
    symlog的逆变换，将对称对数空间的值还原。
    
    公式: symexp(x) = sign(x) * (exp(|x|) - 1)
    
    Args:
        x: 对称对数空间的数据
        
    Returns:
        原始空间的数据
        
    示例:
        >>> x = np.array([-1, 0, 1])
        >>> y = symexp(x)
        >>> print(y)  # [-1.718, 0, 1.718]
        >>> # 验证逆变换
        >>> z = symlog(y)
        >>> print(np.allclose(z, x))  # True
    """
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'symlog',
    'symexp',
]

