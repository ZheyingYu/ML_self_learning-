
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 用平方误差准则生成一个二叉回归树

#数据，x和y都应该是连续型
data = pd.DataFrame([[1,4.5],
                     [2,4.75],
                     [3,4.91],
                     [4,5.34],
                     [5,5.8],
                     [6,7.05],
                     [7,7.90],
                     [8,8.23],
                     [9,8.70],
                     [10,9]], columns = ['x', 'y'])

# 生数据可视化
# plt.scatter(data.x, data.y)
# plt.show()

# 算法：最小二乘回归树--在训练数据集所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上的输出值，构建二叉树
# 初始状态下整个D就是一个大区域。每个区域遍历所有变量序号和他们的值，直到找到最优切分变量j和切分点s，然后再把这个区域按照s,j 切分成更小的区域

# 初始状态为例子, 那么j 就是所有序列1到10, s 就是对应的x的取值, 对每个j,s都分成两个区域, 并计算最小平方差
# 步骤1. 对某个j,s划分成两个两个区域，大于s和小于s
# R1(j,s) = {x|x^j <= s}, R2(j,s) = {x|x^j > s}

def get_split_region_left(original_region, j):
    """输出j,s划分的左边区域(左子树)，即 R1(j,s) = {x|x^j <= s}
     original_region: 原区域，初始状态下原区域就是数据集D
     j,s：切分变量和切分点
     """
    # 第j个变量的x取值 = s
    s = original_region.loc[original_region.index == j,'x'].values[0]
    output_region_left = original_region.loc[original_region.x <= s]
    return output_region_left

def get_split_region_right(original_region, j):
    """输出j,s划分的左边区域(左子树)，即 R2(j,s) = {x|x^j > s}
     original_region: 原区域，初始状态下原区域就是数据集D
     j,s：切分变量和切分点
     """
    # 第j个变量的x取值 = s
    s = original_region.loc[original_region.index == j,'x'].values[0]
    output_region_right = original_region.loc[original_region.x > s]
    return output_region_right

# 步骤2：对每个划分好的区域, 输出一个固定值c
def get_c(part_region):
    """c1 = avg(y_i | x_i in R1) """
    c = np.average(part_region.y)
    return c

# 步骤3. 计算平方误差 m = min(c1) sum(x in R1) (y_i -c1)^2 + min(c2) sum(x in R2) (y_i -c2)^2
def get_m_param(region):
    """计算区域内计算平方误差最小的参数，即j 和s """
    def get_m_value(j):
        """计算平方误差"""
        # 左区域
        R1 = get_split_region_left(region, j)
        c1 = get_c(R1)
        # 右边区域
        R2 = get_split_region_right(region, j)
        c2 = get_c(R2)

        # 平方误差
        m = np.sum((R1.y - c1) ** 2) + np.sum((R2.y - c2) ** 2)

        return m

    j_list = list(region.index)
    min_m = 100
    best_j = -1

    #找到最小平方误差的j,s
    for j_value in j_list:
        m_j = get_m_value(j_value)
        if m_j < min_m:
            min_m = m_j
            best_j = j_value
    # s
    best_s = region.loc[region.index == best_j,'x'].values[0]
    return [best_j, best_s, min_m ]



def CART_regression_algorithm(data):
    """CART回归最小二乘算法"""
    # base
    # 如果一个区域里只有一个x, 则输出这个x所对应的值
    if data.shape[0] <= 1:
        return data.y.mean()
    else:
        # 计算最小平方误差
        best_j, best_s = get_m_param(data)[0], get_m_param(data)[1]
        label_str = 'x<=' + str(best_s)
        T = {label_str : {}} #构建树

        # 两种区域
        bool_list = ['True', 'False']
        a_dic = {}
        for bool in bool_list:
            if bool == 'True':
                R_part_left = get_split_region_left(data, best_j)
                a_dic[bool] = CART_regression_algorithm(R_part_left)
            else:
                R_part_right = get_split_region_right(data, best_j)
                a_dic[bool] = CART_regression_algorithm(R_part_right)

        T[label_str] = a_dic

    return T


print(CART_regression_algorithm(data))
#{'x<=5': {'True': {'x<=3': {'True': {'x<=1': {'True': 4.5, 'False': {'x<=2': {'True': 4.75, 'False': 4.91}}}}, 'False': {'x<=4': {'True': 5.34, 'False': 5.8}}}}, 'False': {'x<=7': {'True': {'x<=6': {'True': 7.05, 'False': 7.9}}, 'False': {'x<=8': {'True': 8.23, 'False': {'x<=9': {'True': 8.7, 'False': 9.0}}}}}}}}



