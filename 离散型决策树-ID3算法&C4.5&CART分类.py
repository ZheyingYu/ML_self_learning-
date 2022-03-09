### 统计学习方法-第五章决策树-ID3算法和C4.5算法和CART分类
### 代码还涉及 类class, 子类subclass 的学习内容



import numpy as np
import pandas as pd



def loadDataSet():  # 本书例题的数据集
    dataset = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否']]
    label = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    data_df = pd.DataFrame(dataset, columns = label)

    return data_df


class ID3_algorithm():
    def __init__(self, D, threshold):
        self.threshold = threshold
        self.D = D


    # 计算 H(D) 经验熵
    def get_empirical_emtropy(self,df):
        '''计算 H(D) 经验熵
        D: dataframe 数据集D
        '''
        # 首先确定类别的个数，也就是标签的个数
        K = 2 #是和否

        p_0 = df.loc[df.类别 == '否'].shape[0]/df.shape[0]
        p_1 = df.loc[df.类别 == '是'].shape[0]/df.shape[0]
        if p_0 != 0 and p_1 != 0:
            output = -sum([p_0*np.log2(p_0), p_1*np.log2(p_1)]) #一定用log2 因为这里k = 2
            return output
        else:
            return 0

    # 经验条件熵 H(D|A)
    def get_empirical_conditional_emtropy(self,A,D):
        '''特征A给定条件下D的经验条件熵 H(D|A)
        H(D|A) = sum((D_i/D) * H(D_i) ) 其中D_i 表示特征 A取值某个值a_i 的时候的数据子集
        A: string 特征名称
        D: dataframe 数据集
        '''
        A_list = list(D[A].unique())
        output_sum = 0

        for a in A_list:
            D_part = D.loc[D[A] == a]
            coordinate = D_part.shape[0] / D.shape[0]
            output_sum += coordinate * self.get_empirical_emtropy(D_part)
        return output_sum

    # 获得信息增益值
    def get_information_gain(self,A, D):
        '''计算信息增益值，表示得知特征X的信息而使得类Y的信息的不确定性减少的程度
        A: string 输入的特征名称
        D: dataframe 原数据集
        g(D|A) = H(D)-H(D|A)
        '''
        H_D = self.get_empirical_emtropy(D)
        H_DA = self.get_empirical_conditional_emtropy(A, D)
        output = H_D - H_DA

        return output

    def best_information_gain(self, D):
        '''选择最大信息增益的特征，输出特征名称和信息增益值'''
        feature_list = list(D.columns)[0:-1]
        max_feature = '' #最佳特征名称
        max_information_gain = 0 #最佳特征信息增益值
        for feature in feature_list:
            information_gain = self.get_information_gain(feature, D)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                max_feature = feature

        return [max_feature, max_information_gain]


    # ID3决策树，重点复习递归算法
    def ID3_algorithm_value(self):
        '''从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，
        由该特征的不同取值建立子结点；再对子节点递归地调用以上方法，构建决策树；知道所有特征的信息增益均很小或
        没有特征可以选择为止，最后得到一颗决策树'''
        df = self.D
        # 若df中所有实例属于同一类，责T为单结点树，并将C_k 作为该结点的类标记
        if df.类别.value_counts().shape[0] == 1:
            return df.类别.values[0]
        # 若没有可以用的特征，即特征A为空集，则D中实例数最大的类作为该结点的类标记
        elif len(list(df.columns)) == 1: # 可以用的特征为空集
            df_class_count = pd.DataFrame(df.类别.value_counts()).reset_index()
            return df_class_count.iloc[0,0]
        # 计算信息增益，选择最大增益的特征 A_good
        else:
            A_good, A_good_gain_value =self.best_information_gain(df)[0], self.best_information_gain(df)[1]
            T = {A_good:{}} #构建树

            # 如果最大特征的信息增益小于阈值，那么T依旧是单节点树，df中实例数最大的类为该结点的类标记
            if A_good_gain_value < self.threshold:
                df_class_count = pd.DataFrame(df.类别.value_counts()).reset_index()
                return df_class_count.iloc[0, 0]
            else:
                # 按照A_good的取值个数，将集合df分成若干非空子集D_i, 生成子节点，然后以D_i 为训练集，以A-{A_good} 为特征集，
                # 递归地调用子树
                a_list = list(df[A_good].unique())
                a_dic = {}
                for a in a_list:
                    df_i = df.loc[df[A_good] == a].drop(A_good, axis = 1)
                    self.D = df_i
                    a_dic[a] = self.ID3_algorithm_value() #字典嵌套的一种方式
                T[A_good] = a_dic
        return T


id3 = ID3_algorithm(D = loadDataSet(), threshold = 0)
print(id3.ID3_algorithm_value()) #{'有自己的房子': {'否': {'有工作': {'否': '否', '是': '是'}}, '是': '是'}}


#----------------------------------------------------------------------------------------------------------------

class C_45_algorithm(ID3_algorithm): #继承父类，有许多函数是重合的
    def __init__(self, D, threshold):
        super(C_45_algorithm, self).__init__(D, threshold) #继承父类参数变量


        # 获得信息增益值

    def get_information_gain_ratio(self,A,D):
        '''获得属性A下的信息增益比
        g_R(D,A) = g(D,A)/H_A(D),
        其中 H_A(D) = - sum((D_a/D)*log2(D_a/D)) for all a in A, a 是A的取值
        输入：
        A：string 特征名称
        D: dataframe 数据集'''
        # 计算信息增益值
        g_DA = self.get_information_gain(A, D)
        # 计算关于特征A 的熵
        H_A_D = 0
        a_list = list(D[A].unique())
        for a in a_list:
            D_a = D.loc[D[A] == a]
            H_A_D -= (D_a.shape[0] / D.shape[0])*np.log2((D_a.shape[0]/D.shape[0]))

        return g_DA / H_A_D

    def best_information_gain_ratio(self,df):
        '''获得最佳属性和信息增益比的值'''
        best_feature = ''
        best_value = 0
        A_list = list(df.columns)[0:-1]
        for A in A_list:
            temp_value = self.get_information_gain_ratio(A,df)
            if temp_value >= best_value:
                best_value = temp_value
                best_feature = A

        return [best_feature, best_value]


    def C45_algorithm_value(self):
        '''C4.5算法与ID3算法相似，区别在于C4.5是用信息增益比来选特征'''
        df = self.D
        # 若df中所有实例属于同一类，责T为单结点树，并将C_k 作为该结点的类标记
        if df.类别.value_counts().shape[0] == 1:
            return df.类别.values[0]
        # 若没有可以用的特征，即特征A为空集，则D中实例数最大的类作为该结点的类标记
        elif len(list(df.columns)) == 1:  # 可以用的特征为空集
            df_class_count = pd.DataFrame(df.类别.value_counts()).reset_index()
            return df_class_count.iloc[0, 0]
        # 计算信息增益，选择最大增益比的特征 A_good
        else:
            A_good, A_good_gain_value = self.best_information_gain_ratio(df)[0], self.best_information_gain_ratio(df)[1]
            T = {A_good: {}}  # 构建树

            # 如果最大特征的信息增益小于阈值，那么T依旧是单节点树，df中实例数最大的类为该结点的类标记
            if A_good_gain_value < self.threshold:
                df_class_count = pd.DataFrame(df.类别.value_counts()).reset_index()
                return df_class_count.iloc[0, 0]
            else:
                # 按照A_good的取值个数，将集合df分成若干非空子集D_i, 生成子节点，然后以D_i 为训练集，以A-{A_good} 为特征集，
                # 递归地调用子树
                a_list = list(df[A_good].unique())
                a_dic = {}
                for a in a_list:
                    df_i = df.loc[df[A_good] == a].drop(A_good, axis=1)
                    self.D = df_i
                    a_dic[a] = self.C45_algorithm_value()  # 字典嵌套的一种方式
                T[A_good] = a_dic
        return T

C45_model = C_45_algorithm(D = loadDataSet(), threshold = 0)
print(C45_model.C45_algorithm_value()) #{'有自己的房子': {'否': {'有工作': {'否': '否', '是': '是'}}, '是': '是'}}


#-----------------------------------------------------------------------------------------------

# CART 生成算法
def get_gini_index_D(D):
    '''计算所有集合D的基尼指数，表示D的不确定性
    Gini(D) = 1-sum((C_k/D)^2) for k in K 类别标签的个数，C_k 表示D中属于第k类的样本子集'''
    output = 1-((D.loc[D.类别 == '是'].shape[0] / D.shape[0])**2 + (D.loc[D.类别 == '否'].shape[0] / D.shape[0])**2)
    return output



def get_gini_index_DA(A, a, D):
    '''计算经A=a分割后的集合的基尼指数
    Gini_DA = D_1/D * Gini(D_1) + D_2/D * Gini(D_2)
    其中 Gini(D) = 1-sum((C_k/D)^2) for k in K 类别标签的个数，C_k 表示D中属于第k类的样本子集'''
    D_1 = D.loc[D[A] == a]
    D_2 = D.loc[D[A] != a]

    gini_index_value = (D_1.shape[0]/D.shape[0]) * get_gini_index_D(D_1) + (D_2.shape[0] / D.shape[0])*get_gini_index_D(D_2)
    return gini_index_value

def best_