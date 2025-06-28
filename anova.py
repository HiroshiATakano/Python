import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
import itertools
from scipy import stats

from anova_utils import DicUtils, Pooling  # type: ignore

from collections import defaultdict
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf

def analyze_index_digit_length(df):
    """Indexが英字+数字で構成されていると仮定し、数字部分の桁数を判定"""
    result = []

    for idx in df.index:
        match = re.match(r"[A-Za-z]+(\d+)", str(idx))
        if match:
            digit_length = len(match.group(1))
            result = digit_length
        else:
            result = 0

    return result


def estimacion(df_d, anova_df):
    
    message = ""
    # Calcular el promedio por condición
    mean_table = df_d.groupby(["A", "B"])["Value"].mean().reset_index()

    # AとBの平均、全体平均を計算
    A_means = df_d.groupby('A')['Value'].mean()
    B_means = df_d.groupby('B')['Value'].mean()
    grand_mean = df_d['Value'].mean()

    # Encontrar la condición con el valor más bajo（＝condición óptima）
    optimal_row = mean_table.loc[mean_table["Value"].idxmin()]
    optimal_condition = (optimal_row["A"], optimal_row["B"])

    a_min = optimal_row['A']
    b_min = optimal_row['B']
    mu_min = A_means[a_min] + B_means[b_min] - grand_mean

    # --- Número efectivo de repeticiones（fómula de Ina） ---
    n_a = df_d[df_d["A"] == a_min].shape[0]
    n_b = df_d[df_d["B"] == b_min].shape[0]
    r_eff = 1 / ((1 / n_a) + (1 / n_b) - (1 / len(df_d)))
    message += f"\nNúmero efectivo de repeticiones (n_e): {1 / r_eff:.3f}\n"

    # Cálculo del intervalo de confianza
    MSE = anova_df.iloc[2,3]  # 0.1519
    r = r_eff  #
    df_error = anova_df.iloc[2,2] # 18
    t_val = stats.t.ppf(1 - 0.05 / 2, df_error)

    margin = t_val * np.sqrt(MSE / r)
    ci_lower = mu_min - margin
    ci_upper = mu_min + margin

    # 結果出力
    message += "\n【Estimación del valor medio poblacional bajo la condición óptima y su IC 95%】"
    message += f"\n● Condición óptimal: A = {optimal_condition[0]}, B = {optimal_condition[1]}"
    message += f"\n● Estimación puntual (promedio mínimo): {mu_min:.2f}"
    message += f"\n● Intervalo de confianza del 95%: ({ci_lower:.2f}, {ci_upper:.2f})"

    return message


def estimacion2(df, anova_table):
    df2 = pd.DataFrame(anova_table)

    # 'Residual' または 'Error' をインデックスから探す
    residual_label = None
    for label in ['Residual', 'Error', 'Residuals']:
        if label in df2.index:
            residual_label = label
            break

    if residual_label is None:
        raise ValueError("ANOVA表に 'Residual' が見つかりません")

    message = ""
    # Determinar las condiciones óptimas (con el valor meio máximo)
    means = df.groupby(['A', 'B'])['Value'].mean().reset_index()
    optimal_condition = means.loc[means['Value'].idxmax()]
    optimal_mean = float(optimal_condition['Value'])

    a_level = optimal_condition['A']
    b_level = optimal_condition['B']
    max_value = float(optimal_condition['Value'])

    # Extracción de los datos bajo las condiciones óptimas
    opt_data = df[(df['A'] == optimal_condition['A']) & (df['B'] == optimal_condition['B'])]['Value']
    mean = np.mean(opt_data)

    # Suma de cuadrados del residuo y grados de libertad (del ANOVA)
    SSE = df2.loc["Residual","sum_sq"]
    df_residual = df2.loc["Residual", "df"]
    MSE = SSE / df_residual  # Media cuadrática del residuo

    n = 2 #　Número de repeticiones
    se = np.sqrt(MSE / n)
    t_critical = stats.t.ppf(1 - 0.05/2, df=df_residual)


    # Cálculo del intervalo de confianza
    margin_of_error = t_critical * se
    ci_lower = optimal_mean - margin_of_error
    ci_upper = optimal_mean + margin_of_error

    message += "\n【Optimal Condition】"
    message += f"\nAの最適条件: {a_level} Bの最適条件: {b_level} 最大のValue: {max_value}"
    message += f"\nOptimal Mean: {optimal_mean:.2f}"

    message += f"\n95% Confidence Interval: [{ci_lower}, {ci_upper}]"

    return message, optimal_condition


def estimation3(df, anova_table):

    message = ""
    # Calcular las medias de A y B y la media global
    A_means = df.groupby('A')['Value'].mean()
    B_means = df.groupby('B')['Value'].mean()
    grand_mean = df['Value'].mean()

    # Calcular el promedio por condición
    mean_table = df.pivot_table(index='A', columns='B', values='Value')

    # Encontrar la condición con el promedio más alto
    best_condition = df.loc[df['Value'].idxmax()]

    a_max = best_condition['A']
    b_max = best_condition['B']
    mu_max = A_means['A1'] + B_means['B1'] - grand_mean

    message += f"\n【Condicion óptima】 A={a_max}, B={b_max} → 透過率 = {mu_max:.2f}\n"

    # Grados de libertad y cuadrado medio residual
    MS_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    df_error = anova_table.loc['Residual', 'df']

    # Número efectivo de repeticiones fórmula de Ina：1 / (1/a + 1/b - 1/N)）
    a = df['A'].nunique()
    b = df['B'].nunique()
    n_effective = 1 / (1/a + 1/b - 1/len(df))
    message += f"\n【Número efectivo de repeticiones fórmula de Ina】: {n_effective:.3f}\n"

    # Calcuro de Intervalo de confianza del 95%
    t_val = stats.t.ppf(0.975, df_error)
    margin_error = t_val * np.sqrt(MS_error / n_effective)
    mean_estimate = best_condition['Value']
    ci_lower = mu_max - margin_error
    ci_upper = mu_max + margin_error

    message += f"\n【Estimación puntual 】: {mu_max:.2f}\n"
    message += f"【Intervalo de confianza del 95%】: ({ci_lower:.2f}, {ci_upper:.2f})"

    return message

def estimation4(df, anova_table):

    message = ""

    df_clean = df.dropna()
    # Suma de cuadrados media del error（MSE）y grados de libertad
    MSE = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    df_resid = anova_table.loc['Residual', 'df']
    t_crit = stats.t.ppf(1 - 0.025, df_resid)

    # 2. Estimación puntual de la media poblacional del nivel A3 e intervalo de confianza del 95%
    a3_values = df_clean[df_clean['Nivel'] == 'A3']['Valor']
    mean_a3 = a3_values.mean()
    n_a3 = a3_values.count()
    se_a3 = np.sqrt(MSE / n_a3)
    ci_half_width_a3 = t_crit * se_a3

    message += f"\n【2. Media poblacional de A3】"
    message += f"\n  Estimación puntual: {mean_a3:.2f}"
    message += f"\n  Intervalo de confianza del 95%: [{mean_a3 - ci_half_width_a3:.2f}, {mean_a3 + ci_half_width_a3:.2f}]"

    # 3. Estimación puntual de las medias poblacionales de A1 y A3 e itervalo de confianza del 95%
    a1_values = df_clean[df_clean['Nivel'] == 'A1']['Valor']
    mean_a1 = a1_values.mean()
    n_a1 = a1_values.count()
    mean_diff = mean_a3 - mean_a1
    se_diff = np.sqrt(MSE * (1/n_a1 + 1/n_a3))
    ci_half_width_diff = t_crit * se_diff

    message += f"\n【3. Deferencia de medidas poblacionales (A3 - A1) 】"
    message += f"\n  Estimación puntual: {mean_diff:.2f}"
    message += f"\n  Intervalo de confianza del 95%: [{mean_diff - ci_half_width_diff:.2f}, {mean_diff + ci_half_width_diff:.2f}]"

    return message


def estimation5(df, anova_table):
    message = ""

    # Estimación puntual de la media poblacional para cada nivel
    group_means = df.groupby('Nivel')['Valor'].mean()
    message += "\n【Estimación puntual de la media poblacional para cada nivel】\n"
    message += group_means.to_string()

    # Itervalo de confianza del 95% para cada nivel
    message += "\n【Intervalo de confianza del 95% para cada nivel】\n"
    n_per_group = df['Nivel'].value_counts().iloc[0]  # Número de repeticiones（n = 4）
    MSE = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']  # 残差平均平方
    df_resid = anova_table.loc['Residual', 'df']  # Grados de libertad

    # Conficiente de confianza（valor t）
    t_crit = stats.t.ppf(1 - 0.025, df_resid)

    for level in group_means.index:
        mean = group_means[level]
        ci_half_width = t_crit * np.sqrt(MSE / n_per_group)
        ci_lower = mean - ci_half_width
        ci_upper = mean + ci_half_width
        message += f"{level}: {mean:.2f} ± {ci_half_width:.2f} → CI = [{ci_lower:.2f}, {ci_upper:.2f}]\n"
    
    return message, group_means


def estimation6(df, anova_table):
    message = ""

    # Estimación del efecto de cada ninvel
    grand_mean = df['Value'].mean()
    a1_mean = df[df['A'] == 'A1']['Value'].mean()
    a2_mean = df[df['A'] == 'A2']['Value'].mean()
    a1 = a1_mean - grand_mean
    a2 = a2_mean - grand_mean

    message += "\n【Estimación del efecto del facotr A】\n"
    message += f"a^1 (Estimación del efecto de A1): {a1}\n"
    message += f"a^2 (Estimación del erecto de A2): {a2}\n"

    # Suma de cuadrados y cuadrados medios
    SSA = anova_table.loc['A', 'sum_sq']
    VA = SSA / anova_table.loc['A', 'df']

    message += f"\n【Valor esperado del cuadrado medio del factor A (VA)】\nVA = {VA:.2f}\n"

    # Estimación del efecto del segundo nivel B2 del factor B
    b2_mean = df[df['B'] == 'B2']['Value'].mean()
    b2 = b2_mean - grand_mean
    message += f"\n【EStimación del efecto del segundo nivel B2 del factor B】\n"
    message +=  f"b^2 = {b2}\n"

    # Estimación del errr experimental ε22
    epsilon_22 = df[(df['A'] == 'A2') & (df['B'] == 'B2')]['Value'].values[0] - grand_mean - a2 - b2
    message += f"\n【Estimación del error ε22 】\n"
    message += f"ε22 = {epsilon_22}\n"

    return message



class Anova:
    def __init__(self, df: pd.DataFrame, min=0):
        self.df = df
        self.min = min

    def creacion_Anova(self):
        message = ""

        df = self.df.copy()

        # 総平均（grand mean）
        grand_mean = df['Valor'].mean()

        # 各水準の平均とデータ数
        group_stats = df.groupby('Nivel').agg(['mean', 'count'])['Valor']

        # 因子の平方和（水準間平方和, SSA）
        ssa = sum(group_stats['count'] * (group_stats['mean'] - grand_mean)**2)
        message += f'SSA: {ssa:.2f}\n'

        # 全体平方和（SST）
        sst = sum((df['Valor'] - grand_mean)**2)

        # 誤差平方和（残差平方和, SSE）
        sse = sst - ssa
        message += f'SSE: {sse:.2f}\n'

        # 自由度
        df_a = len(group_stats) - 1
        df_e = len(df) - len(group_stats)
        df_total = len(df) - 1
        message += f'DF_A: {df_a:.2f}\n'
        message += f'DF_E: {df_e:.2f}\n'

        # 平均平方
        ms_between = ssa / df_a
        ms_within = sse / df_e

        # F値
        F_value = ms_between / ms_within

        # p値
        p_value = stats.f.sf(F_value, df_a, df_e)
      

        # 分散分析表を作成
        anova_table = pd.DataFrame({
            'Source': ['A', 'Residual', 'Total'],
            'df': [df_a, df_e, df_total],
            'sum_sq': [ssa, sse, sst],
            'mean_sq': [ms_between, ms_within, np.nan],
            'F': [F_value, np.nan, np.nan],
            'PR(>F)': [p_value, np.nan, np.nan]
        })

        anova_table = anova_table.set_index('Source')

        if self.min == 1:
            message3 = estimation4(df, anova_table)
            message += message3
        elif self.min == 2:
            message3, group_means = estimation5(df, anova_table)
            message += message3
        

        return anova_table, message


class Anova3:
    def __init__(self, df: pd.DataFrame, min=0, p=0):
        self.df = df
        self.min = min
        self.p = p

    def creacion_Anova3(self):

        message = ""

        if set(['A','B','Value']).issubset(self.df.columns):
            df = self.df.copy()
            df_d = df.copy()

            l = df['A'].nunique()
            m = df['B'].nunique()
            r = df.groupby(['A', 'B']).size().min()

        else:
            records = DicUtils.create_dic1(self.df)
            
            df = pd.DataFrame(records)
            df_d = df.copy()
            
            # AとBの水準数を取得
            ll = sorted(set(d['A'] for d in records))
            mm = sorted(set(d['B'] for d in records))

            # A×Bごとの繰り返し数を取得
            rr = defaultdict(set)
            for d in records:
                key = (d['A'], d['B'])
                rr[key].add(d['Rep'])

            l = len(ll)
            m = len(mm)
            r = len(rr[key])
            
        message += f'l:{l} m:{m} r:{r}\n'

        # 総平均
        grand_mean = df['Value'].mean()

        # 総平方和（SST）
        sst = ((df['Value'] - grand_mean)**2).sum()
        
        message += f'SST: {sst:.2f}\n'

        # 因子Aの水準（A11, A12, A21, A22）
        a_means = df.groupby('A')['Value'].mean()
        ssa = sum(df.groupby('A').size() * (a_means - grand_mean) ** 2)
        message += f'SSA: {ssa:.2f}\n'

        # 因子Bの水準（B1〜B4）
        b_means = df.groupby('B')['Value'].mean()
        ssb = sum(df.groupby('B').size() * (b_means - grand_mean) ** 2)
        message += f'SSB: {ssb:.2f}\n'

        # A平均・B平均・交互作用平均の準備
        ab_means = df.groupby(['A', 'B'])['Value'].mean()

        # 交互作用平方和の計算
        ssab = 0
        for (a, b), ab_mean in ab_means.items():
            a_mean = a_means[a]
            b_mean = b_means[b]
            ssab += (ab_mean - a_mean - b_mean + grand_mean) ** 2
        ssab *= r  # 各セルの繰り返し

        message += f'SSAB: {ssab:.2f}\n'

        # 誤差平方和（SSE）
        sse = sst - ssa - ssb - ssab
        message += f'SSE: {sse:.2f}\n'   

        # 自由度
        df_a = l - 1
        df_b = m - 1
        df_ab = df_a * df_b
        df_e = len(df) - l*m

        message += f'DF_A: {df_a}  DF_B: {df_b}  DF_AB: {df_ab}  DF_E: {df_e}\n'

        # 平均平方
        msa = ssa / df_a
        msb = ssb / df_b
        msab = ssab / df_ab
        mse = sse / df_e if df_e != 0 else np.nan

        # F値
        F_a = msa / mse if mse else np.nan
        F_b = msb / mse if mse else np.nan
        F_ab = msab / mse if mse else np.nan

        # p値
        p_a = stats.f.sf(F_a, df_a, df_e)
        p_b = stats.f.sf(F_b, df_b, df_e)
        p_ab = stats.f.sf(F_ab, df_ab, df_e)


        # 分散分析表の作成
        anova_table = pd.DataFrame({
            'sum_sq': [ssa, ssb, ssab, sse, sst],
            'df': [df_a, df_b, df_ab, df_e, len(df)-1],
            'mean_sq': [msa, msb, msab, mse, np.nan],
            'F': [F_a, F_b, F_ab,np.nan,np.nan],
            'PR(>F)': [p_a, p_b, p_ab, np.nan,np.nan]
        }, index=['A', 'B', 'AxB', 'Residual', 'Total'])

        if self.p == 1:
            if max([p_a, p_b, p_ab]) > 0.05:
                #anova_table = anova_table.rename(index={'A': 'C(A)', 'B':'C(B)', 'AxB':'C(A):C(B)'})
                anova_table2, message2 = Pooling.pooling(anova_table)
                message += message2

       
        if self.min:
            message3 = estimacion(df_d, anova_table)
            message += message3
        else:
            message3, table = estimacion2(df_d, anova_table)
            message += message3
            
        return anova_table, message
    

class Anova4:
    def __init__(self, df: pd.DataFrame, min=0):
        self.df = df
        self.min = min

    def creacion_Anova4(self):
        message = ""
        
        #data = DicUtils.create_dic2(self.df)
        data3 = DicUtils.create_dic3(self.df)

        df = pd.DataFrame(data3)
        # 総平均
        grand_mean = df['Value'].mean()

        # 総平方和 SST
        sst = ((df['Value'] - grand_mean)**2).sum()
        message += f'SST: {sst:.2f}\n'

        # Aの水準平均
        A_means = df.groupby('A')['Value'].mean()
        ssa = sum(df.groupby('A').size() * (A_means - grand_mean) ** 2)
        message += f'SSA: {ssa:.2f}\n'

        # Bの水準平均
        B_means = df.groupby('B')['Value'].mean()
        ssb = sum(df.groupby('B').size() * (B_means - grand_mean) ** 2)
        message += f'SSB: {ssb:.2f}\n'

        # 誤差平方和 SSE（SST から SSA, SSB を引く）
        sse = sst - ssa - ssb
        message += f'SSE: {sse:.2f}\n'

        # 水準
        l = len(df['A'].unique())
        m = len(df['B'].unique())

        # 自由度
        df_t = len(df) -1
        df_a = l - 1
        df_b = m - 1
        df_e = df_t - (df_a + df_b)

        message += f'DF_A: {df_a}  DF_B: {df_b}  DF_E: {df_e}\n'

        # 平均平方
        msa = ssa / df_a
        msb = ssb / df_b
        mse = sse / df_e if df_e != 0 else np.nan

        # F値
        F_a = msa / mse if mse else np.nan
        F_b = msb / mse if mse else np.nan

        # p値
        p_a = stats.f.sf(F_a, df_a, df_e)
        p_b = stats.f.sf(F_b, df_b, df_e)

        # 分散分析表の作成
        anova_table = pd.DataFrame({
            'sum_sq': [ssa, ssb, sse, sst],
            'df': [df_a, df_b, df_e, len(df)-1],
            'mean_sq': [msa, msb, mse, ''],
            'F': [F_a, F_b, np.nan, np.nan],
            'PR(>F)': [p_a, p_b, np.nan,np.nan]
        }, index=['A', 'B', 'Residual', 'Total'])

        if self.min == 0:
            message1 = estimation3(df, anova_table)
            message += message1
        elif self.min == 1:
            df_long = self.df.stack().reset_index()
            df_long.columns = ['A', 'B', 'Value']
            message1 = estimation6(df_long, anova_table)
            message += message1

        return anova_table, message
    

class Anova5:
    def __init__(self, df: pd.DataFrame, min=0, p=0):
        self.df = df
        self.min = min
        self.p = p

    def creacion_Anova5(self):
        message = ""
        df, df_d, l, m, r = self._prepare_data()

        message += f'l:{l} m:{m} r:{r}\n'

        # 総平均と平方和
        grand_mean = df['Value'].mean()
        sst = ((df['Value'] - grand_mean)**2).sum()
        message += f'SST: {sst:.2f}\n'

        # 因子平方和
        ssa, a_means = self._sum_of_squares(df, 'A', grand_mean)
        ssb, b_means = self._sum_of_squares(df, 'B', grand_mean)
        message += f'SSA: {ssa:.2f}\n'
        message += f'SSB: {ssb:.2f}\n'

        # 交互作用平方和
        ssab = self._interaction_sum_of_squares(df, a_means, b_means, grand_mean, r)
        message += f'SSAB: {ssab:.2f}\n'

        sse = sst - ssa - ssb - ssab
        message += f'SSE: {sse:.2f}\n'

        # 自由度の計算
        df_a, df_b = l - 1, m - 1
        df_ab = df_a * df_b
        df_e = len(df) - l * m
        message += f'DF_A: {df_a}  DF_B: {df_b}  DF_AB: {df_ab}  DF_E: {df_e}\n'

        # 平均平方とF値、p値
        msa, msb, msab = ssa / df_a, ssb / df_b, ssab / df_ab
        mse = sse / df_e if df_e != 0 else np.nan
        F_a, F_b, F_ab = msa / mse, msb / mse, msab / mse
        p_a, p_b, p_ab = stats.f.sf(F_a, df_a, df_e), stats.f.sf(F_b, df_b, df_e), stats.f.sf(F_ab, df_ab, df_e)

        # 分散分析表
        anova_table = pd.DataFrame({
            'sum_sq': [ssa, ssb, ssab, sse, sst],
            'df': [df_a, df_b, df_ab, df_e, len(df) - 1],
            'mean_sq': [msa, msb, msab, mse, np.nan],
            'F': [F_a, F_b, F_ab, np.nan, np.nan],
            'PR(>F)': [p_a, p_b, p_ab, np.nan, np.nan]
        }, index=['A', 'B', 'AxB', 'Residual', 'Total'])

        if self.p == 1:
            if max([p_a, p_b, p_ab]) > 0.05:
                anova_table2, msg = Pooling.pooling(anova_table)
                message += msg

        if self.min:
            message += estimacion(df_d, anova_table)
        else:
            msg, _ = estimacion2(df_d, anova_table)
            message += msg

        return anova_table, message

    def _prepare_data(self):
        # 横持ち形式かどうかを列名から判定（目安として）
        if set(['A', 'B', 'Value']).issubset(self.df.columns):
            df = self.df.copy()
        else:
            # 横持ちとみなして縦持ちに変換
            records = DicUtils.create_dic1(self.df)
            df = pd.DataFrame(records)

        df_d = df.copy()
        l = df['A'].nunique()
        m = df['B'].nunique()
        r = df.groupby(['A', 'B']).size().min()

        return df, df_d, l, m, r                                


    def _sum_of_squares(self, df, factor, grand_mean):
        means = df.groupby(factor)['Value'].mean()
        ss = sum(df.groupby(factor).size() * (means - grand_mean) ** 2)
        return ss, means

    def _interaction_sum_of_squares(self, df, a_means, b_means, grand_mean, r):
        ab_means = df.groupby(['A', 'B'])['Value'].mean()
        ssab = sum(
            ((ab_mean - a_means[a] - b_means[b] + grand_mean) ** 2)
            for (a, b), ab_mean in ab_means.items()
        )
        return ssab * r


class Anova33:
    def __init__(self, df: pd.DataFrame, min=0, p=0):
        self.df = df
        self.min = min
        self.p = p

    def creacion_Anova33(self):
        message = ""

        data = DicUtils.create_dic1(self.df)
        df = pd.DataFrame(data)
        df_d = df.copy()

        A_levels = sorted(df['A'].unique())
        B_levels = sorted(df['B'].unique())

        l = len(A_levels)
        m = len(B_levels)
        r = df.groupby(['A','B']).size().min()

        message += f'l: {l}  m: {m}  r: {r}\n'

        # データをAxBxR配列に変換
        data_array = np.empty((l,m,r))
        for i, a in enumerate(A_levels):
            for j, b in enumerate(B_levels):
                vals = df[(df['A']==a) & (df['B']==b)].sort_values('Rep')['Value'].values
                data_array[i,j,:] = vals[:r]

        # 総平均
        grand_mean = np.mean(data_array)
        sst = np.sum((data_array - grand_mean)**2)
        message += f'SST: {sst:.2f}\n'

        # 因子Aの平方和
        a_means = data_array.mean(axis=(1,2))
        ssa = r*m*np.sum((a_means - grand_mean)**2)
        message += f'SSA: {ssa:.2f}\n'

        # 因子Bの平方和
        b_means = data_array.mean(axis=(0,2))
        ssb = r*l*np.sum((b_means - grand_mean)**2)
        message += f'SSB: {ssb:.2f}\n'

        # 交互作用平方和
        ab_means = data_array.mean(axis=2)
        ssab = r*np.sum((ab_means - a_means[:,None] - b_means[None, :] + grand_mean)**2)
        message += f'SSAB: {ssab:.2f}\n'

        # 誤差平方和
        sse = sst - ssa - ssb - ssab
        message += f'SSE: {sse:.2f}\n'

        # 自由度
        df_a = l - 1
        df_b = m - 1
        df_ab = df_a * df_b
        df_e = l * m * (r-1)
        
        dfs = np.array([df_a, df_b, df_ab, df_e])
        ssqs = np.array([ssa, ssb, ssab, sse])
        msqs = ssqs / dfs
        Fs = msqs[:3]/msqs[3]
        ps = stats.f.sf(Fs, dfs[:3], dfs[3])

        sst = ssqs.sum()
        df_total = dfs.sum()
        
        anova_table = pd.DataFrame({
            'sum_sq': np.append(ssqs, sst),
            'df':     np.append(dfs, df_total),
            'mean_sq': np.append(msqs, np.nan),
            'F':      np.append(Fs, [np.nan, np.nan]),
            'PR(>F)': np.append(ps, [np.nan, np.nan])
        }, index=['A', 'B', 'AxB', 'Residual', 'Total'])

        if self.p == 1:
            if max(ps) > 0.05:
                #anova_table = anova_table.rename(index={'A': 'C(A)', 'B':'C(B)', 'AxB':'C(A):C(B)'})
                anova_table, message2 = Pooling.pooling(anova_table)
                message += message2
    
        if self.min == 1:
            message3 = estimacion(df_d, anova_table)
            message += message3
        elif self.min == 2:
            message3, table = estimation6(df_d, anova_table)
            message += message3

        else:
            message3 = estimation5(df_d, anova_table)
            message += message3
        
        return message,  anova_table
    

class Anova00:
    def __init__(self, df, a=0, b=0):
        self.df = df
        self.a = a
        self.b = b

    def creacion_ANOVA(self):
        df = self.df.copy()
        # Diseño experimental
        k = df.shape[0]  # Número de niveles
        N = df.count().sum()
        ni = df.count(axis=1)   # Número de repeticiones por grupo

        # Medio
        group_means = df.mean(axis=1, skipna=True)
        grand_mean = df.stack().mean()

        # Suma de cuadrados entre grupos A (SSA)
        SSA = sum(ni * (group_means - grand_mean) ** 2)
        # Suma total de cuadrados (SST)
        SST = sum((df.stack() - grand_mean) ** 2)
        # Suma de cuadrados del error (SSE)
        SSE = SST - SSA             

        # Grados de libertad
        dfA = k - 1
        dfE = N - k
        dfT = N - 1

        # Cuadrado medio
        MSA = SSA / dfA
        MSE = SSE / dfE

        # Valor F
        F = MSA / MSE
        F = np.round(F,3)

        # Crear una tabla de descomposición
        anova_table = pd.DataFrame({
            'Suma de cuadrados (SS)': [SSA, SSE, SST],
            'Grado de libertad (df)': [dfA, dfE, dfT],
            'Suma de cuadrados (MS)': [MSA, MSE, ''],
            'Valor F': [F, '', '']
        }, index=['Entre grupos', 'Error', 'Total'])

        message = ""
        alpha = 0.05  # 有意水準


        if self.a == 1:
            # Prueba de significancia (comparación con el punto superior del 5% de la distribución F)
            F_crit = stats.f.ppf(1 - alpha, dfA, dfE)
            p_value = 1 - stats.f.cdf(F, dfA, dfE)

            message += "=== Prueba de significacia ===\n"
            message += f"Valor crítical del 5% de la distribución F ({dfA}, {dfE}) = {F_crit:.3f}\n"
            message += f"Valor F obervado = {F:.3f}\n"
            message += f"Valor p = {p_value:.3f}\n"
            
            if F > F_crit:
                message += "⇒ Se rechaza la hipótesis nula: existe una diferencia significativa entre los niveles.\n"
            else:
                message += "⇒ No se puede recharzar la hipótesis nula: no existe una diferencia significativa entre los nileles.\n"

            group_means = df.mean(axis=1)
            n = df.shape[0]  # 各群の繰り返し数
            t_crit = stats.t.ppf(1 - alpha / 2, dfE)  # t(6, 0.975)

            if self.b==1:
                message += "\n=== Media poblacional de cada nivel y su intervalo de confianza del 95% ===\n"
                for group, mean in group_means.items():
                    se = np.sqrt(MSE / n)  # 標準誤差
                    ci_low = mean - t_crit * se
                    ci_high = mean + t_crit * se
                    message += f"{group}: Estimación puntual = {mean:.2f}, 95% CI = ({ci_low:.2f}, {ci_high:.2f})\n"
        
        return anova_table, message
    

class Anova11:
    def __init__(self, df, a=0, b=0, p=0):
        self.df = df
        self.a = a
        self.b = b
        self.p = p

    def creacion_ANOVA11(self):    
        message = ""

        df = self.df.copy()

        # 要因リスト
        factors = df.columns.to_list()[:-1]

        # 総平均
        grand_mean = df['y'].mean()

        # 総平方和（SST）
        sst = ((df['y'] - grand_mean) ** 2).sum()

        # 要因の平方和計算
        def calc_ss(df, factor, grand_mean):
            means = df.groupby(factor)['y'].mean()
            counts = df.groupby(factor).size()
            ss = sum(counts * (means - grand_mean) ** 2)
            return ss

        anova_results = []
        ss_total_factors = 0

        for factor in factors:
            ss = calc_ss(df, factor, grand_mean)
            df_factor = df[factor].nunique() - 1  # 2水準なら自由度 = 1
            ms = ss / df_factor
            anova_results.append({
                'Factor': factor,
                'sum_sq': ss,
                'df': df_factor,
                'mean_sq': ms
            })
            ss_total_factors += ss

        # 残差平方和（誤差平方和）
        df_residual = len(df) - 1 - len(factors)
        ss_residual = sst - ss_total_factors
        ms_residual = ss_residual / df_residual

        anova_results.append({
            'Factor': 'Residual',
            'sum_sq': ss_residual,
            'df': df_residual,
            'mean_sq': ms_residual
        })

        # F値の計算
        for row in anova_results:
            if row['Factor'] != 'Residual':
                row['F'] = row['mean_sq'] / ms_residual
                row['PR(>F)'] = stats.f.sf(row['F'], row['df'], df_residual)
            else:
                row['F'] = np.nan
                row['PR(>F)'] = np.nan


        # 分散分析表の作成
        anova_table = pd.DataFrame(anova_results)
        anova_table = anova_table[['Factor', 'sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        anova_table = anova_table.set_index('Factor')
        # 出力
        print("\n【手動による分散分析表】")
        print(anova_table.round(6))

        if self.p == 1:
            if anova_table['PR(>F)'].max() > 0.05:
                anova_table, msg = Pooling.pooling(anova_table)
                message += msg

        return anova_table, message
    

class AnovaVisualizer:
    def __init__(self, formula='y ~ A + B + C + D + F + G + A:B + A:C + B:C'):
        self.formula = formula
        self.model = None
        self.anova_table = None
        self.df = None

    def fit(self, df):
        self.df = df.copy()
        for col in df.columns:
            if col != 'y':
                self.df[col] = self.df[col].astype('category')
        self.model = smf.ols(self.formula, data=self.df).fit()
        self.anova_table = sm.stats.anova_lm(self.model, typ=2)

    def summary(self):
        if self.anova_table is not None:
            print("\n=== 分散分析表 ===")
            print(self.anova_table)
        else:
            print("モデルがまだ適合されていません。fit()を呼んでください。")

    def plot_main_effects(self):
        factors = [col for col in self.df.columns if col != 'y']
        n = len(factors)
        ncols = 3
        nrows = -(-n // ncols)
        ymin = self.df['y'].min()
        ymax = self.df['y'].max()
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()
        for i, factor in enumerate(factors):
            sns.pointplot(x=factor, y='y', data=self.df, ax=axes[i], errorbar=None)
            axes[i].set_title(f'Main Effect: {factor}')
            axes[i].set_ylim(ymin,ymax)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def plot_interactions(self):
        interactions = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        for f1, f2 in interactions:
            sns.pointplot(x=f1, y='y', hue=f2, data=self.df)
            plt.title(f'Interaction Plot: {f1} × {f2}')
            plt.show()

    def plot_residuals(self):
        residuals = self.model.resid
        fitted = self.model.fittedvalues

        # QQ plot
        sm.qqplot(residuals, line='s')
        plt.title("Normal Q-Q Plot")
        plt.show()

        # Residuals vs Fitted
        plt.scatter(fitted, residuals)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.show()

    def pool_and_summary(self, alpha=0.05):
        if self.anova_table is None:
            print("モデルが未適合です。fit()を実行してください。")
            return

        anova = self.anova_table.copy()

        # Residual行の保存
        residual_row = anova.loc["Residual"]
        anova = anova.drop(index="Residual")

        # NaN除去（交互作用の未推定対策）
        anova = anova.dropna()

        # --- 🔒 プールしてはいけない要因リストを構築 ---
        # 例：交互作用 A:B, A:E, B:E → AとBは保護
        anova2 = anova[(anova["PR(>F)"] <= alpha)]
        protected_factors = set()
        for name in anova2.index:
            if ":" in name:
                terms = name.split(":")
                protected_factors.update(terms)
        
        # --- プール候補選定 ---
        to_pool = anova[(anova["PR(>F)"] > alpha) & (~anova.index.isin(protected_factors))]

        if to_pool.empty:
            print("プールすべき要因はありません（すべて有意、または交互作用に含まれています）。")
            anova.loc["Residual"] = residual_row
            print(anova)
            return

        # プールするSSとdf
        ss_pool = to_pool["sum_sq"].sum()
        df_pool = to_pool["df"].sum()

        # 新しい誤差項
        ss_error = residual_row["sum_sq"] + ss_pool
        df_error = residual_row["df"] + df_pool
        ms_error = ss_error / df_error

        # 残った要因（プールしない要因）
        remaining = anova.drop(index=to_pool.index)

        results = []
        for idx, row in remaining.iterrows():
            ms = row["sum_sq"] / row["df"]
            f_val = ms / ms_error
            p_val = stats.f.sf(f_val, row["df"], df_error)
            results.append([idx, row["sum_sq"], row["df"], ms, f_val, p_val])

        # 表示
        print(f"\n=== プール後の分散分析表 (α = {alpha}) ===")
        print(f"{'Source':<15}{'SS':>10}{'df':>5}{'MS':>10}{'F':>10}{'p-value':>12}")
        print("-" * 62)
        for source, ss, df_, ms, f, p in results:
            print(f"{source:<15}{ss:10.2f}{df_:5}{ms:10.2f}{f:10.2f}{p:12.4f}")
        print(f"{'Pooled Error':<15}{ss_error:10.2f}{df_error:5}{ms_error:10.2f}")
