import pandas as pd
import numpy as np
import io
import base64
import itertools
from scipy import stats

from collections import defaultdict
import re
from itertools import combinations, permutations, product
from pyDOE2 import fullfact

from typing import List


class DicUtils:
    @staticmethod
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

    @staticmethod
    def create_dic1(df):
        data_dict = df.to_dict()

        if DicUtils.analyze_index_digit_length(df) ==2:

            # 辞書を整形してDataFrameへ
            records = []
            for b, a_dict in data_dict.items():
                for a, val in a_dict.items():
                    a_level = a[:2]  # A11, A12 → A1 など
                    rep = a[2:]      # A11, A12 → 1, 2 など
                    records.append({
                        'A': a_level,
                        'B': b,
                        'Rep': int(rep),
                        'Value': float(val)
                    })
            return records
        
        elif DicUtils.analyze_index_digit_length(df) ==1:
            data = []
            for B, row in data_dict.items():
                for A, value in row.items():
                    data.append({'A': A, 'B': B, 'Value': float(value)
                })
            return data

    @staticmethod
    def create_dic2(df):
        data_dict = df.to_dict()

        # データを縦長（tidy）形式に変換
        data = []
        for B, row in data_dict.items():
            for A, value in row.items():
                data.append({'A': A, 'B': B, 'Y': float(value)
            })
        return data

    @staticmethod
    def create_dic3(df):
        data_dict = df.to_dict()

        # データを縦長（tidy）形式に変換
        data = []
        for B, row in data_dict.items():
            for A, value in row.items():
                data.append({'A': A, 'B': B, 'Value': float(value)
            })
        return data
    
    @staticmethod
    def create_dict_from_arrays(A, B):
        if len(A) != len(B):
            raise ValueError("配列Aと配列Bの長さは一致していなければなりません。")
        
        C = {A[i]: B[i] for i in range(len(A))}
        return C
    
    @staticmethod
    def create_formula(df_columns, response='Data'):
        """
        指定されたカラムリストから回帰式用のformula文字列を生成する。
        
        Parameters:
        - df_columns: list of str : DataFrameの列名リスト
        - response: str : 目的変数（デフォルトは 'Data'）

        Returns:
        - formula: str : 例）'Data ~ Factor_A + Factor_B + AxB + ...'
        """
        # 目的変数を除いた説明変数を取得
        exclude = {response, 'Residual'}
        predictors = [col for col in df_columns if col not in exclude]
        # フォーミュラを構築
        formula = f"{response} ~ {' + '.join(predictors)}"
        return formula
        
class LogicalSort:
    @staticmethod
    def _parse_exponents(label: str) -> str:
        """指数付きの記法を展開（例：ab^2 → abb）"""
        pattern = r'([a-zA-Z])(?:\^(\d+))?'
        expanded = ''
        for match in re.finditer(pattern, label):
            factor, exp = match.groups()
            count = int(exp) if exp else 1
            expanded += factor * count
        return expanded

    @staticmethod
    def reconstruct_label(expanded: str) -> str:
        """展開された文字列（abbcc）を指数付き記法に戻す（ab^2c^2）"""
        from collections import Counter
        counts = Counter(expanded)
        return ''.join(
            f"{ch}" if counts[ch] == 1 else f"{ch}^{counts[ch]}"
            for ch in sorted(counts)
        )

    @staticmethod
    def logical_sort_with_exponents(components: List[str], factor_order: List[str]) -> List[str]:
        priority = {f: i for i, f in enumerate(factor_order)}

        def sort_key(original_label: str):
            expanded = LogicalSort._parse_exponents(original_label)
            reversed_factors = sorted(set(expanded), key=lambda x: priority.get(x, 999), reverse=True)
            last_char_priority = priority.get(expanded[-1], 999) if expanded else 999
            return (
                last_char_priority,
                [priority.get(f, 999) for f in reversed_factors],
                len(expanded),
                original_label
            )

        sorted_labels = sorted(components, key=sort_key)
        return sorted_labels
    
    @staticmethod
    def logical_sort_by_last_char(components: List[str], factor_order: List[str]) -> List[str]:
        priority = {f: i for i, f in enumerate(factor_order)}

        def sort_key(comp: str):
            reversed_factors = sorted(set(comp), key=lambda x: priority.get(x, 999), reverse=True)
            last_char_priority = priority.get(comp[-1], 999)
            return (
                last_char_priority,                      # 1. 最後の文字の優先度
                [priority.get(ch, 999) for ch in reversed_factors],  # 2. 逆順因子優先度
                len(comp),                               # 3. 重複含む長さ
                comp                                     # 4. 元文字列（安定化）
            )

        return sorted(components, key=sort_key)

    
    @staticmethod
    def _generate_logical_order(basis=['a', 'b', 'c', 'd']):
        """
        線点図に基づく論理順を生成（a, b, ab, c, ac, bc, abc, ..., abcd）
        """
        n = len(basis)
        order = []
        for bits in range(1, 2**n):
            # bits の 1 の位置に対応する文字を選ぶ（例：bits=3 -> ab）
            comb = ''.join([basis[i] for i in range(n) if (bits >> i) & 1])
            order.append(comb)
        return order
    
    
    @staticmethod
    def logical_sort(df, basis=['a', 'b', 'c', 'd']):
        logical_order = LogicalSort._generate_logical_order(basis)

        # 列名を簡約して対応表を作る
        simplified_map = {col: LogicalSort._simplify_component(col) for col in df.columns}

        # 論理順にマッチする列を並べ替え（元の列名を取得）
        ordered_cols = []
        for target in logical_order:
            for col, simp in simplified_map.items():
                if simp == target and col not in ordered_cols:
                    ordered_cols.append(col)

        # その他の列（論理順に含まれていないもの）
        others = [col for col in df.columns if col not in ordered_cols]

        return ordered_cols + others

    
    @staticmethod
    def logical_sort_2(df, basis=['a', 'b', 'c', 'd']):
        logical_order = LogicalSort._generate_logical_order(basis)
        order = [c for c in logical_order if c in df.columns]
        others = [c for c in df.columns if c not in order]
        return order + others

        
class Pooling:
    @staticmethod
    def pooling_items(raw_pool_terms, raw_pool_terms2):
        
        # 2. 交互作用（2文字以上の名前）
        interaction_terms = [term for term in raw_pool_terms2 if len(term) > 1]
        # 3. 交互作用に含まれる主効果因子（文字をばらしてユニークに集める）
        main_factors_in_interactions = set("".join(interaction_terms))
        
        
        # 4. 主効果は、対応する交互作用が含まれるときに pool_terms に含めない
        final_pool_terms = []
        for term in raw_pool_terms:
            if len(term) > 1:
                final_pool_terms.append(term)  # 交互作用はそのまま追加
            elif term not in main_factors_in_interactions:
            
                final_pool_terms.append(term)  # 主効果も交互作用に含まれていれば追加
        
        return final_pool_terms
    
    @staticmethod
    def pooling(anova_table):
        anova_dict = anova_table.to_dict()

        message = ""

        # Elementos a agrupar（p > 0.05）
        pool_terms = [term for term, pval in anova_dict['PR(>F)'].items() if pval > 0.05]
        pool_terms2 = [term for term, pval in anova_dict['PR(>F)'].items() if pval <= 0.05]
        pool_terms = Pooling.pooling_items(pool_terms, pool_terms2)

        # Inicialización
        pooled_ss = anova_dict['sum_sq']['Residual']
        pooled_df = anova_dict['df']['Residual']

        # Sumar los elementos agrupados al Residual
        for term in pool_terms:
            if term != 'Residual':  #
                pooled_ss += anova_dict['sum_sq'][term]
                pooled_df += anova_dict['df'][term]

        # Calucular el cuadrado medio（Mean Square）
        pooled_ms = pooled_ss / pooled_df

        # Crear una nueva table ANOVA （despues del agrupamiento）
        anova_pooled = {
            'Source': [],
            'sum_sq': [],
            'df': [],
            'mean_sq': [],
            'F': [],
            'PR(>F)': []
        }

        original = anova_table.index.to_list()[:-2]
        result = [x for x in original if x not in pool_terms]

        for term in result:
            ss_ = anova_dict['sum_sq'][term]
            df = anova_dict['df'][term]
            ms = ss_ / df
            f_val = ms / pooled_ms
            p_val = stats.f.sf(f_val, df, pooled_df)
            anova_pooled['Source'].append(term)
            anova_pooled['sum_sq'].append(ss_)
            anova_pooled['df'].append(df)
            anova_pooled['mean_sq'].append(ms)
            anova_pooled['F'].append(f_val)
            anova_pooled['PR(>F)'].append(p_val)

        # Residual（después del agrupamiento）
        anova_pooled['Source'].append('Residual')
        anova_pooled['sum_sq'].append(pooled_ss)
        anova_pooled['df'].append(pooled_df)
        anova_pooled['mean_sq'].append(pooled_ms)
        anova_pooled['F'].append(None)
        anova_pooled['PR(>F)'].append(None)

        # Tabla
        anova_df = pd.DataFrame(anova_pooled)
        anova_df.set_index('Source', inplace=True)
        message = f"\n※ Elementos a grupar：{', '.join(pool_terms)}"

        return anova_df, message

    @staticmethod
    def pooling2(anova_table):
        anova_dict = anova_table.to_dict()
        message = ""

        # Elementos a agrupar（p > 0.05）
        pool_terms = [term for term, pval in anova_dict['PR(>F)'].items() if pval > 0.05]
        pool_terms2 = [term for term, pval in anova_dict['PR(>F)'].items() if pval <= 0.05]

        return anova_table
    
    @staticmethod
    def pool(anova_table, pool_factors, pool_):
        anova_table = anova_table[['sum_sq','df','F','PR(>F)']]
        anova_table = anova_table.astype(float)
        # Calcular el error agrupado (pooling)
        pooled_error_ss = anova_table.loc[pool_factors, 'sum_sq'].sum()  # Calucar el error agrupado (pooling)
        pooled_error_df = anova_table.loc[pool_factors, 'df'].sum()      # Calucar el grados de libertad agrupado (pooling)

        # Tabla de análisis de varianza de los factores
        anova_table = anova_table.drop(index=pool_factors)

        # Actualizar el término de error
        anova_table.loc[pool_] = [
            anova_table.loc[pool_, 'sum_sq'] + pooled_error_ss,  # Nueva suma de cuadrados del error
            anova_table.loc[pool_, 'df'] + pooled_error_df,      # Nuevos grados de libertad
            None,  # Mean Square
            None   # F-statistic
        ]

        # Calcular la media de los cuadrados
        anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

        # Calcular el valor F
        anova_table['F'] = anova_table['mean_sq'] / anova_table.loc['Residual', 'mean_sq']

        # Calcular el valor p
        anova_table['PR(>F)'] = stats.f.sf(anova_table['F'], anova_table['df'], anova_table.loc['Residual', 'df'])

        return anova_table
    
    
    

class Verify:
    @staticmethod
    def verify_l16_table(df):
        print("===== ✅ L16直交表 検証開始 =====\n")
        
        # ① 水準分布チェック
        print("① 各因子・交互作用の水準分布:")
        for col in df.columns:
            counts = df[col].value_counts().to_dict()
            print(f"  {col}: {counts}")
        print()

        # ② 相関係数チェック
        print("② 相関係数行列:")
        corr_matrix = df.corr(numeric_only=True)
        print(corr_matrix.round(2))
        print()

        # ③ ランクチェック（Xの線形独立性）
        print("③ 実験行列のランク確認:")
        X = df.to_numpy()
        rank = np.linalg.matrix_rank(X)
        print(f"  ランク = {rank} / 列数 = {X.shape[1]}")
        if rank == X.shape[1]:
            print("  ✅ 全列が線形独立です（直交性OK）")
        else:
            print("  ⚠️ ランク不足 → 直交性に問題あり")
        print()

        # ④ 重複列名チェック
        print("④ 列名の重複確認:")
        if len(df.columns) != len(set(df.columns)):
            print("  ⚠️ 重複した列名があります")
        else:
            print("  ✅ 列名はユニークです")
        print()

        # ⑤ 行数・列数確認
        print("⑤ 行数・列数の確認:")
        print(f"  行数 = {df.shape[0]}, 列数 = {df.shape[1]}")
        if df.shape[0] == 16:
            print("  ✅ 行数OK")
        else:
            print("  ⚠️ 行数が16ではありません")
        print("\n===== ✅ 検証完了 =====")

    # ===============================
    # 使用例：先ほど作成した final_df を検証
    # ===============================
    # 例として A〜Gと交互作用が定義された表（final_df）を使う
    # final_df = pd.DataFrame(...) # 省略、あなたがすでに作成済の表

    @staticmethod
    def check_orthogonality(df):

        message = ""
        message2 = ""

        columns = df.columns
        all_ok = True
        for col1, col2 in combinations(columns, 2):
            combo_counts = df.groupby([col1, col2]).size()
            expected = len(df) // (df[col1].nunique() * df[col2].nunique())
            if not all(combo_counts == expected):
                message += f"❌ {col1} × {col2} → 不完全直交\n"
                all_ok = False
            else:
                message2 += f"✅ {col1} × {col2} → 完全直交\n"
        if all_ok:
            message = ""
            message += "\n✅✅ 全ペアが完全直交です。"
        else:
            message += "\n⚠️ 一部のペアに直交性の欠如があります。"

        return message
    

    @staticmethod
    def _is_orthogonal(col1, col2):
        """2列の直交性をチェック"""
        df_temp = pd.DataFrame({'a': col1, 'b': col2})
        count = df_temp.groupby(['a', 'b']).size()
        expected = len(col1) // (col1.nunique() * col2.nunique())
        return all(count == expected)
    
    @staticmethod
    def _check_all_pairs(df):
        """全列ペアで直交性を確認し、非直交ペアと非直交列を返す"""
        bad_pairs = []
        bad_columns = set()
        for col1, col2 in combinations(df.columns, 2):
            if not Verify._is_orthogonal(df[col1], df[col2]):
                bad_pairs.append((col1, col2))
                bad_columns.update([col1, col2])
        return bad_pairs, list(bad_columns)
    
    @staticmethod
    def _generate_xor_column(df, col1, col2):
        """XOR列を生成"""
        return df[col1] ^ df[col2]
    
    @staticmethod
    def _try_rebuild_columns(df, target_count, max_rebuild=10):
        message = ""
        """再構築可能な直交列を探索して追加"""
        rebuilt = []
        candidates = list(df.columns)
        attempts = 0
        while len(df.columns) + len(rebuilt) < target_count and attempts < max_rebuild:
            for col1, col2 in combinations(candidates, 2):
                new_col = Verify._generate_xor_column(df, col1, col2)
                name = f"{col1}_xor_{col2}"
                # 既存列と直交性確認
                if all(Verify._is_orthogonal(new_col, df[c]) for c in df.columns) and all(Verify._is_orthogonal(new_col, rc) for rc in rebuilt):
                    rebuilt.append(new_col.rename(name))
                    message += f"✅ {name} を再構築し追加しました\n"
                    if len(df.columns) + len(rebuilt) >= target_count:
                        break
            attempts += 1
        return pd.concat([df] + rebuilt, axis=1)
    
    @staticmethod
    def auto_rebuild_orthogonal(df_original, target_count=6):
        """
        不完全な列を除外し、XORで再構築して完全直交列にする
        target_count: 最終的に残したい直交列数（例：6列）
        """

        message = ""

        df = df_original.copy()
        
        # ステップ1: 非直交列を除外
        _, bad_columns = Verify._check_all_pairs(df)
        if bad_columns:
            message += f"❌ 非直交列: {bad_columns} を除去\n"
            df = df.drop(columns=bad_columns)
        else:
            message += "✅ すべての列は直交しています\n"

        # ステップ2: XOR列で再構築
        df = Verify._try_rebuild_columns(df, target_count=target_count)

        # ステップ3: 再度直交性を検証
        _, still_bad = Verify._check_all_pairs(df)
        if not still_bad and df.shape[1] >= target_count:
            message += f"\n✅ 完全直交な {target_count} 列が構築されました！"
            return df.iloc[:, :target_count] # 最初のtarget_count列を返す
        else:
            message += f"\n⚠️ 再構築後も直交性に問題あり。{target_count}列に満たないか不完全です。"
            return df
        

    @staticmethod
    def _check_balance(series, tolerance=1):
        """1つの因子に対する水準の出現回数が均等かどうか"""
        counts = series.value_counts()
        return counts.max() - counts.min() <= tolerance

    @staticmethod
    def check_orthogonality_2(df):

        message = ""
        """割り付け表の直交性チェック"""
        factors = df.columns
        issues = []

        # 1. 各因子単体の水準バランスチェック
        for col in factors:
            if not Verify._check_balance(df[col], 0):
                issues.append(f"水準不均等: 因子 {col}")

        # 2. 因子のペア（交互作用）についての直交性チェック
        for col1, col2 in combinations(factors, 2):
            # クロス集計
            crosstab = pd.crosstab(df[col1], df[col2])
            # 各セルの出現回数が均等かチェック
            values = crosstab.values.flatten()
            values = values[values > 0]  # 欠損は除外（発生していない水準組合せは許容）
            if len(set(values)) > 1:
                issues.append(f"直交性に問題: {col1} × {col2}")

        # 結果出力
        if not issues:
            message += "✅ この割り付け表は完全直交表です。"
        else:
            message += "⚠️ 不完全な点があります：\n"
            for issue in issues:
                message += " -"
                message += issue
                message += "\n"

        return message
    

    

    @staticmethod
    def check_orthogonality_3(df, factor_levels):
        """
        df: DataFrame
        factor_levels: dict, e.g., {'A': 3, 'B': 2, ..., 'H': 2}
        """

        message = ""


        factors = df.columns
        issues = []

        # 主効果（単因子）のバランスチェック
        for col in factors:
            expected_levels = factor_levels[col]
            actual_counts = df[col].value_counts()
            if len(actual_counts) != expected_levels or not Verify._check_balance(df[col]):
                issues.append(f"水準不均等: 因子 {col}（出現数: {dict(actual_counts)}）")

        # 交互作用の直交性チェック
        for col1, col2 in combinations(factors, 2):
            ctab = pd.crosstab(df[col1], df[col2])
            values = ctab.values.flatten()
            values = values[values > 0]
            if values.max() - values.min() > 1:
                issues.append(f"直交性に問題: {col1} × {col2}")

        # 結果
        if not issues:
            message += "✅ この割り付け表は完全直交表です。"
        else:
            message += "⚠️ 不完全な点があります："
            for issue in issues:
                message += str(issue)
                message += ""
                message += "\n"

        return message

class Optim:
    @staticmethod
    # === 1. 最適水準の組み合わせを求める ===
    def obtener_niveles_optimos(df, factores):
        niveles_optimos = {}
        for f in factores:
            medias = df.groupby(f, observed=True)['y'].mean()
            nivel_optimo = medias.idxmax()
            niveles_optimos[f] = nivel_optimo
        return niveles_optimos

    @staticmethod
    # === 2. 推定値を求める ===
    def estimar_media_optima_2(df, pares_interaccion):
        suma = 0
        for par in pares_interaccion:
            media_max = df.groupby(par, observed=True)['y'].mean().max()
            suma += media_max
        media_global = df['y'].mean()
        return suma - media_global
    
    @staticmethod
    def estimar_media_optima(df, factores):
        """
        factores: ['A', 'B'] など、交互作用を含めたリスト
        """
        group_means = df.groupby(factores, observed=True)['y'].mean()
        media_max = group_means.max()
        return media_max
    

class L16:
    @staticmethod
    def _generate_L16_base():
        # L16の代表的な直交配列表（最初の8列のみ使用）
        data = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 1, 2, 2, 1, 2],
            [2, 1, 2, 2, 1, 1, 2, 1],
            [2, 2, 1, 1, 2, 1, 2, 1],
            [2, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 2, 1, 1, 2, 1, 1],
            [1, 1, 2, 2, 2, 1, 2, 2],
            [1, 2, 1, 1, 2, 1, 1, 2],
            [1, 2, 1, 2, 1, 2, 2, 1],
            [2, 1, 1, 1, 2, 2, 2, 2],
            [2, 1, 1, 2, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2]
        ]


        columns = ['A1', 'A2', 'B', 'C', 'D','E','F','G']
        df = pd.DataFrame(data, columns=columns)
        df.index = range(1, 17)  # 実験番号を 1 から 16 に
        df = df.iloc[:,:-1]
        return df

    # L16直交配列（4因子、2水準）
    @staticmethod
    def _generate_L16_base__():
        # 4因子、それぞれ2水準（2^4）
        levels = [2, 2, 2, 2, 2, 2, 2]  # それぞれの因子の水準数
        
        # fullfactで直交配列を作成
        L16 = fullfact(levels)  # (16行, 4列)
        
        # pandas DataFrame に変換
        df = pd.DataFrame(L16, columns=['A1', 'A2', 'B', 'C', 'D','E','F'])
        df = df.astype(int) + 1
        
        return df
    
    @staticmethod
    def _compute_A(A1, A2):
        """A1, A2 を 3水準因子 A に変換（Taguchi法に準拠）"""
        code = (A1 - 1) * 2 + (A2 - 1)  # 0〜3
        # 0→1, 1→2, 2→3, 3→1（繰り返し割り付け）
        return code.map({0: 1, 1: 2, 2: 3, 3: 1})
    
    @staticmethod
    def _assign_interactions(df):
        """AxB, AxC, BxCの交互作用列の追加"""
        # A1×B, A2×B, A1×C, A2×C（XOR形式で2水準に）
        df['A1xB'] = ((df['A1'] != df['B']).astype(int) + 1)
        df['A2xB'] = ((df['A2'] != df['B']).astype(int) + 1)
        df['A1xC'] = ((df['A1'] != df['C']).astype(int) + 1)
        df['A2xC'] = ((df['A2'] != df['C']).astype(int) + 1)

        # BxC もXOR的に
        df['BxC'] = ((df['B'] != df['C']).astype(int) + 1)
        return df
    
    @staticmethod
    def generate_mixed_L16():
        """混合水準直交表（A:3水準、他:2水準、交互作用含む）を生成"""
        df = L16._generate_L16_base()
        df['A'] = L16._compute_A(df['A1'], df['A2'])
        df = L16._assign_interactions(df)
        #df.index += 1  # 実験番号を1からに
        return df
    
    @staticmethod
    # XOR関数（1,2の水準を0,1にしてXOR → 0,1 → 1,2に戻す）
    def _xor_2level(a, b):
        return (np.bitwise_xor(a - 1, b - 1)) + 1

    @staticmethod
    def generate_mixed_L16_2():
        # A, B, C, D の2水準（1, 2）完全直交配列表（16行）
        base = fullfact([2, 2, 2, 2]) + 1  # 水準を1,2に調整
        df = pd.DataFrame(base, columns=['A', 'B', 'C', 'D'])
        df = df.astype(int)


        # 2. 一次交互作用
        first_order_pairs = list(combinations(['A','B','C','D'],2))
        for f1, f2 in first_order_pairs:
            col_name = f1+f2
            df[col_name] = L16._xor_2level(df[f1], df[f2])

        # 3. 2次交互作用（ABxC, ABxD)
        second_order=[('AB','C'), ('AB', 'D')]
        for f1, f2 in second_order:
            col_name = f1 + 'x' + f2
            df[col_name] = L16._xor_2level(df[f1], df[f2])

        return df
    
    @staticmethod
    def generate_mixed_L16_3():
        # L16(2^15) Tablas de diseño ortogonal
        # (definición de los más representativos)
        L16 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
            [2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1],
            [2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2],
            [2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2],
            [2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2]
        ])

        # Asignación de A (tres niveles): duplicar A2 para expresarlo
        # en una tabla ortogonal de dos niveles

        A = np.tile([1, 2, 3, 2], 4)
        B, C, D, F, G = L16[:, 0], L16[:, 1], L16[:, 2], L16[:, 3], L16[:, 4]

        # Elaborar una tabla de planificación experimental
        df = pd.DataFrame({
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'F': F,
            'G': G
        })

        return df
    
    @staticmethod
    def generate_mixed_L16_4():
        # 手動で定義された L16（2水準×15列）
        L16 = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,1,1,0,0,0,1,1,1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0,0,0,0,1,1,1,0,0],
            [1,0,1,0,1,1,0,0,1,1,0,0,1,0,1],
            [1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
            [1,1,0,0,1,0,1,0,0,1,1,0,1,1,0],
            [1,1,0,1,0,1,0,1,1,0,0,1,0,0,1],
            [0,0,1,0,1,0,0,1,0,1,1,0,1,0,0],
            [0,0,1,1,0,1,1,0,1,0,0,1,0,1,1],
            [0,1,0,0,0,1,1,0,0,1,0,1,1,1,1],
            [0,1,0,1,1,0,0,1,1,0,1,0,0,0,0],
            [1,0,0,0,0,1,0,1,1,0,1,0,1,1,1],
            [1,0,0,1,1,0,1,0,0,1,0,1,0,0,0],
            [1,1,1,0,1,0,1,1,0,0,0,1,0,1,0],
            [1,1,1,1,0,1,0,0,1,1,1,0,1,0,1]
        ]

        # 列名
        columns = [f"L{i+1}" for i in range(15)]

        # DataFrame作成
        df = pd.DataFrame(L16, columns=columns)

        # ▼▼▼ 1. 任意に L1, L2 を選び、交互作用列 L3 を作成
        col1 = 'L1'
        col2 = 'L2'
        col3 = 'L3_generated'

        df[col3] = df[col1] ^ df[col2]  # XOR で L3作成

        # ▼▼▼ 2. L1, L2, L3_generated の中から任意の2列で4水準Aを作成
        # 今回は col1='L1', col2='L2' で作成（自由に変えてOK）
        def create_4level_factor(df, a_col, b_col, new_name='A'):
            combo = df[[a_col, b_col]].astype(str).agg(''.join, axis=1)
            mapping = {'00': 1, '01': 2, '10': 3, '11': 4}
            df[new_name] = combo.map(mapping)
            return df

        df = create_4level_factor(df, col1, col2)

        # ▼▼▼ 3. L1, L2, L3_generated を削除
        df.drop(columns=[col1, col2, col3], inplace=True)

        # ▼▼▼ 4. 残りの列から任意に5列選んで B〜F に割り当て
        # ここでは L4, L5, L6, L7, L8 を選ぶ例（必要に応じて変更可）
        selected_cols = ['L4', 'L5', 'L15', 'L7', 'L8']
        df.rename(columns=dict(zip(selected_cols, ['B', 'C', 'D', 'E', 'F'])), inplace=True)

        # 不要な列を削除（A, B〜F 以外）
        final_cols = ['A', 'B', 'C', 'D','E', 'F']
        df_final = df[final_cols]

        # 手動で
        df_final2 = df_final.copy()
        df_final2 = df_final2.drop(columns=['D'])
        df_final2['D'] = df_final2['E'] ^ df_final2['F']

        # ▼▼▼ 最終出力
        message = "最終的な割付表（Aは4水準、B〜Fは2水準）:"
        return df_final2, message
    
    @staticmethod
    def generate_taguchi():
        # L16 (3^1 2^7) タグチ直交表の定義（文献や表から取得）
        # 各行が1実験、列が因子（A:3水準、B〜H:2水準）
        L16_array = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 2, 2, 2],
            [1, 2, 2, 2, 2, 1, 1, 1],
            [2, 1, 2, 1, 2, 1, 2, 2],
            [2, 1, 2, 2, 1, 2, 1, 1],
            [2, 2, 1, 1, 2, 2, 1, 1],
            [2, 2, 1, 2, 1, 1, 2, 2],
            [3, 1, 2, 1, 2, 2, 1, 2],
            [3, 1, 2, 2, 1, 1, 2, 1],
            [3, 2, 1, 1, 2, 1, 2, 1],
            [3, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 2, 2, 2, 1],
            [1, 1, 1, 2, 1, 1, 1, 2],
            [1, 2, 2, 1, 2, 1, 1, 2],
            [1, 2, 2, 2, 1, 2, 2, 1],
        ])

        # DataFrame化
        df = pd.DataFrame(L16_array, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

        # 各因子の水準をint型に（元からintでも再確認）
        df = df.astype('category')

        # 表示
        return df
    
    @staticmethod
    def _interaction(col1,col2):
        return col1.astype(str) + col2.astype(str)

    @staticmethod
    def generate_mixed_L16_5():
        # Step 1: L16(2^15)の一部（8列分）のデータ（JMPやMinitabからエクスポートして利用可能）
        l16_array = [
            [0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,1,1],
            [0,1,0,0,1,1,1,0],
            [0,1,1,1,0,0,0,1],
            [1,0,0,1,0,1,1,0],
            [1,0,1,0,1,0,0,1],
            [1,1,0,1,1,0,0,1],
            [1,1,1,0,0,1,1,0],
            [0,0,0,1,1,0,0,1],
            [0,0,1,0,0,1,1,0],
            [0,1,0,1,0,1,0,1],
            [0,1,1,0,1,0,1,1],
            [1,0,0,0,1,1,0,0],
            [1,0,1,1,0,0,1,1],
            [1,1,0,0,0,0,1,0],
            [1,1,1,1,1,1,0,1]
        ]

        # Step 2: データフレームに変換
        df = pd.DataFrame(l16_array, columns=['Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8'])

        # Step 3: 各因子の割り当て
        factor_columns = {
            'A': 'Col1',
            'B': 'Col2',
            'C': 'Col3',
            'D': 'Col4',
            'F': 'Col5',
            'G': 'Col6'
        }

        # Step 4: 交互作用の定義（XOR演算）
        interactions = {
            'A×B': ('A', 'B'),
            'A×C': ('A', 'C'),
            'D×F': ('D', 'F'),
            'D×G': ('D', 'G')
        }

        # Step 5: 因子列に名前を付ける
        for name, col in factor_columns.items():
            df[name] = df[col]

        # Step 6: 交互作用列を追加（XOR演算で生成）
        for name, (f1, f2) in interactions.items():
            df[name] = df[f1] ^ df[f2]

        # Step 7: 必要な列だけ取り出す
        final_columns = list(factor_columns.keys()) + list(interactions.keys())
        final_df = df[final_columns]

        # Step 8: 結果表示
        return final_df
    

    @staticmethod
    def generate_mixed_L16_0():
        # Especificar los niveles de cada factor
        levels = [3, 2, 2, 2]  # A: 3 niveles, B～G: 2 niveles

        # Generar todas las combinaciones de factores utilizando fullfact
        design = fullfact(levels)

        # Covertir los niveles a enteros (0, 1, 2 -> 1, 2, 3)
        design = design.astype(int) + 1

        # Muestreo aleatorio para obtener el número necesario de filas (hacerlo L16)
        design = pd.DataFrame(design, columns=['A', 'B', 'C', 'D']).sample(n=16, random_state=42).reset_index(drop=True)

        # Mostrar los resultados
        message = "Tabla de arreglo ortogonal L16 (muestreo aleatorio):"
        
        return design, message


