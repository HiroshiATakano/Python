import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
import calculate_control_limits_factors
import itertools
from anova_utils import DicUtils  # type: ignore

def calculate(n):
    A2, D3, D4, B3, B4 = calculate_control_limits_factors.f(n)
    return A2, D4

def detect_out_of_control(y, cl, threshold=7):
    out_of_control = []

    # 連続した高値・低値のグループを見つける
    # 高い状態が続くか低い状態が続くかを個別に検出
    for key, group in itertools.groupby(enumerate(y), lambda x: x[1] >= cl):
        group_list = list(group)
        if key and len(group_list) >= threshold:
            out_of_control.extend(group_list)

    for key, group in itertools.groupby(enumerate(y), lambda x: x[1] <= cl):
        group_list = list(group)
        if key and len(group_list) >= threshold:
            out_of_control.extend(group_list)

    return out_of_control


def detect_trend_out_of_control(y, threshold=7):
    out_of_control = set()  # 重複防止のため set にする
    inc_streak = [0]
    dec_streak = [0]

    for i in range(1, len(y)):
        if y[i] > y[i - 1]:  # 増加
            inc_streak.append(i)
            dec_streak = [i]
        elif y[i] < y[i - 1]:  # 減少
            dec_streak.append(i)
            inc_streak = [i]
        else:
            inc_streak = [i]
            dec_streak = [i]

        if len(inc_streak) >= threshold:
            out_of_control.update(inc_streak)
        if len(dec_streak) >= threshold:
            out_of_control.update(dec_streak)

    # インデックス順に整列し、(index, value) のタプルにして返す
    return [(i, y[i]) for i in sorted(out_of_control)]


class QCChart:
    def __init__(self, data: pd.DataFrame, group_size: int = 4):
        self.data = data
        self.group_size = group_size
        self.measurements = self.data.iloc[:,:self.group_size].astype(float)
        self.means = self.measurements.mean(axis=1)
        self.ranges = self.measurements.max(axis=1) - self.measurements.min(axis=1)

    def _calculate_limits(self):
        A2, D4 = calculate(self.group_size)
        mean_of_means = self.means.mean()
        r_bar = self.ranges.mean()

        UCL = mean_of_means + A2*r_bar
        UCL2 = mean_of_means + A2*r_bar*2/3
        UCL1 = mean_of_means - A2*r_bar/3
        LCL1 = mean_of_means - A2*r_bar/3
        LCL2 = mean_of_means - A2*r_bar*2/3
        LCL = mean_of_means - A2*r_bar
        
        CL = mean_of_means
        return UCL, LCL, CL, UCL2, UCL1, LCL2, LCL1
    
    def plot_xbar_chart(self):
        UCL, LCL, CL, UCL2, UCL1, LCL2, LCL1 = self._calculate_limits()
        y = self.means.astype(float).tolist()
        x = np.arange(1, len(y)+1)
        
        out_of_control = [(i, value) for i, value in enumerate(y) if value > UCL or value < LCL]
        out_of_control2 = detect_out_of_control(y, CL)
        out_of_control3 = detect_trend_out_of_control(y,threshold=7)

        i = self.means[16]
        fig, ax = plt.subplots(figsize=(10,4))

        ax.plot(x, y, marker='o', label='Average')

        for i, value in out_of_control:
            ax.plot(x[i],y[i], marker='o', color='red')
       
        for i, value in out_of_control2:
            ax.plot(x[i], y[i], marker='o', color='red')
        
        for i, value in out_of_control3:
            ax.plot(x[i], y[i], marker='o', color='red')
        
        ax.set_title(r"$ \bar{X} $ " + f"   n = {self.group_size}")
        ax.axhline(UCL, color='r', linestyle='--', label=f'UCL {UCL:.2f}')
        ax.axhline(UCL2, linestyle='-', linewidth=0.5)
        ax.axhline(UCL1, linestyle='-', linewidth=0.5)
        ax.axhline(CL, color='g', linestyle='-', label=f'CL {CL:.2f}')
        ax.axhline(LCL1, linestyle='-', linewidth=0.5)
        ax.axhline(LCL2, linestyle='-', linewidth=0.5)
        ax.axhline(LCL, color='r', linestyle='--', label=f'LCL {LCL:.2f}')
        ax.legend()
        return fig
    
    def to_base64(self) -> str:
        fig = self.plot_xbar_chart()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_gase64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_gase64


class QCChart2:
    def __init__(self, csv_file, group_size: int = 4):
        self.group_size = group_size
        self.data = pd.read_csv(csv_file, index_col=0)

        self.measurements = self.data.iloc[:, :self.group_size].astype(float)
        self.means = self.measurements.mean(axis=1)
        self.ranges = self.measurements.max(axis=1) - self.measurements.min(axis=1)
        self.ranges_mean = self.ranges.mean()

    def _calculate_limits(self):
        mean_of_means = self.means.mean()
        r_bar = self.ranges.mean()
        param = calculate(self.group_size)
        A2 = param[0]
        D4 = param[1]
        UCL = mean_of_means + A2*r_bar
        LCL = mean_of_means - A2*r_bar
        CL = mean_of_means
        UCL_R = D4*r_bar

        return UCL, LCL, CL, UCL_R
    
    def plot_xbar_chart(self):
        UCL, LCL, CL, UCL_R = self._calculate_limits()

        y = self.means.astype(float).tolist()
        x = self.data.index

        out_of_control = [(i, value) for i, value in enumerate(y) if value > UCL or value < LCL]
        fig, ax = plt.subplots(2, figsize=(10,8))

        ax[0].plot(x, y, marker='o', label='Average')

        for i, value in out_of_control:
            ax[0].plot(x[i],y[i], marker='o', color='red')

        ax[0].set_title(r"$ \bar{X} $ " + f"   n = {self.group_size}")
        ax[0].axhline(UCL, color='r', linestyle='--', label=f'UCL {UCL:.2f}')
        ax[0].axhline(CL, color='g', linestyle='-', label=f'CL {CL:.2f}')
        ax[0].axhline(LCL, color='r', linestyle='--', label=f'LCL {LCL:.2f}')
        ax[0].set_xticks([])
        ax[0].legend()

        ax[1].set_title(r"$ \bar{R} $")
        ax[1].plot(x,self.ranges, marker='x')
        ax[1].axhline(UCL_R, color='r', linestyle='--', label=f'UCL {UCL_R:.2f}')
        ax[1].axhline(self.ranges_mean, linestyle='-', color='g', label=f'CL {self.ranges_mean:.2f}')
        ax[1].set_xticks(self.data.index)
        ax[1].set_xticklabels(self.data.index, rotation=-45)  # 角度も調整可能
        ax[1].legend()

        plt.tight_layout()

        return fig
    
    def to_base64(self) -> str:
        fig = self.plot_xbar_chart()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


class histograma:
    def __init__(self, data_flat, lower_limit, upper_limit, title=""):
        self.data_flat = data_flat
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mu = np.mean(self.data_flat)
        self.s = np.std(self.data_flat,ddof=True)
        self.title = title

    def plot_histograma(self):
        mu = np.mean(self.data_flat)
        s = np.std(self.data_flat,ddof=True)

        # Determinar un intervalo provisional c
        c = np.sqrt(len(self.data_flat)).astype(int)

        # Límite superior, Límite interior
        UCL = self.upper_limit
        LCL = self.lower_limit

        #
        dmax = self.data_flat.max()
        dmin = self.data_flat.min()
        dkan = (self.upper_limit - self.lower_limit)/c

        #
        arr = np.linspace(LCL, UCL, (c+1))
      
        while np.min(arr) > dmin:
            arr = np.insert(arr, 0, np.min(arr) - dkan)
        
        while np.max(arr) < dmax:
            arr = np.append(arr, (np.max(arr) + dkan))
        
        fig, ax = plt.subplots(figsize=(10,6))

        # Creación de un histograma
        ax.hist(self.data_flat, bins=arr, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=LCL, ymin=0, ymax=plt.ylim()[1], color='red', linestyle='dashed', label=f'Limite inferior: {LCL:.1f}')
        ax.axvline(x=UCL, ymin=0, ymax=plt.ylim()[1], color='red', linestyle='dashed', label=f'Limite superior: {UCL:.1f}')
        ax.axvline(x=mu, ymin=0, ymax=plt.ylim()[1], color='green', linestyle='dashed', label=f'Media: {mu:.2f}')

        ax.legend()
        ax.set_xlabel("Espesor de la junta de goma")
        ax.set_ylabel("Frecuencia")
        ax.set_title(self.title + "  Histograma del espesor de la junta de goma ")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        return fig
    
    def to_base64(self) -> str:
        fig = self.plot_histograma()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


class pareto_chart:
    def __init__(self, df: pd.DataFrame, title: str =""):
        self.df = df
        self.title = title

    def plot_pareto(self):
        fig, ax3 = plt.subplots(figsize=(6,5))
        data_num = len(self.df)

        accum_to_plot = [0] + self.df.iloc[:,3].tolist()

        ax3.bar(range(1, data_num + 1), self.df.iloc[:,0], align="edge", width=-1, edgecolor='k')
        ax3.set_xticklabels([])

        ax3.set_xticks([0.5 + i for i in range(data_num)], minor=True)
        ax3.set_xticklabels(self.df.index.tolist(), minor=True, rotation = -45, ha='left')

        ymx = self.df.iloc[:,0].sum()

        ax3.set_ylim([0, ymx])
        ax3.tick_params(axis="x", which="major", direction="in")
        ax3.set_ylim([0, ymx])
        ax3.set_ylabel(self.df.columns[0])
        ax3.set_xlabel(self.df.index.name)

        ax4 = ax3.twinx()
        ax4.plot(range(data_num+1), accum_to_plot, c="k", marker="o")
        ax4.set_ylim([0, 100])
        ax4.set_xlim([0, data_num])
        ax4.set_ylabel("Tasa acumulada (%)")
        yticks = np.linspace(0, 100, 11)
        ax4.set_yticks(yticks)

        percent_labels = [str(i) + "%" for i in yticks]
        ax4.set_yticklabels(percent_labels)
        ax4.grid(True, which='both', axis='y')
        ax4.set_title("PARETO_CHART " + self.title )

        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.40, top=0.92)

        return fig
    
    def to_base64(self) -> str:
        fig = self.plot_pareto()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    



class BoxChart1:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def plot_boxchart(self):

        fig, ax = plt.subplots(figsize=(8,6))

        sns.boxplot(x="A", y="x", data=self.df, hue="A", palette="Set2", width=0.5, legend=False, ax=ax)

        ax.set_title('Distribución de la resistencia por nivel del factor A')
        ax.set_ylabel('Resistencia (MPa)')
        ax.set_xlabel('Nivel de mezcla Q:R')
        plt.tight_layout()

        return fig

    def to_base64(self) -> str:
        fig = self.plot_boxchart()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_gase64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_gase64
    

class BoxChart2:
    def __init__(self, df: pd.DataFrame, x, y, hue=None, title=None, xlabel=None, ylabel=None):
        self.df = df
        self.x = x
        self.y = y
        self.hue = hue
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def box_chart2(self):
        df_ = self.df.stack().reset_index()
        df_.columns = [self.x, 'Rep', self.y]

        fig, ax = plt.subplots(figsize=(8,5))

        sns.boxplot(x=self.x, y=self.y, data=df_, hue=self.hue, palette="Set2", width=0.5, ax=ax)
        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        plt.tight_layout()

        ax.grid(True)

        return fig
    
    def to_base64(self) -> str:
        fig = self.box_chart2()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_gase64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_gase64



class Graph2:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def creacion_Graph2(self):
        records = DicUtils.create_dic1(self.df)
        df = pd.DataFrame(records)

        # 平均値プロット
        fig, ax = plt.subplots(figsize=(8,6))

        sns.pointplot(x="A", y="Value", hue="B", data=df,
                    palette="Set2", dodge=False, errorbar=None,
                    markers=["o", "s", "D"], linestyles=["-", "--", "-."], ax=ax)

        ax.set_title("Gráfico de medias de los factore A y B")
        ax.set_ylabel("Respuesta (y)")
        ax.set_xlabel("Niveles del factor A")
        ax.grid(True)
        ax.legend(title="Factor B")
        plt.tight_layout()

        return fig
    
    def to_base64(self) -> str:
        fig = self.creacion_Graph2()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_gase64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_gase64


class PointChart:
    def __init__(self, df: pd.DataFrame, x: str, y: str, hue: str = None,
                 title: str = "", xlabel: str = "", ylabel: str = ""):
        self.df = df
        self.x = x
        self.y = y
        self.hue = hue
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self):

        if set(['A','B','Value']).issubset(self.df.columns):
            df = self.df.copy()
            df0 = df.copy()
            

        else:
            print(self.df)
            records = DicUtils.create_dic1(self.df)
            
            df = pd.DataFrame(records)
            df0 = df.copy()

        fig, ax = plt.subplots(figsize=(8, 6))
        x = df0[self.x]
        y = df0[self.y]

        # hueの水準数に応じてmarkersとlinestylesを自動調整
        n_levels = df0[self.hue].nunique() if self.hue else 1
        base_markers = ["o", "s", "D", "^", "v"]
        base_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

        markers = base_markers[:n_levels]
        linestyles = base_linestyles[:n_levels]

        sns.pointplot(x=self.x, y=self.y, hue=self.hue, data=df0,
              markers=base_markers, linestyles=base_linestyles,
              palette="Set2", dodge=False, errorbar=None, ax=ax)
        ax.scatter(x,y)
        ax.set_title(self.title or "Point Plot")
        ax.set_xlabel(self.xlabel or self.x)
        ax.set_ylabel(self.ylabel or self.y)
        ax.grid(True)
        ax.legend(title=self.hue)

        plt.tight_layout()
        return fig
    
    def to_base64(self) -> str:
        fig = self.creacion_Graph2()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor='white')
        buf.seek(0)
        img_gase64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_gase64
