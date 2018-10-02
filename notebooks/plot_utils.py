import matplotlib.pyplot as plt
import seaborn as sns

FIGSIZE = (13, 4)
FONTSIZE_TEXT = 16
COLS_NUM = 2


def metrics_str(data, metrics):
    met_str = ''
    for met in metrics:
        met_str += met.__name__+': '
        met_outcome = met(data)
        met_str += "{:.4f}".format(met_outcome) + '\n'
    return met_str

def show_transform_plot(data, trans_fun, fit_dist, metrics):
    data_after = trans_fun(data)
    sub_titles = ['original', (trans_fun.__name__+'-transformed')]
    annotations = [metrics_str(data, metrics), metrics_str(data_after, metrics)]
    comparison_dist_plots([data, data_after], data.name, sub_titles, fit_dist, annotations)


def comparison_dist_plots(data, main_title=None, sub_titles=None,  fist_dist=None, annotations=None):
    _, axes = plt.subplots(nrows=len(data)//COLS_NUM,
                           ncols=COLS_NUM, figsize=FIGSIZE)
    plt.suptitle(main_title, fontsize=FONTSIZE_TEXT)
    for ax, data_ax, title_ax, text_ax in zip(axes.flat, data, sub_titles, annotations):
        ax.set_title(title_ax)
        ax.text(.7, .85, text_ax, transform=ax.transAxes)
        sns.distplot(data_ax, ax=ax, fit=fist_dist)
    plt.subplots_adjust(wspace=0.2)
    plt.show()