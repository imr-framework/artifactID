import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main(cm_test, cm_val):
    CM = [cm_val, cm_test]
    fig, axn = plt.subplots(2, 1, sharex=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, ax in enumerate(axn.flat):
        sns.heatmap(CM[i], ax=ax, annot=True,
                    cbar=True, cbar_ax=cbar_ax,
                    vmin=0, vmax=1,
                    square=True)

    axn = axn.ravel()
    axn[0].set_title('Low-field data')
    axn[0].set_ylabel('Validation')
    axn[0].set_yticklabels(['0', '1'])

    axn[1].set_ylabel('Testing')
    axn[1].set_xticklabels(['0', '1'])
    axn[1].set_yticklabels(['0', '1'])

    plt.show()


if __name__ == "__main__":
    cm_test = np.load(r"")
    cm_val = np.load(r"")

    main(cm_test, cm_val)
