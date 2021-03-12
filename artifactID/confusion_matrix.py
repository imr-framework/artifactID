import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main(cm_sim_val, cm_sim_test, cm_no_sim_val, cm_no_sim_test):
    CM = [cm_no_sim_val, cm_no_sim_test, cm_sim_val, cm_sim_test]
    fig, axn = plt.subplots(2, 2)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, ax in enumerate(axn.flat):
        sns.heatmap(CM[i], ax=ax, annot=True,
                    cbar=True, cbar_ax=cbar_ax,
                    vmin=0, vmax=1,
                    square=True)

    axn = axn.ravel()
    axn[0].set_title('Low-field + simulated data')
    axn[1].set_title('Low-field data')

    axn[0].set_ylabel('Validation')
    axn[2].set_ylabel('Testing')

    axn[0].set_yticklabels(['0', '1'])
    axn[2].set_yticklabels(['0', '1'])

    axn[0].set_xticklabels(['0', '1'])
    axn[1].set_xticklabels(['0', '1'])
    axn[2].set_xticklabels(['0', '1'])
    axn[3].set_xticklabels(['0', '1'])

    plt.show()


if __name__ == "__main__":
    cm_test = np.load(r"")
    cm_val = np.load(r"")

    main(cm_sim_val=cm_sim_val, cm_sim_test=cm_sim_test, cm_no_sim_val=cm_no_sim_val, cm_no_sim_test=cm_no_sim_test)
