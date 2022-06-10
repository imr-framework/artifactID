import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main(wa_tp_val, wa_tp_test, wa_ip_val, wa_ip_test, gibbs_nosim_val, gibbs_nosim_test, gibbs_sim_val,
         gibbs_sim_test):
    fig, ax = plt.subplots(4, 2)
    ax = ax.flatten()
    ax[0].set_title('Validation', fontsize=14)
    ax[1].set_title('Test', fontsize=14)
    cbar_ax = fig.add_axes([.91, 0.3, 0.02, .4])
    ticklabel_fontsize = 12

    # Heatmap 1
    h1 = sns.heatmap(wa_tp_val, ax=ax[0], annot=True,
                     vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax, square=True,
                     xticklabels=False, fmt="0.3g")
    ax[0].set_ylabel('Through-plane\nwrap-around', fontsize=14)
    h1.set_yticklabels(h1.get_ymajorticklabels(), fontsize=ticklabel_fontsize)

    # Heatmap 2
    sns.heatmap(wa_tp_test, ax=ax[1], annot=True,
                vmin=0, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=False, fmt="0.3g")

    # Heatmap 3
    h3 = sns.heatmap(wa_ip_val, ax=ax[2], annot=True,
                     vmin=0, vmax=1, cbar=False, square=True,
                     xticklabels=False, fmt="0.3g")
    ax[2].set_ylabel('In-plane\nwrap-around', fontsize=14)
    h3.set_yticklabels(h3.get_ymajorticklabels(), fontsize=ticklabel_fontsize)

    # Heatmap 4
    sns.heatmap(wa_ip_test, ax=ax[3], annot=True,
                vmin=0, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=False, fmt="0.3g")

    # Heatmap 5
    h5 = sns.heatmap(gibbs_sim_val, ax=ax[4], annot=True,
                     vmin=0, vmax=1, cbar=False, square=True,
                     xticklabels=False, fmt="0.3g")
    ax[4].set_ylabel('Gibbs ringing\n(low-field + sim)', fontsize=14)
    h5.set_yticklabels(h5.get_ymajorticklabels(), fontsize=ticklabel_fontsize)

    # Heatmap 6
    sns.heatmap(gibbs_sim_test, ax=ax[5], annot=True,
                vmin=0, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=False, fmt="0.3g")

    # Heatmap 7
    h7 = sns.heatmap(gibbs_nosim_val, ax=ax[6], annot=True,
                     vmin=0, vmax=1, cbar=False, square=True, fmt="0.3g")
    ax[6].set_ylabel('Gibbs ringing\n(low-field only)', fontsize=14)
    h7.set_xticklabels(h7.get_xmajorticklabels(), fontsize=ticklabel_fontsize)

    # Heatmap 8
    h8 = sns.heatmap(gibbs_nosim_test, ax=ax[7], annot=True,
                     vmin=0, vmax=1, cbar=False, square=True,
                     yticklabels=False, fmt="0.3g")
    h8.set_xticklabels(h8.get_xmajorticklabels(), fontsize=ticklabel_fontsize)

    # fig.tight_layout(rect=[0.15, 0.05, 0.9, 0.95])
    plt.show()


if __name__ == "__main__":
    wa_tp_val = np.load(r"wa/through_plane_wa_Godwin_val.npy")
    wa_tp_test = np.load(r"wa/through_plane_wa_Godwin_test.npy")
    wa_ip_val = np.load(r"wa/in_plane_Godwin_val.npy")
    wa_ip_test = np.load(r"wa/in_plane_Godwin_test.npy")
    gibbs_nosim_val = np.load(r"gibbs/cm_nosim_val.npy")
    gibbs_nosim_test = np.load(r"gibbs/cm_nosim_test.npy")
    gibbs_sim_val = np.load(r"gibbs/cm_sim_val.npy")
    gibbs_sim_test = np.load(r"gibbs/cm_sim_test.npy")

    main(wa_tp_val, wa_tp_test, wa_ip_val, wa_ip_test, gibbs_nosim_val, gibbs_nosim_test, gibbs_sim_val, gibbs_sim_test)
