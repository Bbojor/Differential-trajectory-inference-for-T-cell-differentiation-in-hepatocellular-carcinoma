import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
import numpy as np
from matplotlib import colormaps

CELLTYPE_COLOR_DICT = {
    "c1_CD8_PD1lo effector": "#1f77b4",
    "c4_CD8_PD1hi_TEX": "#ff7f0e",
    "c5_CD8_PD1lo_Mem 1": "#2ca02c",
    "c7_CD8_PD1lo_TREM": "#d62728",
    "c9_CD8_PD1lo_Cytotoxic": "#9467bd",
    "c12_CD8_PD1hi_Cytotoxic": "#8c564b",
    "c13_CD8_PD1lo Mem 2": "#e377c2",
    "c14_CD8_PD1hi Tpex": "#7f7f7f",
}


TISSUE_COLOR_DICT = {"Control": "#1f77b4", "Core": "#ff7f0e", "Rim": "#2ca02c"}


def plot_root_end_point_composition(adata):

    # make colors consistent across different pie charts
    cluster_colors = CELLTYPE_COLOR_DICT
    tissue_colors = TISSUE_COLOR_DICT

    # select root and end points as in cytopath
    root_adata = adata[adata.obs["root_cells"] > 0.99]
    end_adata = adata[adata.obs["end_points"] > 0.99]

    fig, axs = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
    root_tissue_counts = root_adata.obs.Tissue.value_counts()
    root_tissue_counts.plot(
        kind="pie",
        title="Root cell tissue composition",
        autopct="%1.1f%%",
        colors=[tissue_colors[tissue] for tissue in root_tissue_counts.index],
        ax=axs[0][0],
    )

    end_tissue_counts = end_adata.obs.Tissue.value_counts()
    end_tissue_counts.plot(
        kind="pie",
        title="End point tissue composition",
        autopct="%1.1f%%",
        colors=[tissue_colors[tissue] for tissue in end_tissue_counts.index],
        ax=axs[1][0],
    )
    root_cluster_counts = root_adata.obs["Cluster label"].value_counts()
    ax = root_cluster_counts.plot(
        kind="pie",
        title="Root cell cluster composition",
        autopct="%1.1f%%",
        ylabel="",
        labeldistance=None,
        colors=[cluster_colors[cluster] for cluster in root_cluster_counts.index],
        ax=axs[0][1],
    )

    # put labels in legend as they are very long
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    end_cluster_counts = end_adata.obs["Cluster label"].value_counts()
    ax = end_cluster_counts.plot(
        kind="pie",
        title="End point cluster composition",
        autopct="%1.1f%%",
        ylabel="",
        labeldistance=None,
        colors=[cluster_colors[cluster] for cluster in end_cluster_counts.index],
        ax=axs[1][1],
    )

    # put labels in legend as they are very long
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")


def clonotype_cellrank_analysis(
    adata,
    clonotype,
    n_cells=20,
    n_neighbors=5,
    min_states=5,
    max_states=10,
    n_start_states=1,
    n_end_states=1,
):

    # select clonotype
    adata_clon = adata[adata.obs["Clonotype"] == clonotype].copy()
    print(adata_clon.shape)

    # plot clonotype composition
    scp.pl.umap(adata_clon, color=["Tissue", "Cluster label"], size=50)

    scp.pp.neighbors(
        adata_clon,
        n_pcs=20,
        n_neighbors=n_neighbors,
        random_state=0,
        use_rep="X_pca_harmony",
    )

    adata_clon.obs.groupby("Cluster label").size().plot.pie(autopct="%1.1f%%")

    # apply cellrank kernel
    vk = cr.kernels.VelocityKernel(adata_clon, backward=False)
    vk.compute_transition_matrix()

    # use estimator
    g = cr.estimators.GPCCA(vk)
    # compute and plot decomposition
    g.compute_schur(n_components=14)
    g.plot_spectrum(real_only=True, legend_loc="center right")

    n_states = min_states if min_states == max_states else [min_states, max_states]
    # fit macrostates automatically
    g.fit(n_states=n_states, cluster_key="Cluster label", n_cells=n_cells)
    g.plot_macrostates(which="all", legend_loc="right", s=100)
    g.plot_macrostate_composition(key="Cluster label", figsize=(7, 4))
    g.plot_coarse_T()

    # predict initial/terminal states
    g.predict_initial_states(n_states=n_start_states, n_cells=n_cells)
    g.plot_macrostates(which="initial", legend_loc="right", s=100)

    g.predict_terminal_states(
        n_states=n_end_states,
        method="top_n",
        n_cells=n_cells,
        allow_overlap=True,
    )
    g.plot_macrostates(which="terminal", legend_loc="right", s=100)

    # create a very basic ordering [0,1,2]
    adata_clon.obs["cellrank_ordering"] = 1.0
    adata_clon.obs["cellrank_ordering"].mask(
        g.terminal_states.notnull(), 2.0, inplace=True
    )
    adata_clon.obs["cellrank_ordering"].mask(
        g.initial_states.notnull(), 0.0, inplace=True
    )

    scp.pl.umap(adata_clon, color="cellrank_ordering")

    scp.pl.violin(
        adata_clon, keys=["cellrank_ordering"], groupby="Cluster label", rotation=90
    )

    # save it in the big dataframe
    adata.obs.loc[adata_clon.obs.index, "cellrank_ordering"] = adata_clon.obs[
        "cellrank_ordering"
    ]