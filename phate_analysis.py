import os.path

import click
import numpy as np
import pandas as pd
import phate
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def phate_analysis(df: pd.DataFrame, n_components: int):
    """
    Perform PHATE dimensionality reduction.

    :param df: Input dataframe with samples as rows.
    :param n_components: Number of PHATE components to compute.
    :return: PHATE embedding array.
    """
    ph = phate.PHATE(n_components=n_components)
    phate_op = ph.fit_transform(df)
    return phate_op


def compute_elbow_inertias(embedding: np.ndarray, max_k: int = 10):
    """
    Compute inertia values for different k values for elbow analysis.

    :param embedding: PHATE embedding array.
    :param max_k: Maximum number of clusters to test.
    :return: Tuple of (k_values, inertias).
    """
    scaler = StandardScaler()
    scaled_embedding = scaler.fit_transform(embedding)

    max_k = min(max_k, len(embedding) - 1)
    k_values = list(range(2, max_k + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_embedding)
        inertias.append(kmeans.inertia_)

    return k_values, inertias


def find_elbow_point(k_values: list, inertias: list) -> int:
    """
    Find the optimal k using the elbow method (maximum curvature).

    :param k_values: List of k values tested.
    :param inertias: List of corresponding inertia values.
    :return: Optimal number of clusters.
    """
    if len(k_values) < 3:
        return k_values[0]

    k_arr = np.array(k_values)
    inertia_arr = np.array(inertias)

    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    inertia_norm = (inertia_arr - inertia_arr.min()) / (inertia_arr.max() - inertia_arr.min() + 1e-10)

    distances = []
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    for i in range(len(k_values)):
        point = np.array([k_norm[i], inertia_norm[i]])
        dist = np.abs(np.cross(line_vec, p1 - point)) / (line_len + 1e-10)
        distances.append(dist)

    optimal_idx = np.argmax(distances)
    return k_values[optimal_idx]


def generate_elbow_plot(k_values: list, inertias: list, optimal_k: int, output_folder: str):
    """
    Generate and save elbow plot as HTML.

    :param k_values: List of k values tested.
    :param inertias: List of corresponding inertia values.
    :param optimal_k: Optimal number of clusters.
    :param output_folder: Path to output folder.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    optimal_idx = k_values.index(optimal_k)
    fig.add_trace(go.Scatter(
        x=[optimal_k],
        y=[inertias[optimal_idx]],
        mode='markers',
        name=f'Optimal k={optimal_k}',
        marker=dict(color='red', size=15, symbol='star')
    ))

    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=f'Elbow Method for Optimal k (Selected: {optimal_k})',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia (Within-cluster Sum of Squares)',
        template='plotly_white',
        showlegend=True
    )

    fig.write_html(os.path.join(output_folder, "elbow_plot.html"))


def detect_clusters(
    embedding: np.ndarray,
    method: str,
    n_clusters: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
    auto_k: bool = False,
    max_k: int = 10,
    output_folder: str = None
):
    """
    Detect clusters from PHATE embeddings.

    :param embedding: PHATE embedding array.
    :param method: Clustering method ('kmeans', 'dbscan', or 'none').
    :param n_clusters: Number of clusters for KMeans.
    :param eps: DBSCAN epsilon parameter.
    :param min_samples: DBSCAN minimum samples parameter.
    :param auto_k: Whether to automatically determine k using elbow method.
    :param max_k: Maximum k to test for elbow method.
    :param output_folder: Output folder for elbow plot.
    :return: Tuple of (cluster labels array, optimal_k or None).
    """
    if method == "none":
        return None, None

    scaler = StandardScaler()
    scaled_embedding = scaler.fit_transform(embedding)
    optimal_k = None

    if method == "kmeans":
        if auto_k:
            k_values, inertias = compute_elbow_inertias(embedding, max_k)
            optimal_k = find_elbow_point(k_values, inertias)
            n_clusters = optimal_k
            if output_folder:
                generate_elbow_plot(k_values, inertias, optimal_k, output_folder)
            print(f"[AUTO] Optimal number of clusters determined: {optimal_k}")

        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(scaled_embedding)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(scaled_embedding)
    else:
        return None, None

    return labels, optimal_k


def generate_3d_plot(
    phate_df: pd.DataFrame,
    output_folder: str,
    color_by: str = "condition",
    title: str = "PHATE 3D Visualization"
):
    """
    Generate interactive 3D PHATE plot using Plotly.

    :param phate_df: DataFrame with PHATE coordinates.
    :param output_folder: Path to output folder.
    :param color_by: Column to use for coloring points.
    :param title: Plot title.
    """
    if "z_phate" not in phate_df.columns:
        return

    color_col = color_by if color_by in phate_df.columns else None

    fig = px.scatter_3d(
        phate_df,
        x="x_phate",
        y="y_phate",
        z="z_phate",
        color=color_col,
        hover_name="sample",
        hover_data={col: True for col in phate_df.columns if col not in ["x_phate", "y_phate", "z_phate"]},
        title=title
    )

    fig.update_traces(marker=dict(size=8, opacity=0.8))

    fig.update_layout(
        scene=dict(
            xaxis_title="PHATE 1",
            yaxis_title="PHATE 2",
            zaxis_title="PHATE 3",
            aspectmode='cube'
        ),
        template='plotly_white',
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(os.path.join(output_folder, "phate_3d.html"))


def phate_(
    input_file: str,
    output_folder: str,
    columns_name: list[str],
    n_components: int = 2,
    log2: bool = False,
    cluster_method: str = "none",
    n_clusters: int = 5,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    auto_k: bool = False,
    max_k: int = 10
):
    """
    Perform PHATE analysis with optional clustering.

    :param input_file: Path to input data file.
    :param output_folder: Path to output folder.
    :param columns_name: List of column names for analysis.
    :param n_components: Number of PHATE components.
    :param log2: Whether to apply log2 transformation.
    :param cluster_method: Clustering method ('none', 'kmeans', 'dbscan').
    :param n_clusters: Number of clusters for KMeans.
    :param dbscan_eps: DBSCAN epsilon parameter.
    :param dbscan_min_samples: DBSCAN minimum samples parameter.
    :param auto_k: Whether to automatically determine k using elbow method.
    :param max_k: Maximum k to test for elbow method.
    :return: PHATE dataframe with optional cluster assignments.
    """
    assert n_components >= 2, "Number of components must be at least 2"

    if input_file.endswith(".tsv") or input_file.endswith(".txt"):
        df = pd.read_csv(input_file, sep="\t")
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file, sep=",")
    else:
        raise ValueError("Invalid file extension")

    data = np.log2(df[columns_name].transpose()) if log2 else df[columns_name].transpose()
    data.replace([np.inf, -np.inf], 0, inplace=True)

    phate_op = phate_analysis(data, n_components)

    phate_df = pd.DataFrame(phate_op)
    col_names = {i: f"phate_{i+1}" for i in range(n_components)}
    col_names[0] = "x_phate"
    col_names[1] = "y_phate"
    if n_components >= 3:
        col_names[2] = "z_phate"
    phate_df.rename(columns=col_names, inplace=True)

    phate_df["sample"] = columns_name

    os.makedirs(output_folder, exist_ok=True)

    optimal_k = None
    if cluster_method != "none":
        cluster_labels, optimal_k = detect_clusters(
            phate_op,
            method=cluster_method,
            n_clusters=n_clusters,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            auto_k=auto_k,
            max_k=max_k,
            output_folder=output_folder
        )
        if cluster_labels is not None:
            phate_df["cluster"] = [f"Cluster_{label}" for label in cluster_labels]

    phate_df.to_csv(os.path.join(output_folder, "phate_output.txt"), sep="\t", index=False)

    if n_components == 3:
        color_by = "cluster" if "cluster" in phate_df.columns else "sample"
        generate_3d_plot(phate_df, output_folder, color_by=color_by, title="PHATE 3D Visualization")

    if optimal_k is not None:
        print(f"Optimal number of clusters (elbow method): {optimal_k}")

    return phate_df


@click.command()
@click.option("--input_file", "-i", help="Path to the input file")
@click.option("--output_folder", "-o", help="Path to the output folder")
@click.option("--columns_name", "-c", help="Name of the columns to be included in the analysis")
@click.option("--n_components", "-n", type=int, help="Number of components", default=2)
@click.option("--log2", "-l", is_flag=True, help="Log2 transform the data")
@click.option("--cluster_method", "-m", type=click.Choice(["none", "kmeans", "dbscan"]), default="none", help="Clustering method")
@click.option("--n_clusters", "-k", type=int, default=5, help="Number of clusters for KMeans")
@click.option("--dbscan_eps", "-e", type=float, default=0.5, help="DBSCAN epsilon parameter")
@click.option("--dbscan_min_samples", "-s", type=int, default=5, help="DBSCAN minimum samples parameter")
@click.option("--auto_k", "-a", is_flag=True, help="Automatically determine optimal k using elbow method")
@click.option("--max_k", type=int, default=10, help="Maximum k to test for elbow method")
def main(
    input_file: str,
    output_folder: str,
    columns_name: str,
    n_components: int,
    log2: bool,
    cluster_method: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    auto_k: bool,
    max_k: int
):
    """PHATE dimensionality reduction with optional cluster detection."""
    phate_(
        input_file,
        output_folder,
        columns_name.split(","),
        n_components,
        log2,
        cluster_method,
        n_clusters,
        dbscan_eps,
        dbscan_min_samples,
        auto_k,
        max_k
    )


if __name__ == '__main__':
    main()
