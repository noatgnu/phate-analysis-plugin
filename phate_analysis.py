import os.path

import click
import numpy as np
import pandas as pd
import phate
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


def detect_clusters(embedding: np.ndarray, method: str, n_clusters: int = 5, eps: float = 0.5, min_samples: int = 5):
    """
    Detect clusters from PHATE embeddings.

    :param embedding: PHATE embedding array.
    :param method: Clustering method ('kmeans', 'dbscan', or 'none').
    :param n_clusters: Number of clusters for KMeans.
    :param eps: DBSCAN epsilon parameter.
    :param min_samples: DBSCAN minimum samples parameter.
    :return: Cluster labels array.
    """
    if method == "none":
        return None

    scaler = StandardScaler()
    scaled_embedding = scaler.fit_transform(embedding)

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(scaled_embedding)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(scaled_embedding)
    else:
        return None

    return labels


def phate_(
    input_file: str,
    output_folder: str,
    columns_name: list[str],
    n_components: int = 2,
    log2: bool = False,
    cluster_method: str = "none",
    n_clusters: int = 5,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5
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
    :return: PHATE dataframe with optional cluster assignments.
    """
    assert n_components in [2, 3], "Invalid number of components"

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
    if n_components == 2:
        phate_df.rename(columns={0: "x_phate", 1: "y_phate"}, inplace=True)
    else:
        phate_df.rename(columns={0: "x_phate", 1: "y_phate", 2: "z_phate"}, inplace=True)

    phate_df["sample"] = columns_name

    if cluster_method != "none":
        cluster_labels = detect_clusters(
            phate_op,
            method=cluster_method,
            n_clusters=n_clusters,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples
        )
        if cluster_labels is not None:
            phate_df["cluster"] = [f"Cluster_{label}" for label in cluster_labels]

    os.makedirs(output_folder, exist_ok=True)
    phate_df.to_csv(os.path.join(output_folder, "phate_output.txt"), sep="\t", index=False)

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
def main(
    input_file: str,
    output_folder: str,
    columns_name: str,
    n_components: int,
    log2: bool,
    cluster_method: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int
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
        dbscan_min_samples
    )


if __name__ == '__main__':
    main()
