# PHATE Analysis


## Installation

**[⬇️ Click here to install in Cauldron](http://localhost:50060/install?repo=https%3A%2F%2Fgithub.com%2Fnoatgnu%2Fphate-analysis-plugin)** _(requires Cauldron to be running)_

> **Repository**: `https://github.com/noatgnu/phate-analysis-plugin`

**Manual installation:**

1. Open Cauldron
2. Go to **Plugins** → **Install from Repository**
3. Paste: `https://github.com/noatgnu/phate-analysis-plugin`
4. Click **Install**

**ID**: `phate-analysis`  
**Version**: 1.0.0  
**Category**: analysis  
**Author**: CauldronGO Team

## Description

PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) for dimensionality reduction

## Runtime

- **Environments**: `python`

- **Entrypoint**: `phate_analysis.py`

## Inputs

| Name | Label | Type | Required | Default | Visibility |
|------|-------|------|----------|---------|------------|
| `input_file` | Input File | file | Yes | - | Always visible |
| `annotation_file` | Sample Annotation File | file | No | - | Always visible |
| `columns_name` | Sample Columns | column-selector (multiple) | Yes | - | Always visible |
| `n_components` | Number of Components | number (min: 2, max: 10, step: 1) | Yes | 2 | Always visible |
| `log2` | Apply Log2 Transform | boolean | No | false | Always visible |
| `cluster_method` | Clustering Method | select (none, kmeans, dbscan) | No | none | Always visible |
| `n_clusters` | Number of Clusters | number (min: 2, max: 20, step: 1) | No | 5 | Visible when `cluster_method` = `kmeans` |
| `dbscan_eps` | DBSCAN Epsilon | number (min: 0, max: 10, step: 0) | No | 0.5 | Visible when `cluster_method` = `dbscan` |
| `dbscan_min_samples` | DBSCAN Min Samples | number (min: 2, max: 50, step: 1) | No | 5 | Visible when `cluster_method` = `dbscan` |

### Input Details

#### Input File (`input_file`)

Data matrix file containing samples and measurements


#### Sample Annotation File (`annotation_file`)

Optional annotation file for sample grouping and coloring (Sample, Condition, Batch, Color)


#### Sample Columns (`columns_name`)

Select columns containing sample data for PHATE analysis

- **Column Source**: `input_file`

#### Number of Components (`n_components`)

Number of PHATE components to compute


#### Apply Log2 Transform (`log2`)

Apply log2 transformation to the data before PHATE


#### Clustering Method (`cluster_method`)

Method to detect clusters from PHATE embeddings

- **Options**: `none`, `kmeans`, `dbscan`

#### Number of Clusters (`n_clusters`)

Number of clusters for KMeans clustering


#### DBSCAN Epsilon (`dbscan_eps`)

Maximum distance between samples for DBSCAN neighborhood


#### DBSCAN Min Samples (`dbscan_min_samples`)

Minimum number of samples in a neighborhood for DBSCAN core points


## Outputs

| Name | File | Type | Format | Description |
|------|------|------|--------|-------------|
| `phate_output` | `phate_output.txt` | data | tsv | PHATE coordinates for each sample |

## Sample Annotation

This plugin supports sample annotation:

- **Samples From**: `columns_name`
- **Annotation File**: `annotation_file`

## Visualizations

This plugin generates 2 plot(s):

### PHATE Plot (by Condition) (`phate-scatter-condition`)

- **Type**: scatter
- **Data Source**: `phate_output`
- **Default**: Yes
- **Customization Options**: 9 available

### PHATE Plot (by Cluster) (`phate-scatter-cluster`)

PHATE scatter plot colored by detected clusters

- **Type**: scatter
- **Data Source**: `phate_output`
- **Customization Options**: 9 available

## Requirements

- **Python Version**: >=3.11

### Package Dependencies (Inline)

Packages are defined inline in the plugin configuration:

- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `phate>=1.0.0`
- `matplotlib>=3.7.0`
- `scikit-learn>=1.3.0`

> **Note**: When you create a custom environment for this plugin, these dependencies will be automatically installed.

## Example Data

This plugin includes example data for testing:

```yaml
  input_file: diann/imputed.data.txt
  columns_name_source: diann/imputed.data.txt
  columns_name: [C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_01.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_02.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_03.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_01.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_02.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_03.raw]
  n_components: 2
  log2: true
```

Load example data by clicking the **Load Example** button in the UI.

## Usage

### Via UI

1. Navigate to **analysis** → **PHATE Analysis**
2. Fill in the required inputs
3. Click **Run Analysis**

### Via Plugin System

```typescript
const jobId = await pluginService.executePlugin('phate-analysis', {
  // Add parameters here
});
```
