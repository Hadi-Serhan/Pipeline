import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp
import itertools
import networkx as nx


# Loads the dataset from a csv file
def load_data(file_path, columns_to_numeric):
    df = pd.read_csv(file_path, engine='python')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df[columns_to_numeric] = df[columns_to_numeric].apply(pd.to_numeric, errors='coerce')
    return df

# Cleans the dataset by removing rows with NaN values
def clean_data(df):
    return df.dropna()

# Plots a correlation heatmap for the specified columns
def plot_correlation_heatmap(df, columns):
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, square=True, cbar_kws={'label': 'Correlation Coefficient (R)'})
    plt.title("Correlation Matrix with R-values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Checks for missing values in the dataset and prints the count of missing values per column
def analyze_missing_values(df):
    print("Missing values per column:\n")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"{col}: {count}")

# Prints rows with missing values 
def print_missing_rows(df):
    for col in df.columns:
        missing_rows = df[df[col].isna()]
        if not missing_rows.empty:           
            print(f"Rows with missing values in column '{col}':\n")
            print(missing_rows)

# Plots density distributions for each numeric column in the dataset
def plot_density_per_column(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=df, x=col, fill=True)
        plt.title(f"Density Plot: {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

# Describes the distribution and frequency of labels in the dataset
def describe_label_distribution(df, label_col='Label'):
    labels = df[label_col]
    label_counts = labels.value_counts(normalize=True) * 100
    print("Label Frequencies (%):")
    for label, pct in label_counts.items():
        print(f"- {label}: {pct:.2f}%")
    mode_label = labels.mode().iloc[0]
    mode_count = (labels == mode_label).sum()
    mode_pct = label_counts[mode_label]
    print(f"\nMost frequent label: {mode_label} ({mode_count} occurrences, {mode_pct:.2f}%)")

# Describes the variables in the dataset, including their scales and statistics
def describe_variables(df, variable_info):
    for col, scale in variable_info.items():
        print(f"\nVariable: {col}")
        print(f"Scale: {scale.capitalize()}")
        col_data = df[col].dropna()
        if not col_data.empty:
            mode = col_data.mode().iloc[0]
            mode_count = (col_data == mode).sum()
            mode_pct = (mode_count / len(col_data)) * 100
            print(f"Mode: {mode} ({mode_count} occurrences, {mode_pct:.2f}%)")
        else:
            print("Mode: N/A")

        if scale in ['ordinal', 'interval', 'ratio']:
            print(f"Median: {col_data.median()}")
        if scale in ['interval', 'ratio']:
            print(f"Mean: {col_data.mean()}")
            print(f"Standard Deviation: {col_data.std()}")

# Removes duplicate rows from the dataset
def remove_duplicate_rows(df):
    df.drop_duplicates(inplace=True)

def flag_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_col_name = f"{column}_Outlier"
    df[outlier_col_name] = (df[column] < lower_bound) | (df[column] > upper_bound)
    return df, lower_bound, upper_bound

def apply_outlier_flagging(df, columns):
    for col in columns:
        df, lb, ub = flag_outliers_iqr(df, col)
        outlier_col = f"{col}_Outlier"
        num_outliers = df[outlier_col].sum()
        total = len(df)
        percentage = (num_outliers / total) * 100

        print(f"Outlier threshold for {col}:")
        print(f" - Lower bound: {lb}")
        print(f" - Upper bound: {ub}")
        print(f" - Number of outliers in '{col}': {num_outliers} ({percentage:.2f}%)\n")
    return df

def analyze_outlier_overlap(df, outlier_columns):
    print("\nOutlier Overlap Matrix (% of shared outliers):\n")
    for col1 in outlier_columns:
        for col2 in outlier_columns:
            if col1 == col2:
                continue
            # Rows where both columns are marked as outliers
            both_outliers = df[df[col1] & df[col2]]
            count = len(both_outliers)

            # % of outliers in col1 that are also outliers in col2
            col1_outliers = df[df[col1]]
            percent = (count / len(col1_outliers) * 100) if len(col1_outliers) else 0
            if percent >= 50:
                print(f"{col1} ∩ {col2}: {count} rows ({percent:.2f}%)")


def pca_spss_style(df, columns, n_components=None):
    """
    Perform PCA and print SPSS-style tables: variance explained and component matrix.

    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        columns (list): Numeric columns to include in PCA.
        n_components (int or None): Number of components (default = all).
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    components = pca.transform(X_scaled)

    # Explained Variance Table
    eigenvalues = pca.explained_variance_
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)

    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
        'Eigenvalue': eigenvalues,
        '% of Variance': variance_ratios * 100,
        'Cumulative %': cumulative_variance * 100
    })

    print("\n--- Total Variance ---")
    print(variance_df.round(2).to_string(index=False))

    # Component Loadings (Component Matrix)
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                            index=columns)

    print("\n--- Component Matrix --")
    print(loadings.round(2))

    return variance_df, loadings, pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

def run_kmeans_pca_analysis(pca_data, max_k=10):
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_data)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(pca_data, labels))

    # Plot elbow curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouettes, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

    # Return both lists to help decide
    return inertias, silhouettes



def plot_3d_clusters_sampled(pca_df, labels, sample_size=5000):
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=pca_df.index)

    # Create a new DataFrame combining PCA and labels
    combined_df = pca_df.copy()
    combined_df["Cluster"] = labels

    # Stratified sampling: keep at least some of each cluster
    sampled_df = combined_df.groupby("Cluster", group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // len(combined_df["Cluster"].unique())), random_state=42)
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        sampled_df['PC1'],
        sampled_df['PC2'],
        sampled_df['PC3'],
        c=sampled_df['Cluster'],
        cmap='tab10',
        s=10,
        alpha=0.7
    )

    ax.set_title("3D PCA Cluster Visualization (sampled, stratified)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.show()


def run_kmeans_clustering(pca_data, n_clusters):
    """
    Runs KMeans on PCA-transformed data with a specified number of clusters.

    Parameters:
        pca_data (pd.DataFrame or np.ndarray): PCA-reduced feature data
        n_clusters (int): Number of clusters to form

    Returns:
        labels (np.ndarray): Cluster labels for each point
        kmeans (KMeans): The fitted KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pca_data)
    return labels, kmeans


def plot_elbow_method(data, max_k=10):
    inertias = []
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.xticks(ks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def interpret_clusters(df, feature_columns):
    summary = df.groupby('Cluster')[feature_columns].agg(['mean', 'std']).round(2)
    return summary


def plot_normalized_cluster_mean_heatmap(df, feature_columns, cluster_col='Cluster'):
    """
    Plots a heatmap of normalized mean feature values for each cluster.
    Values are normalized per feature using min-max scaling for comparability.
    """
    cluster_means = df.groupby(cluster_col)[feature_columns].mean()
    
    # Min-max normalization (column-wise)
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means_normalized, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Normalized Heatmap of Cluster Mean Feature Values")
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def temporal_analysis(file_path, timestamp_col='Timestamp', cluster_col='Cluster'):
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    
    # Parse timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
    df = df.dropna(subset=[timestamp_col, cluster_col])  # Drop bad timestamps or missing clusters

    # Extract temporal features
    df['Hour'] = df[timestamp_col].dt.hour
    df['DayOfWeek'] = df[timestamp_col].dt.day_name()

    # Plot hourly distribution for each cluster
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Hour', hue=cluster_col, palette='tab10')
    plt.title('Cluster Activity by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # Plot day of week distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='DayOfWeek', hue=cluster_col, order=[
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    plt.title('Cluster Activity by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    hourly_summary = df.groupby([cluster_col, 'Hour']).size().unstack(fill_value=0)
    daily_summary = df.groupby([cluster_col, 'DayOfWeek']).size().unstack(fill_value=0)
    hourly_summary.to_csv("hourly_cluster_distribution.csv")
    daily_summary.to_csv("daily_cluster_distribution.csv")


def run_temporal_ks_tests(df, features, time_col='Hour', cluster_col='Cluster'):
    results = []

    time_groups = df[time_col].dropna().unique()
    for time_val in sorted(time_groups):
        df_time = df[df[time_col] == time_val]
        clusters = df_time[cluster_col].dropna().unique()
        cluster_pairs = list(itertools.combinations(clusters, 2))

        for feature in features:
            for c1, c2 in cluster_pairs:
                x = df_time[df_time[cluster_col] == c1][feature].dropna()
                y = df_time[df_time[cluster_col] == c2][feature].dropna()

                if len(x) > 0 and len(y) > 0:
                    stat, p = ks_2samp(x, y)
                    results.append({
                        'Time': time_val,
                        'Feature': feature,
                        'Cluster 1': c1,
                        'Cluster 2': c2,
                        'KS Statistic (D)': round(stat, 4),
                        'Raw p-value': p
                    })

    ks_df = pd.DataFrame(results)

    # Apply Bonferroni correction
    m = len(ks_df)
    ks_df['Bonferroni alpha'] = 0.05 / m
    ks_df['Adjusted Significant'] = ks_df['Raw p-value'] < ks_df['Bonferroni alpha']
    ks_df['Adjusted p-value'] = ks_df['Raw p-value'] * m
    ks_df['Adjusted p-value'] = ks_df['Adjusted p-value'].clip(upper=1.0)  # Ensure max 1.0

    return ks_df



def extract_ip_pair(flow_id):
    try:
        src_ip, dst_ip, *_ = flow_id.split('-')
        return src_ip, dst_ip
    except:
        return None, None

def build_info_flow_graph(df, flow_column='Flow ID'):
    # Step 1: Extract source and destination IPs
    df[['SrcIP', 'DstIP']] = df[flow_column].apply(lambda x: pd.Series(extract_ip_pair(x)))
    
    # Step 2: Create directed graph
    G = nx.DiGraph()
    
    # Step 3: Add edges for each flow
    for _, row in df.iterrows():
        if pd.notna(row['SrcIP']) and pd.notna(row['DstIP']):
            G.add_edge(row['SrcIP'], row['DstIP'])
    
    return G

def plot_info_flow_graph(G, max_nodes=100):

    if len(G.nodes) > max_nodes:
        print(f"Showing only {max_nodes}.")
        G = G.subgraph(list(G.nodes)[:max_nodes])

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)  # layout with spacing

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',    \
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'  \
    )

    plt.title("Directed Information Flow Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def main():
    
    #  ==== Setup ====
    file_path = "DarknetWorking.csv"
    selected_columns = [
        'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Packet Length Std', 'Idle Mean'
    ]

    variable_scales = {col: 'ratio' for col in selected_columns}

    # ==== Load and clean ====
    df = load_data(file_path, selected_columns)
    analyze_missing_values(df)
    print_missing_rows(df)
    df = clean_data(df)
    
    # ==== Analysis and Visualization ====
    plot_correlation_heatmap(df, selected_columns)
    plot_density_per_column(df)
    describe_label_distribution(df, label_col='Label')
    describe_variables(df, variable_scales)
    
    # Count how many fully duplicated rows exist
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    # Remove duplicate rows
    remove_duplicate_rows(df)
        
    # ==== Outlier Detection and Analysis ====
    df = apply_outlier_flagging(df, selected_columns)
    output_path = "DarknetWorking_with_outliers.csv"
    df.to_csv(output_path, index=False)
    analyze_outlier_overlap(df, [f"{col}_Outlier" for col in selected_columns])
    
    # ==== PCA analysis====
    variance_df, loadings_df, pc_scores_df = pca_spss_style(df, selected_columns, n_components=3)
    pc_scores_df.index = df.index  # Ensure alignment with df

    # ==== Clustering Analysis ====
    plot_elbow_method(pc_scores_df, max_k=10)
    labels, kmeans_model = run_kmeans_clustering(pc_scores_df, n_clusters=3)
    df['Cluster'] = labels
    df.to_csv("DarknetWorking_with_outliers.csv", index=False)
    plot_3d_clusters_sampled(pc_scores_df, df['Cluster'], sample_size=5000)
    
    
    summary_df = interpret_clusters(df, selected_columns)
    summary_df.to_csv("cluster_summary.csv", index=True)
    plot_normalized_cluster_mean_heatmap(df, selected_columns, cluster_col='Cluster')
    
    # ==== Temporal feature engineering ====
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # remove rows with bad timestamp
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
    df['WeekGroup'] = df['DayOfWeek'].apply(lambda x: 'DuringWeek' if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] else 'Weekend')
    df['PartOfDay'] = pd.cut(
        df['Hour'],
        bins=[6, 12, 18, 24],
        labels=['Morning', 'Afternoon', 'Evening'],
        right=False
    )
    
    # ==== Temporal Analysis ====
    temporal_analysis("DarknetWorking_with_outliers.csv", timestamp_col="Timestamp", cluster_col="Cluster")
    
    temporal_ks = run_temporal_ks_tests(df, selected_columns, time_col='PartOfDay', cluster_col='Cluster')
    temporal_ks.to_csv("temporal_ks_results.csv", index=False)
    
    temporal_ks = run_temporal_ks_tests(df, selected_columns, time_col='WeekGroup', cluster_col='Cluster')
    temporal_ks.to_csv("temporal_ks_resultsday.csv", index=False)
    
    # ==== Visualize Temporal Differences ====
    sns.kdeplot(data=df[df['Cluster'].isin([0,1]) & (df['PartOfDay'] == 'Morning')], x='Idle Mean', hue='Cluster')
    plt.title("Idle Mean Distribution in the Morning (Cluster 0 vs 1)")
    plt.xlabel("Idle Mean")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
    
    
    sns.kdeplot(data=df[df['Cluster'].isin([0,1]) & (df['WeekGroup'] == 'DuringWeek')], x='Bwd Packet Length Mean', hue='Cluster')
    plt.title("Bwd Packet Length Mean Distribution in the Weekend (Cluster 0 vs 1)")
    plt.xlabel("Bwd Packet Length Mean")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
    
    # ==== Information Flow Graph ====
    G = build_info_flow_graph(df)
    plot_info_flow_graph(G)
    
if __name__ == "__main__":
    main()
