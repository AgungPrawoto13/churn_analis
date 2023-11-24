import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def visualize_value_based_clustering(data, k=3):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(data_scaled)

    data["Cluster"] = kmeans.labels_

    fig, axes = plt.subplots(1, len(data.columns)-1, figsize=(16, 5))

    for i, column in enumerate(data.columns[:-1]):
        sns.boxplot(ax=axes[i], data=data, x='Cluster', y=column)
        axes[i].set_title(column)

    fig.tight_layout()

    return fig

def visualize_needs_based_clustering(data, k=2, num_top_itemsets=10):
    # Preprocess the data
    for col in data.columns:
        data[col] = data[col].replace({"Yes": 1, "No": 0})

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(data_scaled)

    # Add cluster labels to the data
    result = data.copy()
    result["label"] = kmeans.labels_

    for col in data.columns:
        result[col] = result[col].replace({1: "Yes", 0: "No"})

    # Visualize Apriori results for each cluster
    fig_list = []
    for cluster_label in range(k):
        cluster_data = result[result["label"] == cluster_label].drop("label", axis=1).reset_index(drop=True)

        transactions = []
        for i in range(len(cluster_data)):
            x = cluster_data.iloc[i]
            product_used = [product_name for product_name, usage in zip(x.index, x) if usage == "Yes"]
            transactions.append(product_used)

        encoder = TransactionEncoder()
        one_hot = encoder.fit(transactions).transform(transactions)
        one_hot_df = pd.DataFrame(one_hot, columns=encoder.columns_)

        apriori_result = apriori(one_hot_df, min_support=1e-5, use_colnames=True, verbose=1)

        # Sort Apriori results by support in descending order
        apriori_result = apriori_result.sort_values(by='support', ascending=False).reset_index(drop=True)

        # Select the top itemsets
        top_itemsets = apriori_result.head(num_top_itemsets)

        # Plotting the horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_itemsets['itemsets'].astype(str), top_itemsets['support'], color='skyblue')
        ax.set_xlabel('Support')
        ax.set_ylabel('Itemsets')
        ax.set_title(f'Top {num_top_itemsets} Itemsets with Highest Support (Cluster {cluster_label})')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # Display the support values on the bars
        for index, value in enumerate(top_itemsets['support']):
            ax.text(value, index, f'{value:.5f}', va='center')

        fig_list.append(fig)

    return fig_list
