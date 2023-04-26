import os
import ast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from source.report.report_utils import preprocess_results

# Preprocessing
episodes, config = preprocess_results('classicalCartPole-v1')
hyperparameters_df = pd.DataFrame(hyperparameters).T.reset_index()
hyperparameters_df.columns = ["trial_id", "layers", "learning_rate", "neurons"]
data = episodes.merge(hyperparameters_df, on="trial_id")
max_evaluation_score = data.groupby(["trial_id", "run_id"]).agg({"evaluation_score": "max"}).reset_index()
merged_data = max_evaluation_score.merge(hyperparameters_df, on="trial_id")

output_folder = "plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Heatmap 1
heatmap_data = merged_data.groupby(["layers", "neurons"]).agg({"evaluation_score": "mean"}).reset_index()
heatmap_data_pivot = heatmap_data.pivot("layers", "neurons", "evaluation_score")
plt.figure()
sns.heatmap(heatmap_data_pivot, cmap="YlGnBu", annot=True)
plt.title("Mean Max Evaluation Score by Layers and Neurons")
plt.savefig(os.path.join(output_folder, "mean_max_evaluation_score_by_layers_and_neurons.png"))
plt.show()

# Heatmap 2
heatmap_data_lr_layers = merged_data.groupby(["learning_rate", "layers"]).agg(
    {"evaluation_score": "mean"}).reset_index()
heatmap_data_lr_layers_pivot = heatmap_data_lr_layers.pivot("learning_rate", "layers", "evaluation_score")
plt.figure()
sns.heatmap(heatmap_data_lr_layers_pivot, cmap="YlGnBu", annot=True)
plt.title("Mean Max Evaluation Score by Learning Rate and Layers")
plt.savefig(os.path.join(output_folder, "mean_max_evaluation_score_by_learning_rate_and_layers.png"))
plt.show()

# Heatmap 3
heatmap_data_lr_neurons = merged_data.groupby(["learning_rate", "neurons"]).agg(
    {"evaluation_score": "mean"}).reset_index()
heatmap_data_lr_neurons_pivot = heatmap_data_lr_neurons.pivot("learning_rate", "neurons", "evaluation_score")
plt.figure()
sns.heatmap(heatmap_data_lr_neurons_pivot, cmap="YlGnBu", annot=True)
plt.title("Mean Max Evaluation Score by Learning Rate and Neurons")
plt.savefig(os.path.join(output_folder, "mean_max_evaluation_score_by_learning_rate_and_neurons.png"))
plt.show()

# Input-output correlation heatmap
episodes['evaluation_observations'] = episodes['evaluation_observations'].apply(ast.literal_eval)
episodes['evaluation_predictions'] = episodes['evaluation_predictions'].apply(ast.literal_eval)

observations_flat = []
predictions_flat = []

for i, row in episodes.iterrows():
    observations_flat.extend(row['evaluation_observations'])
    predictions_flat.extend(row['evaluation_predictions'])

observations_df = pd.DataFrame(observations_flat, columns=[f'Feature {i + 1}' for i in range(4)])
predictions_df = pd.DataFrame(predictions_flat, columns=[f'Output {i + 1}' for i in range(2)])

input_output_df = pd.concat([observations_df, predictions_df], axis=1)

correlation_matrix = input_output_df.corr()

feature_output_correlations = correlation_matrix.loc[
    [f'Feature {i + 1}' for i in range(4)], [f'Output {i + 1}' for i in range(2)]]

plt.figure(figsize=(8, 6))
sns.heatmap(feature_output_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Coefficients between Input Features and Outputs')
plt.xlabel('Outputs')
plt.ylabel('Input Features')
plt.savefig(os.path.join(output_folder, "correlation_coefficients_between_input_features_and_outputs.png"))
plt.show()

# Create scatter plots for each input feature and output combination
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(4):
    for j in range(2):
        sns.scatterplot(ax=axes[i, j], data=input_output_df, x=f'Feature {i + 1}', y=f'Output {j + 1}')
        axes[i, j].set_title(f'Feature {i + 1} vs Output {j + 1}')

plt.show()
