import pandas as pd
from neuprint import Client
from neuprint import NeuronModel
from neuprint import fetch_roi_hierarchy 
from neuprint import NeuronCriteria as NC 
from neuprint import fetch_neurons 
from neuprint import fetch_adjacencies, NeuronCriteria as NC 
from neuprint import merge_neuron_properties
from neuprint.utils import connection_table_to_matrix
import holoviews as hv
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
import seaborn as sns


TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im5vZWxsZS5lZ2hiYWxpQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSUNtblhJeEJJbG85TVEzanZ1NmZnWlduVVg4dEoySjlybDlpdWw2emJBbGhrPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODc2MjU2MTMxfQ.yzW_Z8fcrJ8o2d6c_eioiBNAfxrSSobM_eHy4468I4w'
c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', TOKEN)
c.fetch_version()

'" Overview of ROI hierarchy "'
#print(fetch_roi_hierarchy(False, mark_primary=True, format = 'text'))

'" Determine neuron search criteria "'
# criteria = NC(bodyId = ) # Select by Id
criteria = NC(type = 'MBON.*') # Select by type, or all bodies by type name pattern
#criteria = NC(rois = ) # Select by region ex. CX

'" Fetch neuron properties "'
neuron_df, roi_counts_df = fetch_neurons(criteria)
#print(neuron_df)
#print(roi_counts_df)

'" Fetch connections --> connection table "'
# Example: Fetch all downstream connections FROM a set of neurons
#neuron_df, conn_df = fetch_adjacencies([387023620, 387364605, 416642425], None)
# Example: Fetch all upstream connections TO a set of neurons
#neuron_df, conn_df = fetch_adjacencies(None, [387023620, 387364605, 416642425])
# Example: Fetch all direct connections between a set of upstream neurons and downstream neurons
neuron_df, conn_df = fetch_adjacencies(NC(type='MBON21'), NC(type='FB.*'))
conn_df.sort_values('weight', ascending = False)

'" Merge neuron properties to a connection table "'
conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
#print(conn_df)

'" Convert a connection table into a connectivity matrix "'
#matrix = connection_table_to_matrix(conn_df, 'bodyId', sort_by='type')
matrix = conn_df.pivot_table(index = 'bodyId_pre', columns = 'bodyId_post', values = 'weight', fill_value = None)
matrix = matrix.fillna(0)
matrix.index = matrix.index.astype(str)
matrix.columns = matrix.columns.astype(str)
heatmap = matrix.hvplot.heatmap(height=700, width=1000, xaxis='top').opts(xrotation=60)
hv.render(heatmap)

''' Top outputs ex. MBON30 '''
# Fetch neuron properties for MBON21 and FB.*
mbon_neurons, _ = fetch_neurons(NC(type='MBON30'))
fb_neurons, _ = fetch_neurons(NC(type='FB.*'))
# Fetch the adjacencies
neuron_df, conn_df = fetch_adjacencies(NC(type='MBON30'), NC(type='FB.*'))
conn_df.sort_values('weight', ascending=False, inplace=True)
# Merge the downstream neuron properties to get the neuron types
fb_neurons = fb_neurons[['bodyId', 'type']]
fb_neurons.columns = ['bodyId_post', 'type_post']
conn_df = conn_df.merge(fb_neurons, left_on='bodyId_post', right_on='bodyId_post', how='left')
# Group by downstream type and get the highest weight synapse for each type
highest_weight_synapses = conn_df.loc[conn_df.groupby('type_post')['weight'].idxmax()]
# Filter out synapses with weight < 10
highest_weight_synapses = highest_weight_synapses[highest_weight_synapses['weight'] >= 10]
# Sort the result by weight in descending order
highest_weight_synapses = highest_weight_synapses.sort_values(by='weight', ascending=False)
# Display the result as a table
print("Highest weight synapses from MBON21 onto different FB types (weight >= 10):")
print(highest_weight_synapses)

''' Top outputs heatmap '''
# Fetch neuron properties for MBON30 and FB.*
mbon_neurons, _ = fetch_neurons(NC(type='MBON09'))
fb_neurons, _ = fetch_neurons(NC(type='FB.*'))
# Fetch the adjacencies
neuron_df, conn_df = fetch_adjacencies(NC(type='MBON09'), NC(type='FB.*'))
conn_df.sort_values('weight', ascending=False, inplace=True)
# Merge the downstream neuron properties to get the neuron types
fb_neurons = fb_neurons[['bodyId', 'type']]
fb_neurons.columns = ['bodyId_post', 'type_post']
conn_df = conn_df.merge(fb_neurons, left_on='bodyId_post', right_on='bodyId_post', how='left')
# Sum the weights of the synaptic connections from all MBON30 neurons to each FB.* type
aggregated_weights = conn_df.groupby('type_post')['weight'].sum().reset_index()
# Filter out synapses with weight < 10
aggregated_weights = aggregated_weights[aggregated_weights['weight'] >= 10]
# Sort the result by weight in descending order
aggregated_weights = aggregated_weights.sort_values(by='weight', ascending=False)
# Create a heatmap DataFrame
heatmap_data = aggregated_weights.pivot_table(index='type_post', values='weight', aggfunc='sum').sort_values(by='weight', ascending=False)
# Plot the heatmap
plt.figure(figsize=(4, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', linewidths=.5, fmt='g')  # 'g' to disable scientific notation
plt.title('Aggregated Synaptic Weights from MBON30 onto Different FB Types')
plt.xlabel('Presynaptic Neuron')
plt.ylabel('FB Neuron Type')
# Ensure the directory exists
output_dir = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/spring_2024_exp/analysis/connectome"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "aggregated_mbon09_to_fb_heatmap.pdf")
# Save the heatmap as a PDF
plt.tight_layout()
plt.savefig(output_path, format='pdf')
plt.show()

''' Top MBON to CX outputs bar plot'''
# Fetch neuron properties for MBONs and CX neurons
mbon_neurons, _ = fetch_neurons(NC(type='MBON.*'))  # Fetch all MBON types
cx_neurons, _ = fetch_neurons(NC(type=['FB.*', 'EB.*', 'PB.*', 'NO.*', 'AB(L).*', 'AB(R).*']))    # Fetch all CX types
# Fetch the adjacencies from MBONs to CX neurons
neuron_df, conn_df = fetch_adjacencies(NC(type='MBON.*'),NC(rois ='CX'))
# Sum the weights of the synaptic connections from each MBON to CX neurons
mbon_to_cx_weights = conn_df.groupby('bodyId_pre')['weight'].sum().reset_index()
# Merge with MBON neuron types to get readable labels
mbon_neurons = mbon_neurons[['bodyId', 'type']]
mbon_neurons.columns = ['bodyId_pre', 'type_pre']
mbon_to_cx_weights = mbon_to_cx_weights.merge(mbon_neurons, on='bodyId_pre', how='left')
# Sort by total output weight in descending order to identify MBONs with the most outputs to CX neurons
mbon_to_cx_weights = mbon_to_cx_weights.sort_values(by='weight', ascending=False)
aggregated_mbon_to_cx_weights = mbon_to_cx_weights.groupby('type_pre')['weight'].sum().reset_index()
aggregated_mbon_to_cx_weights = aggregated_mbon_to_cx_weights.sort_values(by='weight', ascending=False)
# Display the result
aggregated_mbon_to_cx_weights
# Save the result as a CSV for further analysis or visualization
output_dir = "/Users/noelleeghbali/Desktop/exp/tethered_behavior/spring_2024_exp/analysis/connectome"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "aggregated_mbon_to_cx_weights.csv")
aggregated_mbon_to_cx_weights.to_csv(output_path, index=False)
# Plot the top 10 MBONs with the most outputs onto CX neurons
top_n = 10
top_aggregated_mbon_to_cx_weights = aggregated_mbon_to_cx_weights.head(top_n)
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.barh(top_aggregated_mbon_to_cx_weights['type_pre'], top_aggregated_mbon_to_cx_weights['weight'], color='skyblue')
plt.xlabel('Total Synaptic Weight')
plt.ylabel('MBON Type')
plt.title('Top 10 MBONs with Most Outputs onto Central Complex Neurons')
plt.gca().invert_yaxis()  # Highest weights at the top
plt.tight_layout()
# Further customization
axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
for pos in ['right', 'top']:
    axs.spines[pos].set_visible(False)
plt.tight_layout()
sns.despine(offset=10)
for _, spine in axs.spines.items():
    spine.set_linewidth(2)
for spine in axs.spines.values():
    spine.set_edgecolor('black')
# Save the plot as a PDF
plot_output_path = os.path.join(output_dir, "top_aggregated_mbon_to_cx_outputs.pdf")
plt.savefig(plot_output_path, format='pdf')
plt.show()
