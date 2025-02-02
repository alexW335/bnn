import math
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import OneHotEncoder

from bnn.generator import DataGenerator
from bnn.schemas import SimulatedDataSchema
from bnn.models import BayesianShipmentModel

def generate_pairplots(data: SimulatedDataSchema):
    # get unique products
    unique_products = sorted(data['product'].unique())
    n_products = len(unique_products)
    # determine grid dimensions (roughly square)
    n_cols = math.ceil(math.sqrt(n_products))
    n_rows = math.ceil(n_products / n_cols)
    
    # create an outer figure with subplots for each product
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    outer_gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.3, hspace=0.3)
    
    # iterate over products to create a jointplot style grid for each
    for idx, prod in enumerate(unique_products):
        row = idx // n_cols
        col = idx % n_cols
        # create an inner gridspec for the joint plot:
        # layout: top row for the marginal histogram of weight,
        # bottom row for the scatter and marginal histogram of volume
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_gs[row, col],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.05,
            hspace=0.05
        )
        # create axes: top left for weight histogram, bottom left for scatter,
        # bottom right for volume histogram; top right is unused
        ax_hist_x = fig.add_subplot(inner_gs[0, 0])
        ax_scatter = fig.add_subplot(inner_gs[1, 0])
        ax_hist_y = fig.add_subplot(inner_gs[1, 1])
        ax_empty = fig.add_subplot(inner_gs[0, 1])
        ax_empty.axis('off')  # turn off the unused subplot
        
        # subset data for the current product
        sub_data = data[data['product'] == prod]
        
        # plot the scatter of weight vs volume on the main axis
        ax_scatter.scatter(sub_data['weight'], sub_data['volume'], alpha=0.5)
        
        # plot the marginal histograms
        ax_hist_x.hist(sub_data['weight'], bins=20, color='grey')
        ax_hist_y.hist(sub_data['volume'], bins=20, orientation='horizontal', color='grey')
        
        # set labels on the main axes only when appropriate
        ax_scatter.set_xlabel('weight')
        ax_scatter.set_ylabel('volume')
        # add a title to identify the product (aligned to the left)
        ax_scatter.set_title(f'product {prod}', loc='left')
        
        # remove redundant tick labels on marginal axes
        plt.setp(ax_hist_x.get_xticklabels(), visible=False)
        plt.setp(ax_hist_y.get_yticklabels(), visible=False)
    
    plt.show(block=True)

def main():
    # create the data generator and generate the data
    gen = DataGenerator(
        p_bern=np.array([0.3, 0.7]),
        lambda_s=np.array([3, 2]),
        mu=np.array([[2, 3], [6, 5]]),
        cov=np.array([[[0.04, 0.048], [0.048, 0.09]], [[0.06, 0.07], [0.07, 0.09]]])
    )

    data = gen.generate(samples_per_product=1000)

    # generate faceted pairplots by product
    generate_pairplots(data)

    # generate overall pairplots
    sns.pairplot(data[['weight', 'volume']], diag_kind='hist')
    plt.show(block=True)

    enc = OneHotEncoder()
    enc.fit(data[['product']])
    trans_prod = enc.transform(data[['product']])

    X = trans_prod.toarray()
    y = data[['weight', 'volume']].to_numpy()

    bsm = BayesianShipmentModel(input_dim = 2, kl_weight=0)
    bsm.fit(X, y, epochs=10)

    n_new = 1000
    newdata = np.tile(np.eye(2), (n_new, 1))
    
    preds = bsm.predict(newdata)

    learned_data = SimulatedDataSchema({
        'product': 1-newdata[:,0],
        'weight': preds[:,1].flatten(),
        'volume': preds[:,1].flatten()
    })
    
    generate_pairplots(learned_data)




if __name__ == "__main__":
    main()
