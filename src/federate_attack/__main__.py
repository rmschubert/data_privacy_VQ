import logging
import os

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

from .attack import attack_vq
from .glvq import PlainGLVQ
from .plot_imgs import create_visible_results

trainset_size = 1000

## make results directory in current working directory
cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.exists(results_path):
    os.mkdir(results_path)

logging.info("Fetching Data")
## Load data from https://www.openml.org/d/554
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
)
logging.info("Fetching done")

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

## We do not consider test sets here
X_train, _, y_train, _ = train_test_split(X, y, train_size=trainset_size, test_size=1)

## Scale data into [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


## initialize prototypes as noisy k-means centers
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_train)
p_t = kmeans.cluster_centers_ + np.random.rand(10, X_train.shape[1])

## initialize training routine
all_protos = [p_t]
pc = [f"{i}" for i in range(10)]
lr = 1.0
glvq = PlainGLVQ()
accs = [0.0]
epochs = 100
results = {
    "Images": [],
    "Approximations": [],
    "MSE": [],
    "SSIM": [],
    "Accuracy": [],
}

## training + attack
logging.info("Run training and Attack")
for e in range(epochs):
    train_preds = []

    ## pick random index from train_set
    img_ind = int(np.random.choice(trainset_size))

    for i, (x, Y) in enumerate(zip(X_train, y_train)):
        ## get current prototypes and run update
        p_t = all_protos[-1]
        mags, grads, p_t1 = glvq(p_t, pc, x, Y, lr=lr)

        ## check if current index matches the chosen one
        ## and if then run the attack
        if i == img_ind:
            approximation = attack_vq(grads, p_t, p_t1)

            if approximation is not None:
                x_img = x.reshape((28, 28))
                results["Images"].append(x_img)
                results["Approximations"].append(approximation)
                mses = []
                ssims = []
                for res in approximation:
                    x_app = res[-1].reshape((28, 28)).clip(0, 1)
                    mse = np.mean((x_img - x_app) ** 2)
                    ssim = structural_similarity(x_img, x_app, data_range=1.0)
                    mses.append(mse)
                    ssims.append(ssim)
                results["MSE"].append(mses)
                results["SSIM"].append(ssims)

        ## make prediction on update for accuracy
        ## and store updated prototypes
        train_preds.append(glvq(p_t1, pc, x, mode="test"))
        all_protos[-1] = p_t1

    epoch_acc = np.mean(y_train == train_preds)
    results["Accuracy"].append(epoch_acc)

logging.info("Training and Attack Done, Now creating results")
## save results
res_df = pd.DataFrame.from_dict(results)
res_path = os.path.join(results_path, "results.pkl")
res_df.to_pickle(res_path)

create_visible_results(res_path=res_path)

logging.info("Done")
