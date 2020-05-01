{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#importing dependancies\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.datasets import load_boston"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#understanding the dataset\n",
    "boston=load_boston()\n",
    "print(boston.DESCR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#access data attributes\n",
    "dataset = boston.data\n",
    "for name index is enumerate(boston.feature_names):\n",
    "    print(index, name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reshaping data\n",
    "data = dataset[:,12],reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#shape of the data\n",
    "np.shape(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#target values\n",
    "target = boston.target.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#shape of target\n",
    "np.shape(target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ensuring that matplotlib is working\n",
    "%matplotlib inline\n",
    "plt.scatter(data,target,color='green')\n",
    "plt.xlabel('Lower income population')\n",
    "plt.ylabel('Cost of House')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#regression\n",
    "from sklearn.linear_model import LinearRegression\n",
    "#creating a regression model\n",
    "reg=LinearRegression()\n",
    "#fit the model\n",
    "reg.fit(data, target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#prediction\n",
    "pred = reg.predict(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ensuring matplotlib is working\n",
    "%matplotlib inline\n",
    "plt.scatter(data, target, color='red')\n",
    "plt.xlabel('Lower income population')\n",
    "plt.ylabel('Cost of House')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#circumventing curve issue using polynomial model\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "#to allow merging models\n",
    "from sklearn.pipeline import make_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = make_pipeline(PloynomialFeatures(3),reg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(data, target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred=model.predict(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ensuring matplotlib is working\n",
    "%matplotlib inline\n",
    "plt.scatter(data, target, color='red')\n",
    "plt.xlabel('Lower income population')\n",
    "plt.ylabel('Cost of House')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#r_2 metric\n",
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict\n",
    "r2_score(pred,target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
