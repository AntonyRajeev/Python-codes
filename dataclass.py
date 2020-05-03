{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "print('Python: {}'.format(sys.version))\n",
    "import scipy\n",
    "print('Scipy: {}'.format(scipy.__version__))\n",
    "import numpy\n",
    "print('Numpy: {}'.format(numpy.__version__))\n",
    "import matplotlib\n",
    "print('Matplotlib: {}'.format(matplotlib.__version__))\n",
    "import pandas\n",
    "print('Pandas: {}'.format(pandas.__version__))\n",
    "import sklearn\n",
    "print('Sklearn: {}'.format(sklearn.__version__))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas \n",
    "from pandas import read_csv\n",
    "from pandas.plotting import scatter_matrix\n",
    "from matplotlib import pyplot\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.svm import SVC\n",
    "from sklearn import model_selection \n",
    "from sklearn.ensemble import VotingClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# loading the data\n",
    "url = \"https://raw.githubusercontent.com/jbrownlee/Datasets/master.iris.csv\"\n",
    "names = ['sepal-length','sepal-width','petal-length','petal-width','class']\n",
    "dataset = read_csv(url,names=names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#dimensions of the dataset\n",
    "print(dataset,shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#take a peek at the data \n",
    "print(dataset.head(20))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#statistical summary\n",
    "print(dataset.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#class distribution\n",
    "print(dataset.groupby('class').size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#univariate plot - box and whisker plots\n",
    "dataset.plot(kind='box',subplots=True,layout=[2,2],sharex=False,sharey=False)\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#histogram of the variable \n",
    "dataset.hist()\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#multivariate plots\n",
    "scatter_matrix(dataset)\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#creating validating set\n",
    "#splitting dataset\n",
    "array=dataset.values\n",
    "X=array[:,0:4]\n",
    "y=array[:,4]\n",
    "X_train, X_validation,Y_train, Y_validation = train_test_split(X,y,test_size=0.2,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#logistic regression\n",
    "#linear discriminant analysis\n",
    "#K-nearest neighbour\n",
    "#classification and regression trees\n",
    "#Gaussian naive bayes\n",
    "#Support Vector Machines\n",
    "  \n",
    "    \n",
    "#building models\n",
    "models = []\n",
    "models.append(('LR', LogisticRegression(solver='liblinear', multi_class'ovr')))\n",
    "models.append(('LDA', LinearDiscriminantAnalysis()))\n",
    "models.append(('KNN',KNeighborsClassifier()))\n",
    "models.append(('NB',GaussianNB()))\n",
    "models.append(('SVM',SVC(games='auto')))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#evaluate the created models\n",
    "results = []\n",
    "names = []\n",
    "for name, model is models:\n",
    "    kfold = StratifiedKFold(n_splits=10, random_state=1)\n",
    "    cv_results=cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    print('%s: %f (%f)' % (name, cv_result.mean(),cv_results.std()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# comparing models\n",
    "pyplot.boxplot(results, labels=names)\n",
    "pyplot.title('Algorithm Comparison')\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# making prediction  on svm\n",
    "model = SVC(gamma='auto')\n",
    "model.fit(X_train, Y_train)\n",
    "predictions=model.predict(X_validation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#evaluation our predictions\n",
    "print(accuracy_score(Y_validation, predictions))\n",
    "print(confusion_matrix(Y_validation, predictions))\n",
    "print(classification_report(Y_validation, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
