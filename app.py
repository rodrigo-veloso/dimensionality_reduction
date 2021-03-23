
import streamlit as st

st.sidebar.write("""## Choose a Page""")
page = st.sidebar.radio('Pages', ['Dimensionality Reduction Examples','Visual Comparison'])

if page == 'Dimensionality Reduction Examples':

    st.write("""# Dimensionality Reduction Examples""")

    st.write("""### Examples of dimensionality reduction using the hermione framework from A3Data""")
    st.write("""### https://github.com/A3Data/hermione""")

    st.image('vertical_logo.png', width = 600)

    st.write("""## Import:""")
    st.write("""### All necessary modules""")

    with st.echo():
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        from ml.data_source.spreadsheet import Spreadsheet
        from ml.preprocessing.preprocessing import Preprocessing
        from ml.preprocessing.feature_selection import FeatureSelector
        from ml.preprocessing.dimensionality_reduction import DimensionalityReducer
        #from ml.model.trainer import TrainerSklearn
        from ml.preprocessing.normalization import Normalizer

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVR
        from sklearn.svm import LinearSVC
        from mpl_toolkits.mplot3d import Axes3D

    st.write("""### The data""")
    with st.echo():
        df = Spreadsheet().get_data('train.csv',columns=['Survived','Pclass','Sex','Age'])

    st.write(df.head(5))

    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    df_ = df.copy()
    df_['Sex'] = X['Sex_male']
    class_0 = df_[df_['Survived']==0]
    class_1 = df_[df_['Survived']==1]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(class_0['Pclass'],class_0['Sex'],class_0['Age'])
    ax.scatter(class_1['Pclass'],class_1['Sex'],class_1['Age'])

    st.pyplot(plt)

    st.write("""## Defining preprocess function:""")
    with st.echo():
        def preprocess(df, norm_cols = {}):
            p = Preprocessing()
            n = Normalizer(norm_cols = norm_cols)
            df = p.clean_data(df)
            X = df.drop(columns=['Survived'])
            X = p.categ_encoding(X)
            X = n.fit_transform(X)
            return X

    def vis_support(df,X):
        df_ = df.copy()
        df_ = df_.drop(columns=['Pclass','Age','Sex'])
        df_['1'] = X_t[:,0]
        df_['2'] = X_t[:,1]
        class_0 = df_[df_['Survived']==0]
        class_1 = df_[df_['Survived']==1]
        return class_0,class_1

    def plot(cls1,cls2,cls3 = None, cls4 = None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(cls3, pd.DataFrame) else 1
        ax = fig.add_subplot(1, num_plots, 1)
        ax.scatter(cls1['1'],cls1['2'])
        ax.scatter(cls2['1'],cls2['2'])
        if isinstance(cls3, pd.DataFrame):
            ax = fig.add_subplot(1, num_plots, 2)
            ax.scatter(cls3['1'],cls3['2'])
            ax.scatter(cls4['1'],cls4['2'])

    def plot3d(cls1,cls2,cls3=None,cls4=None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(cls3, pd.DataFrame) else 1
        ax = fig.add_subplot(1, num_plots, 1,projection='3d')
        ax.scatter(cls1['1'],cls1['2'],cls1['3'])
        ax.scatter(cls2['1'],cls2['2'],cls2['3'])
        if isinstance(cls3, pd.DataFrame):
            ax = fig.add_subplot(1, num_plots, 2,projection='3d')
            ax.scatter(cls3['1'],cls3['2'],cls3['3'])
            ax.scatter(cls4['1'],cls4['2'],cls4['3'])

    def get_3d(**kwargs):
        plt.clf()
        d = DimensionalityReducer(**kwargs)
        X = preprocess(df)
        X_t = d.fit_transform(X)
        df_['1'] = X_t[:,0]
        df_['2'] = X_t[:,1]
        df_['3'] = X_t[:,2]
        class_0 = df_[df_['Survived']==0]
        class_1 = df_[df_['Survived']==1]
        plot3d(class_0, class_1)
        st.pyplot(plt)
    #st.write("""It removes all features which variance doesn’t meet the threshold. By default threshold = 0, features with zero variance are features that have the same value in all samples.""")

    st.write("""## Factor Analysis""")

    st.write("""### 2 dimensions example""")

    #st.selectbox("""Compare with: """, ["PCA","ICA","Factor Analysis","Locally Linear Embedding","Modified Locally Linear Embedding","Hessian Eigenmapping","Spectral Embedding","Local Tangent Space Alignment","Multi-dimensional Scaling","Isomap","t-distributed Stochastic Neighbor Embedding","UMAP: Uniform Manifold Approximation and Projection"])

    #dic = {"PCA":{""},"ICA","Factor Analysis","Locally Linear Embedding","Modified Locally Linear Embedding","Hessian Eigenmapping","Spectral Embedding","Local Tangent Space Alignment","Multi-dimensional Scaling","Isomap","t-distributed Stochastic Neighbor Embedding","UMAP: Uniform Manifold Approximation and Projection"}

    plt.clf()
    with st.echo():
        d = DimensionalityReducer('factor_analysis', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "factor_analysis", n_components = 3)

    st.write("""## Principal Comonent Analysis""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('pca', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "pca", n_components = 3)

    st.write("""## Independent Component Analysis""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('ica', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "ica", n_components = 3)

    #st.write("""## Latent Dirichlet Allocation""")

    #st.write("""### 2 dimensions example""")
    #plt.clf()
    #with st.echo():
    #    d = DimensionalityReducer('lda', n_components=2)
    #    X = preprocess(df)
    #    d.fit(X)
    #    X_t = d.transform(X)
    #class_0, class_1 = vis_support(df,X)
    #plot(class_0,class_1)

    #st.pyplot(plt)

    #st.write("""### 3 dimensions example""")
    #get_3d(reducer = "lda", n_components = 3)

    st.write("""## Truncated SVD""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('truncated_svd', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "truncated_svd", n_components = 3)

    st.write("""## Non-Negative Matrix Factorization (NMF)""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('nmf', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "nmf", n_components = 3)

    st.write("""## Locally Linear Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 5, n_components=2)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "locally_linear_embedding", n_components = 3)

    st.write("""## Modified Locally Linear Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 15, n_components=2, method = 'modified')
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 15, method = 'modified')

    st.write("""## Hessian Eigenmapping""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 20, n_components=2, method = 'hessian')
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 20, method = 'hessian')

    #############################################################################################################

    st.write("""## Spectral Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('spectral_embedding', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        X_t = d.fit_transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = 'spectral_embedding', n_components = 3)

    #############################################################################################################

    st.write("""## Local Tangent Space Alignment""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 18, n_components=2, method = 'ltsa')
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        X_t = d.fit_transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 18, method = 'ltsa')

    #############################################################################################################

    st.write("""## Multi-dimensional Scaling""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('mds', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        X_t = d.fit_transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "mds", n_components = 3)

    #############################################################################################################

    st.write("""## Isomap""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('isomap', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        X_t = d.fit_transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "isomap", n_components = 3)

    #############################################################################################################

    st.write("""## t-distributed Stochastic Neighbor Embedding (t-SNE)""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('tsne', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        X_t = d.fit_transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "tsne", n_components = 3)

    #############################################################################################################

    st.write("""## Uniform Manifold Approximation and Projection (UMAP)""")

    st.write(""" The details for the underlying mathematics can be found in: https://arxiv.org/abs/1802.03426""")
    st.write(""" To learn even more: https://umap-learn.readthedocs.io/en/latest/""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('umap', n_components=2, n_neighbors = 15, min_dist = 0.1)
        X = preprocess(df,norm_cols = {'z-score':['Pclass','Age','Sex_male','Sex_female']})
        d.fit(X)
        X_t = d.transform(X)
    class_0, class_1 = vis_support(df,X)
    plot(class_0,class_1)

    st.pyplot(plt)

    st.write("""### 3 dimensions example""")
    get_3d(reducer = "umap", n_components = 3, n_neighbors = 15, min_dist = 0.1)

    #############################################################################################################

if page == 'Visual Comparison':

    st.write("""# Visual Comparison""")

    st.write("""### Interactive visual comparison between dimensionality reduction algorithms implemented in hermione framework from A3Data""")
    st.write("""### https://github.com/A3Data/hermione""")

    st.image('vertical_logo.png', width = 600)

    
