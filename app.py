
import streamlit as st
import time

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
        from ml.preprocessing.dimensionality_reduction import DimensionalityReducer
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
        df = Spreadsheet().get_data('iris.csv')

    st.write(df.head(5))

    def preprocess(df, norm_cols = {}):
        p = Preprocessing()
        n = Normalizer(norm_cols = norm_cols)
        df = p.clean_data(df)
        X = df.drop(columns=['class'])
        y = df['class']
        X = p.categ_encoding(X)
        X = n.fit_transform(X)
        return X

    def vis_support(df,X_, target = 'class'):
        df_ = df.copy()
        for column in df_.columns:
            if column != target:
                df_ = df_.drop(columns=[column])
        df_['1'] = X_[:,0]
        df_['2'] = X_[:,1]
        print(df_)
        class_list = []
        for label in df_[target].unique():
            class_list.append(df_[df_[target]==label])
        return class_list

    def plot(class_list1, class_list2 = None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(class_list2, list) else 1
        ax = fig.add_subplot(1, num_plots, 1)
        for cls in class_list1:
            ax.scatter(cls['1'],cls['2'])
        if isinstance(class_list2, list):
            ax = fig.add_subplot(1, num_plots, 2)
            for cls in class_list2:
                ax.scatter(cls['1'],cls['2'])
        st.pyplot(plt)

    def plot3d(class_list1, class_list2 = None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(class_list2, list) else 1
        ax = fig.add_subplot(1, num_plots, 1,projection='3d')
        for cls in class_list1:
            ax.scatter(cls['1'],cls['2'],cls['3']) 
        if isinstance(class_list2, list):
            ax = fig.add_subplot(1, num_plots, 2,projection='3d')
            for cls in class_list2:
                ax.scatter(cls['1'],cls['2'],cls['3'])

    def get_3d(X, df, y = None, target = 'class', **kwargs):
        plt.clf()
        X = preprocess(df)
        reducer = kwargs.get('reducer')
        if reducer:
            d = DimensionalityReducer(**kwargs)
        X_t = d.fit_transform(X,y) if reducer else X.values
        if 'reducer' in kwargs:
            if  kwargs['reducer'] == 'pca':
                X_t = X_t.values
        df_ = df.copy()
        for column in df_.columns:
            if column != target:
                df_ = df_.drop(columns=[column])
        df_['1'] = X_t[:,0]
        df_['2'] = X_t[:,1]
        df_['3'] = X_t[:,2]
        class_list = []
        for label in df_[target].unique():
            class_list.append(df_[df_[target]==label])
        plot3d(class_list)
        st.pyplot(plt)

    p = Preprocessing()
    df = p.clean_data(df)
    y = df['class']
    X = df.drop(columns=['class'])
    X = p.categ_encoding(X)
    get_3d(X, df)
    #st.write("""It removes all features which variance doesn’t meet the threshold. By default threshold = 0, features with zero variance are features that have the same value in all samples.""")

    st.write("""## Defining preprocess function:""")
    with st.echo():
        def preprocess(df, norm_cols = {}):
            p = Preprocessing()
            n = Normalizer(norm_cols = norm_cols)
            df = p.clean_data(df)
            X = df.drop(columns=['class'])
            y = df['class']
            X = p.categ_encoding(X)
            X = n.fit_transform(X)
            return X

    #############################################################################################################

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
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "factor_analysis", n_components = 3)

    #############################################################################################################

    st.write("""## Principal Comonent Analysis""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        X = preprocess(df)
        columns = X.columns
        d = DimensionalityReducer('pca', columns = columns)
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t.values)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "pca", columns = columns, k = 3)

    #############################################################################################################

    st.write("""## Independent Component Analysis""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('ica', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "ica", n_components = 3)

    #############################################################################################################

    st.write("""## Linear Discriminant Analysis""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('linear_discriminant', n_components=2)
        X = preprocess(df)
        y = df['class']
        d.fit(X,y)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    #############################################################################################################

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

    #############################################################################################################

    st.write("""## Truncated SVD""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('truncated_svd', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "truncated_svd", n_components = 3)

    #############################################################################################################

    st.write("""## Non-Negative Matrix Factorization (NMF)""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('nmf', n_components=2)
        X = preprocess(df)
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "nmf", n_components = 3)

    #############################################################################################################

    st.write("""## Locally Linear Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 5, n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "locally_linear_embedding", n_components = 3)

    #############################################################################################################

    st.write("""## Modified Locally Linear Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 15, n_components=2, method = 'modified')
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 15, method = 'modified')

    #############################################################################################################

    st.write("""## Hessian Eigenmapping""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 20, n_components=2, method = 'hessian')
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        d.fit(X)
        X_t = d.transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 20, method = 'hessian')

    #############################################################################################################

    st.write("""## Spectral Embedding""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('spectral_embedding', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = 'spectral_embedding', n_components = 3)

    #############################################################################################################

    st.write("""## Local Tangent Space Alignment""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('locally_linear_embedding', n_neighbors = 18, n_components=2, method = 'ltsa')
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "locally_linear_embedding", n_components = 3, n_neighbors = 18, method = 'ltsa')

    #############################################################################################################

    st.write("""## Multi-dimensional Scaling""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('mds', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "mds", n_components = 3)

    #############################################################################################################

    st.write("""## Isomap""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('isomap', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "isomap", n_components = 3)

    #############################################################################################################

    st.write("""## t-distributed Stochastic Neighbor Embedding (t-SNE)""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('tsne', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "tsne", n_components = 3)

#############################################################################################################

    st.write("""## UMAP: Uniform Manifold Approximation and Projection""")

    st.write("""### 2 dimensions example""")
    plt.clf()
    with st.echo():
        d = DimensionalityReducer('umap', n_components=2)
        X = preprocess(df,norm_cols = {'z-score':X.columns})
        X_t = d.fit_transform(X)
    class_list = vis_support(df,X_t)
    plot(class_list)

    st.write("""### 3 dimensions example""")
    get_3d(X, df, reducer = "umap", n_components = 3)

    #############################################################################################################

    #st.write("""## Uniform Manifold Approximation and Projection (UMAP)""")

    #st.write(""" The details for the underlying mathematics can be found in: https://arxiv.org/abs/1802.03426""")
    #st.write(""" To learn even more: https://umap-learn.readthedocs.io/en/latest/""")

    #st.write("""### 2 dimensions example""")
    #plt.clf()
    #with st.echo():
    #    d = DimensionalityReducer('umap', n_components=2, n_neighbors = 15, min_dist = 0.1)
    #    X = preprocess(df,norm_cols = {'z-score':X.columns})
    #    d.fit(X)
    #    X_t = d.transform(X)
    #class_list = vis_support(df,X_t)
    #plot(class_list)

    #st.write("""### 3 dimensions example""")
    #get_3d(X, df, reducer = "umap", n_components = 3, n_neighbors = 15, min_dist = 0.1)

    #############################################################################################################

if page == 'Visual Comparison':

    st.write("""# Visual Comparison""")

    st.write("""### Interactive visual comparison between dimensionality reduction algorithms implemented in hermione framework from A3Data""")
    st.write("""### https://github.com/A3Data/hermione""")

    st.image('vertical_logo.png', width = 600)

    col1, col2, col3 = st.beta_columns((3,6,6))

    col1.write("""### Choose dimensions: """)
  
    n_components = col1.radio('', ['2','3'])

    col2.write("""### Choose Algorithm: """)

    dic = {"PCA":{"reducer":"pca"},"ICA":{"reducer":"ica"},"Factor Analysis":{"reducer":"factor_analysis"},"Locally Linear Embedding":{"reducer":"locally_linear_embedding"},"Modified Locally Linear Embedding":{"reducer":"locally_linear_embedding","method":"modified","n_neighbors":18},"Hessian Eigenmapping":{"reducer":"locally_linear_embedding","method":"hessian","n_neighbors":18},"Spectral Embedding":{"reducer":"spectral_embedding"},"Local Tangent Space Alignment":{"reducer":"locally_linear_embedding","method":"ltsa","n_neighbors":18},"Multi-dimensional Scaling":{"reducer":"mds"},"Isomap":{"reducer":"isomap"},"t-SNE":{"reducer":"tsne"},"UMAP":{"reducer":"umap"}}

    options = ["PCA","ICA","Factor Analysis","Locally Linear Embedding","Modified Locally Linear Embedding","Hessian Eigenmapping","Spectral Embedding","Local Tangent Space Alignment","Multi-dimensional Scaling","Isomap","t-SNE","UMAP"]

    algo_1 =col2.selectbox("""""", options, key='1')

    col3.write("""### Compare with: """)

    algo_2 = col3.selectbox("""""", options, key='2')

    #if st.button('Advanced Options'):
    #    input1 = st.text_input('Enter parameters for '+algo_1+':',key=1)
    #    input2 = st.text_input('Enter parameters for '+algo_2+':',key=2)
    #    st.write("""Example for t-SNE: learning_rate=200.0, n_iter=1000""")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from ml.data_source.spreadsheet import Spreadsheet
    from ml.preprocessing.preprocessing import Preprocessing
    from ml.preprocessing.feature_selection import FeatureSelector
    from ml.preprocessing.dimensionality_reduction import DimensionalityReducer
    from ml.preprocessing.normalization import Normalizer

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.svm import LinearSVC
    from mpl_toolkits.mplot3d import Axes3D
    df = Spreadsheet().get_data('iris.csv')

    def vis_support(df,X_, target = 'class'):
        df_ = df.copy()
        for column in df_.columns:
            if column != target:
                df_ = df_.drop(columns=[column])
        df_['1'] = X_[:,0]
        df_['2'] = X_[:,1]
        print(df_)
        class_list = []
        for label in df_[target].unique():
            class_list.append(df_[df_[target]==label])
        return class_list

    def preprocess(df, norm_cols = {}):
        p = Preprocessing()
        n = Normalizer(norm_cols = norm_cols)
        df = p.clean_data(df)
        X = df.drop(columns=['class'])
        X = p.categ_encoding(X)
        X = n.fit_transform(X)
        return X

    def plot(class_list1, class_list2 = None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(class_list2, list) else 1
        ax = fig.add_subplot(1, num_plots, 1)
        for cls in class_list1:
            ax.scatter(cls['1'],cls['2'])
        if isinstance(class_list2, list):
            ax = fig.add_subplot(1, num_plots, 2)
            for cls in class_list2:
                ax.scatter(cls['1'],cls['2'])
        st.pyplot(plt)

    def plot3d(class_list1, class_list2 = None):
        plt.clf()
        fig = plt.figure()
        num_plots = 2 if isinstance(class_list2, list) else 1
        ax = fig.add_subplot(1, num_plots, 1,projection='3d')
        for cls in class_list1:
            ax.scatter(cls['1'],cls['2'],cls['3']) 
        if isinstance(class_list2, list):
            ax = fig.add_subplot(1, num_plots, 2,projection='3d')
            for cls in class_list2:
                ax.scatter(cls['1'],cls['2'],cls['3'])
        st.pyplot(plt)

    def vis_support3d(df,X_, target = 'class'):
        df_ = df.copy()
        for column in df_.columns:
            if column != target:
                df_ = df_.drop(columns=[column])
        df_['1'] = X_[:,0]
        df_['2'] = X_[:,1]
        df_['3'] = X_[:,2]
        print(df_)
        class_list = []
        for label in df_[target].unique():
            class_list.append(df_[df_[target]==label])
        return class_list

    p = Preprocessing()
    df = p.clean_data(df)

    X = preprocess(df)

    if algo_1 == 'PCA':
        dic[algo_1]["k"] = int(n_components)
        dic[algo_1]["columns"] = X.columns
    else:
        dic[algo_1]["n_components"] = int(n_components)
    if algo_2 == 'PCA':
        dic[algo_2]["k"] = int(n_components)
        dic[algo_2]["columns"] = X.columns
    else:
        dic[algo_2]["n_components"] = int(n_components)

    plt.clf()
    d1 = DimensionalityReducer(**dic[algo_1])
    d2 = DimensionalityReducer(**dic[algo_2])
    begin1 = time.process_time()
    X_t1 = d1.fit_transform(X).values if algo_1 == 'PCA' else d1.fit_transform(X)
    time_1 = time.process_time() - begin1
    begin2 = time.process_time()
    X_t2 = d2.fit_transform(X).values if algo_1 == 'PCA' else d1.fit_transform(X)
    time_2 = time.process_time() - begin2
    class_list1 = vis_support(df,X_t1)
    class_list2 = vis_support(df,X_t2)

    if n_components == '2':
        class_list1 = vis_support(df,X_t1)
        class_list2 = vis_support(df,X_t2)
        plot(class_list1,class_list2)
    else:
        class_list1 = vis_support3d(df,X_t1)
        class_list2  = vis_support3d(df,X_t2)
        plot3d(class_list1,class_list2)

    #col1, col2 = st.beta_columns((1,1))

    #col1.write("""### Time for {}: {}""".format(algo_1,time_1))
    #col2.write("""### Time for {}: {}""".format(algo_2,time_2))
