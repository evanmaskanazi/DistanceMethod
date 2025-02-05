import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from scipy import spatial, stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class DataLoader:
    def __init__(self, test_path, train_path):
        self.test_path = test_path
        self.train_path = train_path
        
    def load_data(self):
        dftest0 = pd.read_csv(self.test_path)
        dftrain0 = pd.read_csv(self.train_path)
        
        X_train = dftrain0.drop(["Ef", "Eg"], axis=1)
        X_test = dftest0.drop(["Ef", "Eg"], axis=1)
        y_train = dftrain0['Eg'].astype('float')
        y_test = dftest0['Eg'].astype('float')
        
        return X_train, X_test, y_train, y_test

class ErrorAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('SVM', SVR())
        ])
        self.grid_params = {
            'SVM__C': [100],
            'SVM__gamma': ['auto'],
            'SVM__kernel': ['rbf'],
            'SVM__epsilon': [0.001]
        }
    
    @staticmethod
    def gram_schmidt(X):
        Q, _ = np.linalg.qr(X)
        return Q
    
    def analyze_errors(self, X_train, X_test, y_train, y_test):
        grid = GridSearchCV(self.pipeline, self.grid_params, cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
        # Calculate prediction errors
        errors = np.abs(y_pred - y_test)
        
        # Stack data for analysis
        stack = np.vstack((np.array(X_test).T, errors, y_test, y_pred))
        return stack.T, grid

    def compute_feature_importance(self, X, y, model):
        r = permutation_importance(model, X, y, n_repeats=30, random_state=0)
        return r.importances_mean

    def calculate_distances(self, data, reference_data, k=10):
        return np.array([
            np.mean(spatial.KDTree(reference_data).query(point, k)[0])
            for point in data
        ])

class ErrorVisualizer:
    def __init__(self):
        self.vec = list(range(1, 11))
        
    def split_by_metric(self, metric_values, error_values, y_test):
        splits = np.array_split(np.sort(metric_values), 10)
        means = []
        
        for split in splits:
            mask = np.isin(metric_values, split)
            means.append(np.mean(error_values[mask]))
            
        return means
    
    def plot_results(self, results_dict):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        styles = {
            'GS and Feature Importance': 'b',
            'No GS or Feature Importance': 'g',
            'Ten Points, GS and Feature Importance': 'r',
            'Ten Points, No GS or Feature Importance': 'k'
        }
        
        for label, (values, color) in styles.items():
            ax.plot(self.vec, values, color, label=label)
        
        self._setup_plot_style(ax)
        plt.savefig('error_analysis.svg')
        
    def _setup_plot_style(self, ax):
        ax.legend(loc='upper left', shadow=True, fontsize='medium')
        ax.grid(True)
        self._setup_ticks(ax)
        ax.set_facecolor('w')
        plt.xlabel('Group Number', fontsize=15)
        plt.ylabel('Error (Counts)', fontsize=15)
    
    def _setup_ticks(self, ax):
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(MultipleLocator(1.0 if axis == ax.xaxis else 0.02))
            axis.set_minor_locator(MultipleLocator(1.0 if axis == ax.xaxis else 0.02))

def main():
    # Initialize components
    loader = DataLoader('testEgEf.txt', 'trainEgEf.txt')
    analyzer = ErrorAnalyzer()
    visualizer = ErrorVisualizer()
    
    # Load and process data
    X_train, X_test, y_train, y_test = loader.load_data()
    stack, model = analyzer.analyze_errors(X_train, X_test, y_train, y_test)
    
    # Compute various metrics
    feature_importance = analyzer.compute_feature_importance(X_train, y_train, model)
    
    # Generate results for different approaches
    results = {}
    
    # Original data with GS and feature importance
    gs_data = analyzer.gram_schmidt(stack[:, :-3])
    for i in range(gs_data.shape[1]):
        gs_data[:, i] *= np.sqrt(feature_importance[i])
    
    # Calculate and store all variants
    variants = {
        'GS and Feature Importance': gs_data,
        'No GS or Feature Importance': stack[:, :-3],
        'Ten Points, GS and Feature Importance': analyzer.calculate_distances(gs_data, X_train),
        'Ten Points, No GS or Feature Importance': analyzer.calculate_distances(X_test, X_train)
    }
    
    for name, data in variants.items():
        results[name] = visualizer.split_by_metric(data, stack[:, -3], y_test)
    
    # Visualize results
    visualizer.plot_results(results)

if __name__ == "__main__":
    main()
