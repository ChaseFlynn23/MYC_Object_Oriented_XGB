import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import json
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import Image, display

class XGBoostTrainer:
    def __init__(self, wt_dict, D132H_dict, window_sizes, default_hyperparameters, eta_values, max_depth_values, subsample_values):
        self.wt_dict = wt_dict
        self.D132H_dict = D132H_dict
        self.window_sizes = window_sizes
        self.default_hyperparameters = default_hyperparameters
        self.eta_values = eta_values
        self.max_depth_values = max_depth_values
        self.subsample_values = subsample_values
        self.default_accuracy_values = {}
        self.best_accuracy_values = {}
        self.best_eta_values = {}
        self.best_max_depth_values = {}
        self.best_subsample_values = {}

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def prepare_data(self, window_size):
        wildtype_data = self.wt_dict[window_size]
        wildtype_label = np.zeros(len(wildtype_data))
        mutant_data = self.D132H_dict[window_size]
        mutant_label = np.ones(len(mutant_data))

        lcc_data = np.vstack((wildtype_data, mutant_data))
        label_data = np.hstack((wildtype_label, mutant_label))
        lcc_data, label_data = self.unison_shuffled_copies(lcc_data, label_data)
        lcc_data /= 100
        upper_training_limit = int(len(lcc_data) * 0.8)

        return lcc_data[:upper_training_limit], label_data[:upper_training_limit], lcc_data[upper_training_limit:], label_data[upper_training_limit:]

    def train_and_evaluate(self, train_data, train_label, test_data, test_label, **hyperparameters):
        model = XGBClassifier(**hyperparameters)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)
        return accuracy_score(test_label, predictions)

    def find_best_hyperparameter(self, train_data, train_label, test_data, test_label, hyperparameter_name, values):
        best_score = 0
        best_value = None
        for value in values:
            self.default_hyperparameters[hyperparameter_name] = value
            score = self.train_and_evaluate(train_data, train_label, test_data, test_label, **self.default_hyperparameters)
            if score > best_score:
                best_score = score
                best_value = value
        return best_value, best_score

    def evaluate_default_hyperparameters(self):
        for window_size in self.window_sizes:
            train_data, train_label, test_data, test_label = self.prepare_data(window_size)
            accuracy = self.train_and_evaluate(train_data, train_label, test_data, test_label, **self.default_hyperparameters)
            self.default_accuracy_values[window_size] = accuracy

    def tune_hyperparameters(self):
        for window_size in self.window_sizes:
            train_data, train_label, test_data, test_label = self.prepare_data(window_size)
            hyperparameters = self.default_hyperparameters.copy()
            best_eta, _ = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'eta', self.eta_values)
            self.best_eta_values[window_size] = best_eta
            hyperparameters['eta'] = best_eta

            best_max_depth, _ = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'max_depth', self.max_depth_values)
            self.best_max_depth_values[window_size] = best_max_depth
            hyperparameters['max_depth'] = best_max_depth

            best_subsample, best_accuracy = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'subsample', self.subsample_values)
            self.best_subsample_values[window_size] = best_subsample
            self.best_accuracy_values[window_size] = best_accuracy

    def tune_hyperparameters_and_save(self):
        self.tune_hyperparameters()
        self.save_tuning_results()

    def save_tuning_results(self, trial_number=None):
        if trial_number is None:
            trial_number = self.get_next_trial_number()

        save_path = f'XGB_Tuning/XGB_Tuning_Trial_{trial_number}'
        os.makedirs(save_path, exist_ok=True)

        results = {
            'best_eta_values': self.best_eta_values,
            'best_max_depth_values': self.best_max_depth_values,
            'best_subsample_values': self.best_subsample_values,
            'best_accuracy_values': self.best_accuracy_values
        }

        with open(f'{save_path}/tuning_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Tuning results saved in {save_path}")

    def load_tuning_results(self, trial_number):
        path = f'XGB_Tuning/XGB_Tuning_Trial_{trial_number}/tuning_results.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                results = json.load(f)
            self.best_eta_values = results['best_eta_values']
            self.best_max_depth_values = results['best_max_depth_values']
            self.best_subsample_values = results['best_subsample_values']
            self.best_accuracy_values = results['best_accuracy_values']
            print(f"Tuning results loaded from {path}")
        else:
            raise FileNotFoundError(f"No tuning results found for trial number {trial_number}")

    @staticmethod
    def get_next_trial_number():
        base_path = 'XGB_Tuning'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            return 1
        else:
            existing_trials = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            trial_numbers = [int(trial.split('_')[-1]) for trial in existing_trials if trial.startswith('XGB_Tuning_Trial_')]
            if trial_numbers:
                return max(trial_numbers) + 1
            else:
                return 1
        
    def save_important_features_and_plot(self, importance_threshold, hyperparameters):
        output_folder = 'XGB_filtered_data'
        feature_importances_folder = 'XGB_Position_Importance_Values'
        feature_importance_plot_folder = 'XGB_Pos_Imp_Figs'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(feature_importances_folder, exist_ok=True)
        os.makedirs(feature_importance_plot_folder, exist_ok=True)
        

        total_positions_saved = 0

        for window_size in self.window_sizes:
            adjusted_threshold = importance_threshold * (68 / (70 - window_size))
            train_data, train_label, _, _ = self.prepare_data(window_size)
            model = XGBClassifier(**hyperparameters)
            model.fit(train_data, train_label)

            importances = model.feature_importances_
            important_features = np.where(importances >= adjusted_threshold)[0]

            # Save feature importances for each window size
            positions = np.arange(1, len(importances) + 1)
            df_importances = pd.DataFrame({'Position': positions, 'Importance': importances})
            importance_file_path = os.path.join(feature_importances_folder, f'feature_importances_window_{window_size}.csv')
            df_importances.to_csv(importance_file_path, index=False)

            # Generate and save feature importance plots
            fig, ax = plt.subplots(figsize=(9, 6))
            indices = np.arange(len(importances))
            ax.bar(indices, importances, color='#0504aa')
            ax.set_title(f'Window size = {window_size}, Mean = {np.mean(importances):.2f}, Std = {np.std(importances):.3f}')
            ax.grid(True)
            ax.set_xlabel('Feature (Position)')
            ax.set_ylabel('Feature Importance')
            ax.set_ylim(0, np.max(importances) * 1.1)
            plot_path = os.path.join(feature_importance_plot_folder, f'feature_importance_{window_size}.png')
            fig.savefig(plot_path)
            plt.close(fig)

            if len(important_features) > 0:
                total_positions_saved += len(important_features)
                print(f"Window size {window_size}: {len(important_features)} positions saved")

                # Assuming position_numbers is correctly extracted based on your data structure
                position_numbers = self.wt_dict[window_size].iloc[0, important_features + 1]
                wt_filtered_data = self.wt_dict[window_size].iloc[:, important_features + 1]
                D132H_filtered_data = self.D132H_dict[window_size].iloc[:, important_features + 1]

                wt_filtered = pd.concat([pd.DataFrame([position_numbers.values], columns=position_numbers.index), wt_filtered_data])
                D132H_filtered = pd.concat([pd.DataFrame([position_numbers.values], columns=position_numbers.index), D132H_filtered_data])

                wt_filtered.to_csv(f'{output_folder}/wt_{window_size}f.lccdata', index_label='Index')
                D132H_filtered.to_csv(f'{output_folder}/D132H_{window_size}f.lccdata', index_label='Index')

        print(f"Total positions saved: {total_positions_saved}")



    def plot_hyperparameter_values(self, window_sizes, default_accuracies, tuned_accuracies, title):
        plt.figure(figsize=(10, 6))
        window_sizes_list = [int(ws) for ws in window_sizes]
        default_acc_values = [default_accuracies.get(ws, None) for ws in window_sizes_list]
        tuned_acc_values = [tuned_accuracies.get(str(ws), None) for ws in window_sizes_list]

        valid_window_sizes = [ws for ws, d_acc in zip(window_sizes_list, default_acc_values) if d_acc is not None]
        default_acc_values = [d_acc for d_acc in default_acc_values if d_acc is not None]
        tuned_acc_values = [t_acc for t_acc in tuned_acc_values if t_acc is not None]

        if not default_acc_values or not tuned_acc_values:
            print("Error: No valid accuracy data available for plotting.")
            return

        plt.plot(valid_window_sizes, default_acc_values, label='Default', marker='o')
        plt.plot(valid_window_sizes, tuned_acc_values, label='Tuned', marker='x')
        plt.xticks(valid_window_sizes)
        plt.xlabel('Window Size')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_specific_hyperparameter_values(self, hyperparameter_values_dict, title, ylabel, possible_values=None):
        window_sizes = list(hyperparameter_values_dict.keys())
        hyperparameter_values = list(hyperparameter_values_dict.values())

        if possible_values is not None:
            print(f"Possible values for {ylabel}: {possible_values}")

        plt.figure(figsize=(10, 6))
        plt.plot(window_sizes, hyperparameter_values, marker='o')
        plt.xticks(window_sizes)
        plt.xlabel('Window Size')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def display_feature_importance_plots(self, output_folder):
        for window_size in self.window_sizes:
            image_path = f"{output_folder}/feature_importance_{window_size}.png"
            display(Image(filename=image_path))

