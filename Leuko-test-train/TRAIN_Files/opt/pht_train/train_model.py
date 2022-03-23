from sklearn.svm import SVC
import pandas as pd
import pickle
from joblib import dump, load
import glob

class TrainModel:
    def __init__(self, inputs, targets, result_path):
        self.inputs = inputs
        self.targets = targets
        self.result_path =result_path

        self.model = self._train_model()
        self._save_model()
        self._save_results()

    def _train_model(self):
        model = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=0, probability=True)
        model.fit(self.inputs, self.targets)
        return model


    def _save_results(self):
        y_pre_class_linear = self.model.predict(self.inputs)
        # pd.DataFrame(linear_result).to_csv(f'{RESULT_PATH}/binary_linear.csv')
        print(y_pre_class_linear)
        print(self.targets)

    def _save_model(self):
        number_models = len(glob.glob(f"{self.result_path}/*.joblib"))
        dump(self.model, f'{self.result_path}/model_{number_models}.joblib')

