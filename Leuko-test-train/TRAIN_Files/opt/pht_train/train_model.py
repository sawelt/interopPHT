import os

from sklearn.svm import SVC

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
        if not os.path.isfile(f"{self.result_path}/station_acc.csv"):
            with open(f"{self.result_path}/station_acc.csv", "a") as results:
                results.write("acc\n")

        with open(f"{self.result_path}/station_acc.csv", "a") as results:
            accuracy = (y_pre_class_linear == self.targets.to_numpy()).all().mean()
            results.write(f"{accuracy} \n")


    def _save_model(self):
        number_models = len(glob.glob(f"{self.result_path}/*.joblib"))
        dump(self.model, f'{self.result_path}/model_{number_models}.joblib')

