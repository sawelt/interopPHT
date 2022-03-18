from sklearn.svm import SVC


class TrainModel:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        svm_class_linear = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=0, probability=True)
        svm_class_linear.fit(self.inputs, self.targets)
        y_pre_class_linear = svm_class_linear.predict(self.inputs)
        print(y_pre_class_linear)
