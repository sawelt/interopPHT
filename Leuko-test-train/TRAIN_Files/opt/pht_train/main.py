from data_set import DataSet
from train_model import TrainModel

# Global pahts
#DATA_PATH = "/opt/train_data"
#RESULT_PATH = "/opt/pht_results"
DATA_PATH = "../train_data"
RESULT_PATH = "../pht_results"

def main():
    data_set = DataSet(DATA_PATH)
    x, y = data_set.get_data_all()
    TrainModel(x,y,RESULT_PATH)


if __name__ == '__main__':
    main()
