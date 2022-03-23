import glob
import os

from minio import Minio
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class DataSet:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_raw = self._load_data()
        self.data_clean = self._clean_data(self.data_raw)

    def _load_data(self):
        if os.environ.get('MINIO_ADDRESS') is not None:
            # ENV Variables
            minio_address = str(os.environ['MINIO_ADDRESS'])
            minio_port = str(os.environ['MINIO_PORT'])
            minio_access_key = str(os.environ['MINIO_ACCESS'])
            minio_secret_key = str(os.environ['MINIO_SECRET'])
            bucket_name = str(os.environ['MINIO_BUCKET_NAME'])
            object_name = str(os.environ['MINIO_OBJECT_NAME'])

            minioClient = Minio(
                '{0}:{1}'.format(minio_address, minio_port),
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False,
            )
            res = minioClient.get_object(bucket_name, object_name)
            print("Data loaded")
            df = pd.read_csv(res)
        else:
            data_files_list = glob.glob(f"{self.data_path}/*")
            if len(data_files_list) != 1:
                raise RuntimeError(f"one file expected but fund {len(data_files_list)}")
            else:
                try:
                    df = pd.read_csv(data_files_list[0])
                except IOError:
                    print("CSV data not accessible")
        return df

    def _clean_data(self, raw_df):
        match = lambda a, b: [b.index(x) + 1 if x in b else None for x in a]

        record_ids = raw_df["record_id"][pd.isna(raw_df["diagnosed_leuk"]) == False]

        matched_record_ids = list(match(list(raw_df["record_id"]), list(record_ids)))

        matched_record_ids_none = []

        number = 0

        for x in matched_record_ids:
            if x is not None:
                matched_record_ids_none.append(number)
            number = number + 1

        df_labels = raw_df.iloc[matched_record_ids_none, :]

        df_only_labels = df_labels[df_labels["redcap_repeat_instrument"].isna()]

        rri_list = df_labels["redcap_repeat_instrument"] == "examination_data_use_new_sheet_for_every_visit"

        exam_numbers = []
        number = 0
        for x in rri_list:
            if x is True:
                exam_numbers.append(number)
            number = number + 1

        symptom_df = df_labels.iloc[exam_numbers, :]

        labels = list(df_only_labels["diagnosed_leuk"])

        symptom_first_visit = symptom_df.loc[symptom_df['redcap_repeat_instance'] == 1]

        only_symptoms = symptom_first_visit.iloc[:, symptom_first_visit.columns.get_loc("visit1_fir"):]

        columns_without_na = []

        for coloums in range(0, len(only_symptoms.iloc[0, :])):
            if not (all(only_symptoms.iloc[:, coloums].isna())):
                columns_without_na.append(coloums)

        only_symptoms_with_out_na = only_symptoms.iloc[:, columns_without_na]

        columns_without_var = []

        for coloums in range(0, len(only_symptoms_with_out_na.iloc[0, :])):
            if (len(set(only_symptoms_with_out_na.iloc[:, coloums])) != 1):
                columns_without_var.append(coloums)

        only_symptoms_final = only_symptoms_with_out_na.iloc[:, columns_without_var]

        only_symptoms_final["visit1_fir"][only_symptoms_final["visit1_fir"].isna()] = -1  # TODO klären

        only_symptoms_final["cog"][only_symptoms_final["cog"].isna()] = 0
        only_symptoms_final=self._fix_all_related_variables(only_symptoms_final)

        only_symptoms_final = only_symptoms_final.drop(columns="visit1_fir")
        only_symptoms_final = only_symptoms_final.drop(
            columns="examination_data_use_new_sheet_for_every_visit_complete")

        only_symptoms_final = self._fill_missing(only_symptoms_final)
        sc = StandardScaler()

        only_symptoms_final = pd.DataFrame(sc.fit_transform(only_symptoms_final.values))

        only_symptoms_final.insert(loc=0, column='label', value=labels)

        print("Data preparation is complete")
        return only_symptoms_final

    def _fix_all_related_variables(self, df):
        related_variables = [
            ("apha", "cog"),
            ("cogloss", "cog"),
            ("eap", "cog"),
            ("loc", "cogloss"),
            ("ic", "cog"),
            ("ii", "cog"),
            ("fati", "cog"),
            ("apr", "cog"),
            ("red_consciousness_confus", "cog"),
            ("agnosia", "cog"),
            ("psychosis", "cog"),
            ("hallucinations_delusions", "cog"),
            ("mab", "visit1_fir"),
            ("adh", "mab"),
            ("depr", "mab"),
            ("ma", "mab"),
            ("personality", "mab"),
            ("s_e", "visit1_fir"),
            ("fs___2", "s_e"),
            ("emd", "visit1_fir"),
            ("diplopia", "emd"),
            ("nys", "emd"),
            ("ino", "emd"),
            ("oculomot", "emd"),
            ("fourth_cranial_nerve_palsy", "emd"),
            ("abducens", "emd"),
            ("thy", "crn"),
            ("fp", "crn"),
            ("od", "crn"),
            ("hi", "crn"),
            ("hp", "crn"),
            ("trig_neur", "crn"),
            ("spsw", "visit1_fir"),
            ("dya", "spsw"),
            ("scs", "spsw"),
            ("dysphon", "spsw"),
            ("slurred_speech", "spsw"),
            ("bulbar_palsy", "spsw"),
            ("pseudobulbar_palsy", "spsw"),
            ("dyp", "visit1_fir"),
            ("emp", "visi"),
            ("var", "visi"),
            ("cvd", "visi"),
            ("cvi", "visi"),
            ("sim", "visit1_fir"),
            ("sper", "sim"),
            ("vs", "sim"),
            ("cersy", "visit1_fir"),
            ("trem", "cersy"),
            ("hyo", "cersy"),
            ("dyt", "cersy"),
            ("fmd", "cersy"),
            ("ataxia", "cersy"),
            ("bd", "visit1_fir"),
            ("sexd", "visit1_fir"),
            ("edy", "sexd"),
            ("ll", "sexd"),
            ("bi", "visit1_fir"),
            ("prs", "visit1_fir"),
            ("tprs", "prs"),
            #("severity_of_paresis", "tprs"),
            ("psi", "prs"),
            ("spas", "visit1_fir"),
            ("tspas", "spas"),
            ("vertigo_dizziness", "visit1_fir"),
            ("type_of_dizziness", "vertigo_dizziness"),
            ("gdis", "visit1_fir"),
            ("exgdis", "gdis"),
            ("nnsym", "visit1_fir"),
            ("addd", "nnsym"),
            ("hypogon", "nnsym"),
            ("pd", "visit1_fir")
        ]
        for related_variabl_pair in related_variables:
            try:
                df=self._fix_related_variables(df, related_variabl_pair[0], related_variabl_pair[1])
            except Exception as e:
                print("Block entry doesnt work")
                print(e)

        return df

    def _fix_related_variables(self, df, target_var, super_var):
        df[target_var][(df[super_var] == 1) == (df[target_var].isna())] = -1
        df[target_var][(list(df[super_var] == 2)) and list((df[target_var].isna()))] = 0
        df[target_var][df[super_var].isna()] = 0
        return df

    def _fill_missing(self, df):
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=2), missing_values=-1, max_iter=100)
        data_real = df
        data_imputed = imputer.fit(data_real).transform(data_real)
        data = pd.DataFrame(data=data_imputed, columns=data_real.columns.values.tolist())
        return data

    def _transform_diagnosis_split_into_leuko_and_not_leuko(self, labels):
        non_leuko_codes = [84]
        leuko_codes = [2, 103, 1, 7, 29, 60]
        for leuko_code in leuko_codes:
            labels = labels.replace(leuko_code, 0)
        for non_leuko_code in non_leuko_codes:
            labels = labels.replace(non_leuko_code, 1)
        return labels

    def get_data_all(self):
        input_all = self.data_clean.iloc[:, 1:]

        target_all = self._transform_diagnosis_split_into_leuko_and_not_leuko(self.data_clean.iloc[:, 0])

        return (input_all, target_all)
