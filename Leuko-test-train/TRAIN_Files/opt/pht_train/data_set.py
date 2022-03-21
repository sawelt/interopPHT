import glob
import os

from minio import Minio
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import  RandomForestRegressor


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

        try:
            only_symptoms_final["apha"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["apha"].isna())] = -1
            only_symptoms_final["apha"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["apha"].isna()))] = 0
            only_symptoms_final["apha"][only_symptoms_final["apha"].isna()] = 0

            # TODO was passiert bei cog == 3

            only_symptoms_final["cogloss"][
                (only_symptoms_final["cog"] == 1) == (only_symptoms_final["cogloss"].isna())] = -1
            only_symptoms_final["cogloss"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["cogloss"].isna()))] = 0
            only_symptoms_final["cogloss"][only_symptoms_final["cogloss"].isna()] = 0

            only_symptoms_final["eap"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["eap"].isna())] = -1
            only_symptoms_final["eap"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["eap"].isna()))] = 0
            only_symptoms_final["eap"][only_symptoms_final["eap"].isna()] = 0

            only_symptoms_final["loc"][
                (only_symptoms_final["cogloss"] == 1) == (only_symptoms_final["eap"].isna())] = -1
            only_symptoms_final["loc"][
                (list(only_symptoms_final["cogloss"] == 2)) and list((only_symptoms_final["eap"].isna()))] = 0
            only_symptoms_final["loc"][only_symptoms_final["loc"].isna()] = 0

            only_symptoms_final["ic"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["ic"].isna())] = -1
            only_symptoms_final["ic"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["ic"].isna()))] = 0
            only_symptoms_final["ic"][only_symptoms_final["ic"].isna()] = 0

            only_symptoms_final["ii"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["ii"].isna())] = -1
            only_symptoms_final["ii"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["ii"].isna()))] = 0
            only_symptoms_final["ii"][only_symptoms_final["ii"].isna()] = 0

            only_symptoms_final["fati"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["fati"].isna())] = -1
            only_symptoms_final["fati"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["fati"].isna()))] = 0
            only_symptoms_final["fati"][only_symptoms_final["fati"].isna()] = 0

            only_symptoms_final["apr"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["apr"].isna())] = -1
            only_symptoms_final["apr"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["apr"].isna()))] = 0
            only_symptoms_final["apr"][only_symptoms_final["apr"].isna()] = 0

            only_symptoms_final["red_consciousness_confus"][
                (only_symptoms_final["cog"] == 1) == (only_symptoms_final["red_consciousness_confus"].isna())] = -1
            only_symptoms_final["red_consciousness_confus"][
                (list(only_symptoms_final["cog"] == 2)) and list(
                    (only_symptoms_final["red_consciousness_confus"].isna()))] = 0
            only_symptoms_final["red_consciousness_confus"][only_symptoms_final["red_consciousness_confus"].isna()] = 0

            only_symptoms_final["agnosia"][
                (only_symptoms_final["cog"] == 1) == (only_symptoms_final["agnosia"].isna())] = -1
            only_symptoms_final["agnosia"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["agnosia"].isna()))] = 0
            only_symptoms_final["agnosia"][only_symptoms_final["agnosia"].isna()] = 0

            only_symptoms_final["psychosis"][
                (only_symptoms_final["cog"] == 1) == (only_symptoms_final["psychosis"].isna())] = -1
            only_symptoms_final["psychosis"][
                (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["psychosis"].isna()))] = 0
            only_symptoms_final["psychosis"][only_symptoms_final["psychosis"].isna()] = 0

            only_symptoms_final["hallucinations_delusions"][
                (only_symptoms_final["cog"] == 1) == (only_symptoms_final["hallucinations_delusions"].isna())] = -1
            only_symptoms_final["hallucinations_delusions"][
                (list(only_symptoms_final["cog"] == 2)) and list(
                    (only_symptoms_final["hallucinations_delusions"].isna()))] = 0
            only_symptoms_final["hallucinations_delusions"][only_symptoms_final["hallucinations_delusions"].isna()] = 0

            only_symptoms_final["sleep_disturbance"][only_symptoms_final["sleep_disturbance"].isna()] = -1

            only_symptoms_final["mab"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["mab"].isna())] = -1
            only_symptoms_final["mab"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["mab"].isna()))] = 0
            only_symptoms_final["mab"][only_symptoms_final["mab"].isna()] = 0

            only_symptoms_final["adh"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["adh"].isna())] = -1
            only_symptoms_final["adh"][
                (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["adh"].isna()))] = 0
            only_symptoms_final["adh"][only_symptoms_final["adh"].isna()] = 0

            only_symptoms_final["depr"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["depr"].isna())] = -1
            only_symptoms_final["depr"][
                (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["depr"].isna()))] = 0
            only_symptoms_final["depr"][only_symptoms_final["depr"].isna()] = 0

            only_symptoms_final["ma"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["ma"].isna())] = -1
            only_symptoms_final["ma"][
                (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["ma"].isna()))] = 0
            only_symptoms_final["ma"][only_symptoms_final["ma"].isna()] = 0

            only_symptoms_final["personality"][
                (only_symptoms_final["mab"] == 1) == (only_symptoms_final["personality"].isna())] = -1
            only_symptoms_final["personality"][
                (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["personality"].isna()))] = 0
            only_symptoms_final["personality"][only_symptoms_final["personality"].isna()] = 0

            only_symptoms_final["s_e"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["s_e"].isna())] = -1
            only_symptoms_final["s_e"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["s_e"].isna()))] = 0
            only_symptoms_final["s_e"][only_symptoms_final["s_e"].isna()] = 0

            only_symptoms_final["fs___2"][
                (only_symptoms_final["s_e"] == 1) == (only_symptoms_final["fs___2"].isna())] = -1
            only_symptoms_final["fs___2"][
                (list(only_symptoms_final["s_e"] == 2)) and list((only_symptoms_final["fs___2"].isna()))] = 0
            only_symptoms_final["fs___2"][only_symptoms_final["fs___2"].isna()] = 0

            only_symptoms_final["emd"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["emd"].isna())] = -1
            only_symptoms_final["emd"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["emd"].isna()))] = 0
            only_symptoms_final["emd"][only_symptoms_final["emd"].isna()] = 0

            only_symptoms_final["diplopia"][
                (only_symptoms_final["emd"] == 1) == (only_symptoms_final["diplopia"].isna())] = -1
            only_symptoms_final["diplopia"][
                (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["diplopia"].isna()))] = 0
            only_symptoms_final["diplopia"][only_symptoms_final["diplopia"].isna()] = 0

            only_symptoms_final["nys"][(only_symptoms_final["emd"] == 1) == (only_symptoms_final["nys"].isna())] = -1
            only_symptoms_final["nys"][
                (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["nys"].isna()))] = 0
            only_symptoms_final["nys"][only_symptoms_final["nys"].isna()] = 0

            only_symptoms_final["ino"][(only_symptoms_final["emd"] == 1) == (only_symptoms_final["ino"].isna())] = -1
            only_symptoms_final["ino"][
                (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["ino"].isna()))] = 0
            only_symptoms_final["ino"][only_symptoms_final["ino"].isna()] = 0

            only_symptoms_final["oculomot"][
                (only_symptoms_final["emd"] == 1) == (only_symptoms_final["oculomot"].isna())] = -1
            only_symptoms_final["oculomot"][
                (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["oculomot"].isna()))] = 0
            only_symptoms_final["oculomot"][only_symptoms_final["oculomot"].isna()] = 0

            only_symptoms_final["fourth_cranial_nerve_palsy"][
                (only_symptoms_final["emd"] == 1) == (only_symptoms_final["fourth_cranial_nerve_palsy"].isna())] = -1
            only_symptoms_final["fourth_cranial_nerve_palsy"][(list(only_symptoms_final["emd"] == 2)) and list(
                (only_symptoms_final["fourth_cranial_nerve_palsy"].isna()))] = 0
            only_symptoms_final["fourth_cranial_nerve_palsy"][
                only_symptoms_final["fourth_cranial_nerve_palsy"].isna()] = 0

            only_symptoms_final["abducens"][
                (only_symptoms_final["emd"] == 1) == (only_symptoms_final["abducens"].isna())] = -1
            only_symptoms_final["abducens"][
                (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["abducens"].isna()))] = 0
            only_symptoms_final["abducens"][only_symptoms_final["ino"].isna()] = 0

            only_symptoms_final["thy"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["thy"].isna())] = -1
            only_symptoms_final["thy"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["thy"].isna()))] = 0
            only_symptoms_final["thy"][only_symptoms_final["thy"].isna()] = 0

            only_symptoms_final["fp"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["fp"].isna())] = -1
            only_symptoms_final["fp"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["fp"].isna()))] = 0
            only_symptoms_final["fp"][only_symptoms_final["fp"].isna()] = 0

            only_symptoms_final["od"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["od"].isna())] = -1
            only_symptoms_final["od"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["od"].isna()))] = 0
            only_symptoms_final["od"][only_symptoms_final["od"].isna()] = 0

            only_symptoms_final["hi"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["hi"].isna())] = -1
            only_symptoms_final["hi"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["hi"].isna()))] = 0
            only_symptoms_final["hi"][only_symptoms_final["hi"].isna()] = 0

            only_symptoms_final["hp"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["hp"].isna())] = -1
            only_symptoms_final["hp"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["hp"].isna()))] = 0
            only_symptoms_final["hp"][only_symptoms_final["hp"].isna()] = 0

            only_symptoms_final["trig_neur"][
                (only_symptoms_final["crn"] == 1) == (only_symptoms_final["trig_neur"].isna())] = -1
            only_symptoms_final["trig_neur"][
                (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["trig_neur"].isna()))] = 0
            only_symptoms_final["trig_neur"][only_symptoms_final["trig_neur"].isna()] = 0

            only_symptoms_final["spsw"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["spsw"].isna())] = -1
            only_symptoms_final["spsw"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["spsw"].isna()))] = 0
            only_symptoms_final["spsw"][only_symptoms_final["spsw"].isna()] = 0

            only_symptoms_final["dya"][(only_symptoms_final["spsw"] == 1) == (only_symptoms_final["dya"].isna())] = -1
            only_symptoms_final["dya"][
                (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["dya"].isna()))] = 0
            only_symptoms_final["dya"][only_symptoms_final["dya"].isna()] = 0

            only_symptoms_final["scs"][(only_symptoms_final["spsw"] == 1) == (only_symptoms_final["scs"].isna())] = -1
            only_symptoms_final["scs"][
                (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["scs"].isna()))] = 0
            only_symptoms_final["scs"][only_symptoms_final["scs"].isna()] = 0

            only_symptoms_final["dysphon"][
                (only_symptoms_final["spsw"] == 1) == (only_symptoms_final["dysphon"].isna())] = -1
            only_symptoms_final["dysphon"][
                (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["dysphon"].isna()))] = 0
            only_symptoms_final["dysphon"][only_symptoms_final["dysphon"].isna()] = 0

            only_symptoms_final["slurred_speech"][
                (only_symptoms_final["spsw"] == 1) == (only_symptoms_final["slurred_speech"].isna())] = -1
            only_symptoms_final["slurred_speech"][
                (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["slurred_speech"].isna()))] = 0
            only_symptoms_final["slurred_speech"][only_symptoms_final["slurred_speech"].isna()] = 0

            only_symptoms_final["bulbar_palsy"][
                (only_symptoms_final["spsw"] == 1) == (only_symptoms_final["bulbar_palsy"].isna())] = -1
            only_symptoms_final["bulbar_palsy"][
                (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["bulbar_palsy"].isna()))] = 0
            only_symptoms_final["bulbar_palsy"][only_symptoms_final["bulbar_palsy"].isna()] = 0

            only_symptoms_final["pseudobulbar_palsy"][
                (only_symptoms_final["spsw"] == 1) == (only_symptoms_final["pseudobulbar_palsy"].isna())] = -1
            only_symptoms_final["pseudobulbar_palsy"][
                (list(only_symptoms_final["spsw"] == 2)) and list(
                    (only_symptoms_final["pseudobulbar_palsy"].isna()))] = 0
            only_symptoms_final["pseudobulbar_palsy"][only_symptoms_final["pseudobulbar_palsy"].isna()] = 0

            only_symptoms_final["dyp"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["dyp"].isna())] = -1
            only_symptoms_final["dyp"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["dyp"].isna()))] = 0
            only_symptoms_final["dyp"][only_symptoms_final["dyp"].isna()] = 0

            only_symptoms_final["emp"][(only_symptoms_final["visi"] == 1) == (only_symptoms_final["emp"].isna())] = -1
            only_symptoms_final["emp"][
                (list(only_symptoms_final["visi"] == 2)) and list((only_symptoms_final["emp"].isna()))] = 0
            only_symptoms_final["emp"][only_symptoms_final["emp"].isna()] = 0

            only_symptoms_final["var"][(only_symptoms_final["visi"] == 1) == (only_symptoms_final["var"].isna())] = -1
            only_symptoms_final["var"][
                (list(only_symptoms_final["visi"] == 2)) and list((only_symptoms_final["var"].isna()))] = 0
            only_symptoms_final["var"][only_symptoms_final["var"].isna()] = 0

            only_symptoms_final["cvd"][(only_symptoms_final["visi"] == 1) == (only_symptoms_final["cvd"].isna())] = -1
            only_symptoms_final["cvd"][
                (list(only_symptoms_final["visi"] == 2)) and list((only_symptoms_final["cvd"].isna()))] = 0
            only_symptoms_final["cvd"][only_symptoms_final["cvd"].isna()] = 0

            only_symptoms_final["cvi"][(only_symptoms_final["visi"] == 1) == (only_symptoms_final["cvi"].isna())] = -1
            only_symptoms_final["cvi"][
                (list(only_symptoms_final["visi"] == 2)) and list((only_symptoms_final["cvi"].isna()))] = 0
            only_symptoms_final["cvi"][only_symptoms_final["cvi"].isna()] = 0

            only_symptoms_final["sim"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["sim"].isna())] = -1
            only_symptoms_final["sim"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["sim"].isna()))] = 0
            only_symptoms_final["sim"][only_symptoms_final["sim"].isna()] = 0

            only_symptoms_final["sper"][(only_symptoms_final["sim"] == 1) == (only_symptoms_final["sper"].isna())] = -1
            only_symptoms_final["sper"][
                (list(only_symptoms_final["sim"] == 2)) and list((only_symptoms_final["sper"].isna()))] = 0
            only_symptoms_final["sper"][only_symptoms_final["sper"].isna()] = 0

            only_symptoms_final["vs"][(only_symptoms_final["sim"] == 1) == (only_symptoms_final["vs"].isna())] = -1
            only_symptoms_final["vs"][
                (list(only_symptoms_final["sim"] == 2)) and list((only_symptoms_final["vs"].isna()))] = 0
            only_symptoms_final["vs"][only_symptoms_final["vs"].isna()] = 0

            only_symptoms_final["cersy"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["cersy"].isna())] = -1
            only_symptoms_final["cersy"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["cersy"].isna()))] = 0
            only_symptoms_final["cersy"][only_symptoms_final["cersy"].isna()] = 0

            only_symptoms_final["trem"][
                (only_symptoms_final["cersy"] == 1) == (only_symptoms_final["trem"].isna())] = -1
            only_symptoms_final["trem"][
                (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["trem"].isna()))] = 0
            only_symptoms_final["trem"][only_symptoms_final["trem"].isna()] = 0

            only_symptoms_final["hyo"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["hyo"].isna())] = -1
            only_symptoms_final["hyo"][
                (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["hyo"].isna()))] = 0
            only_symptoms_final["hyo"][only_symptoms_final["hyo"].isna()] = 0

            only_symptoms_final["dyt"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["dyt"].isna())] = -1
            only_symptoms_final["dyt"][
                (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["dyt"].isna()))] = 0
            only_symptoms_final["dyt"][only_symptoms_final["dyt"].isna()] = 0

            only_symptoms_final["fmd"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["fmd"].isna())] = -1
            only_symptoms_final["fmd"][
                (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["fmd"].isna()))] = 0
            only_symptoms_final["fmd"][only_symptoms_final["fmd"].isna()] = 0

            only_symptoms_final["ataxia"][
                (only_symptoms_final["cersy"] == 1) == (only_symptoms_final["ataxia"].isna())] = -1
            only_symptoms_final["ataxia"][
                (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["ataxia"].isna()))] = 0
            only_symptoms_final["ataxia"][only_symptoms_final["ataxia"].isna()] = 0

            only_symptoms_final["bd"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["bd"].isna())] = -1
            only_symptoms_final["bd"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["bd"].isna()))] = 0
            only_symptoms_final["bd"][only_symptoms_final["bd"].isna()] = 0

            only_symptoms_final["sexd"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["sexd"].isna())] = -1
            only_symptoms_final["sexd"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["sexd"].isna()))] = 0
            only_symptoms_final["sexd"][only_symptoms_final["sexd"].isna()] = 0

            only_symptoms_final["edy"][(only_symptoms_final["sexd"] == 1) == (only_symptoms_final["edy"].isna())] = -1
            only_symptoms_final["edy"][
                (list(only_symptoms_final["sexd"] == 2)) and list((only_symptoms_final["edy"].isna()))] = 0
            only_symptoms_final["edy"][only_symptoms_final["edy"].isna()] = 0

            only_symptoms_final["ll"][(only_symptoms_final["sexd"] == 1) == (only_symptoms_final["ll"].isna())] = -1
            only_symptoms_final["ll"][
                (list(only_symptoms_final["sexd"] == 2)) and list((only_symptoms_final["ll"].isna()))] = 0
            only_symptoms_final["ll"][only_symptoms_final["ll"].isna()] = 0

            only_symptoms_final["bi"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["bi"].isna())] = -1
            only_symptoms_final["bi"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["bi"].isna()))] = 0
            only_symptoms_final["bi"][only_symptoms_final["bi"].isna()] = 0

            only_symptoms_final["prs"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["prs"].isna())] = -1
            only_symptoms_final["prs"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["prs"].isna()))] = 0
            only_symptoms_final["prs"][only_symptoms_final["prs"].isna()] = 0

            only_symptoms_final["tprs"][(only_symptoms_final["prs"] == 1) == (only_symptoms_final["tprs"].isna())] = -1
            only_symptoms_final["tprs"][
                (list(only_symptoms_final["prs"] == 2)) and list((only_symptoms_final["tprs"].isna()))] = 0
            only_symptoms_final["tprs"][only_symptoms_final["tprs"].isna()] = 0

            only_symptoms_final["severity_of_paresis"][
                (only_symptoms_final["tprs"] == 1) == (only_symptoms_final["severity_of_paresis"].isna())] = -1
            only_symptoms_final["severity_of_paresis"][
                (list(only_symptoms_final["tprs"] == 2)) and list(
                    (only_symptoms_final["severity_of_paresis"].isna()))] = 0
            only_symptoms_final["severity_of_paresis"][only_symptoms_final["severity_of_paresis"].isna()] = 0

            only_symptoms_final["psi"][(only_symptoms_final["prs"] == 1) == (only_symptoms_final["psi"].isna())] = -1
            only_symptoms_final["psi"][
                (list(only_symptoms_final["prs"] == 2)) and list((only_symptoms_final["psi"].isna()))] = 0
            only_symptoms_final["psi"][only_symptoms_final["psi"].isna()] = 0

            only_symptoms_final["spas"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["spas"].isna())] = -1
            only_symptoms_final["spas"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["spas"].isna()))] = 0
            only_symptoms_final["spas"][only_symptoms_final["spas"].isna()] = 0

            only_symptoms_final["tspas"][
                (only_symptoms_final["spas"] == 1) == (only_symptoms_final["tspas"].isna())] = -1
            only_symptoms_final["tspas"][
                (list(only_symptoms_final["spas"] == 2)) and list((only_symptoms_final["tspas"].isna()))] = 0
            only_symptoms_final["tspas"][only_symptoms_final["tspas"].isna()] = 0

            only_symptoms_final["vertigo_dizziness"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["vertigo_dizziness"].isna())] = -1
            only_symptoms_final["vertigo_dizziness"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list(
                    (only_symptoms_final["vertigo_dizziness"].isna()))] = 0
            only_symptoms_final["vertigo_dizziness"][only_symptoms_final["vertigo_dizziness"].isna()] = 0

            only_symptoms_final["type_of_dizziness"][
                (only_symptoms_final["vertigo_dizziness"] == 1) == (
                    only_symptoms_final["type_of_dizziness"].isna())] = -1
            only_symptoms_final["type_of_dizziness"][(list(only_symptoms_final["vertigo_dizziness"] == 2)) and list(
                (only_symptoms_final["type_of_dizziness"].isna()))] = 0
            only_symptoms_final["type_of_dizziness"][only_symptoms_final["type_of_dizziness"].isna()] = 0

            only_symptoms_final["gdis"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["gdis"].isna())] = -1
            only_symptoms_final["gdis"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["gdis"].isna()))] = 0
            only_symptoms_final["gdis"][only_symptoms_final["gdis"].isna()] = 0

            only_symptoms_final["exgdis"][
                (only_symptoms_final["gdis"] == 1) == (only_symptoms_final["exgdis"].isna())] = -1
            only_symptoms_final["exgdis"][
                (list(only_symptoms_final["gdis"] == 2)) and list((only_symptoms_final["exgdis"].isna()))] = 0
            only_symptoms_final["exgdis"][only_symptoms_final["exgdis"].isna()] = 0

            only_symptoms_final["nnsym"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["nnsym"].isna())] = -1
            only_symptoms_final["nnsym"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["nnsym"].isna()))] = 0
            only_symptoms_final["nnsym"][only_symptoms_final["nnsym"].isna()] = 0

            only_symptoms_final["addd"][
                (only_symptoms_final["nnsym"] == 1) == (only_symptoms_final["addd"].isna())] = -1
            only_symptoms_final["addd"][
                (list(only_symptoms_final["nnsym"] == 2)) and list((only_symptoms_final["addd"].isna()))] = 0
            only_symptoms_final["addd"][only_symptoms_final["addd"].isna()] = 0

            only_symptoms_final["hypogon"][
                (only_symptoms_final["nnsym"] == 1) == (only_symptoms_final["hypogon"].isna())] = -1
            only_symptoms_final["hypogon"][
                (list(only_symptoms_final["nnsym"] == 2)) and list((only_symptoms_final["hypogon"].isna()))] = 0
            only_symptoms_final["hypogon"][only_symptoms_final["hypogon"].isna()] = 0

            only_symptoms_final["pd"][
                (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["pd"].isna())] = -1
            only_symptoms_final["pd"][
                (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["pd"].isna()))] = 0
            only_symptoms_final["pd"][only_symptoms_final["pd"].isna()] = 0
        except Exception as e:
            print("Block entry doesnt work")
            print(e)
        only_symptoms_final.drop("visit1_fir", 1)
        only_symptoms_final.drop("examination_data_use_new_sheet_for_every_visit_complete", 1)
        only_symptoms_final = self._fill_missing(only_symptoms_final)
        sc = StandardScaler()

        only_symptoms_final = pd.DataFrame(sc.fit_transform(only_symptoms_final.values))

        only_symptoms_final.insert(loc=0, column='label', value=labels)

        print("Data preparation is complete")
        return only_symptoms_final

    def _fill_missing(self, df):
        imputer = IterativeImputer(estimator=RandomForestRegressor(),  missing_values=-1, max_iter=5)
        data_real = df
        # del data_real[data_real.columns[0]]
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
