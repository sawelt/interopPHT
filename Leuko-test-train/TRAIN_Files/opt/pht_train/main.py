# evaluate knn imputation and random forest for the horse colic dataset
import math
import os
import sys
import glob

import numpy as np
import pandas as pd
import neurolab as nl

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from minio import Minio

DATA_PATH ="../train_data"
RESULT_PATH = "../pht_results"


def getDataframe():
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
        data_files_list = glob.glob(f"{DATA_PATH}/*")
        if len(data_files_list) != 1:
            raise RuntimeError(f"one file expected but fund {len(data_files_list)}")
        else:
            try:
                df = pd.read_csv(data_files_list[0])
            except IOError:
                print("CSV data not accessible")
    return df


df = getDataframe()


match = lambda a, b: [b.index(x) + 1 if x in b else None for x in a]

record_ids = df["record_id"][pd.isna(df["diagnosed_leuk"]) == False]

matched_record_ids = list(match(list(df["record_id"]), list(record_ids)))

matched_record_ids_none = []

number = 0

for x in matched_record_ids:
    if x is not None:
        matched_record_ids_none.append(number)
    number = number + 1

df_labels = df.iloc[matched_record_ids_none, :]

df_only_labels = df_labels[df_labels["redcap_repeat_instrument"].isna()]

rri_list = df_labels["redcap_repeat_instrument"] == "examination_data_use_new_sheet_for_every_visit"

exam_numbers = []
number = 0
for x in rri_list:
    if x is True:
        exam_numbers.append(number)
    number = number + 1

symptom_df = df_labels.iloc[exam_numbers, :]

#matched_record_ids_labels = match(list(df_only_labels["record_id"]), list(symptom_df["record_id"]))
#matched_record_ids_none_2 = []
#number = 0
#for x in matched_record_ids_labels:
#    if x is not None:
#        matched_record_ids_none_2.append(number)
#    number = number + 1

#labels = list(df_only_labels.iloc[matched_record_ids_none_2, :]["diagnosed_leuk"])
# Replace with next line
labels = list(df_only_labels["diagnosed_leuk"])



#rri_list_2 = symptom_df["redcap_repeat_instance"] == 1

#exam_numbers = []
#number = 0
#for x in rri_list_2:
#    if x is True:
#        exam_numbers.append(number)
#    number = number + 1

#symptom_first_visit = symptom_df.iloc[exam_numbers, :]
# Replace with next line
symptom_first_visit = symptom_df.loc[symptom_df['redcap_repeat_instance'] == 1]

#first_visit_col_number = (symptom_first_visit.columns == "visit1_fir")

#number = 0
#for x in first_visit_col_number:
#    if x == True:
#        exam_number = number
#    number = number + 1

#only_symptoms = symptom_first_visit.iloc[:, exam_number:]
# Replace with next line
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

only_symptoms_final["visit1_fir"][only_symptoms_final["visit1_fir"].isna()] = -1 #TODO klären

only_symptoms_final["cog"][only_symptoms_final["cog"].isna()] = 0


try:
    only_symptoms_final["apha"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["apha"].isna())] = -1
    only_symptoms_final["apha"][
        (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["apha"].isna()))] = 0
    only_symptoms_final["apha"][only_symptoms_final["apha"].isna()] = 0

    # TODO was passiert bei cog == 3

    only_symptoms_final["cogloss"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["cogloss"].isna())] = -1
    only_symptoms_final["cogloss"][
        (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["cogloss"].isna()))] = 0
    only_symptoms_final["cogloss"][only_symptoms_final["cogloss"].isna()] = 0

    only_symptoms_final["eap"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["eap"].isna())] = -1
    only_symptoms_final["eap"][
        (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["eap"].isna()))] = 0
    only_symptoms_final["eap"][only_symptoms_final["eap"].isna()] = 0

    only_symptoms_final["loc"][(only_symptoms_final["cogloss"] == 1) == (only_symptoms_final["eap"].isna())] = -1
    only_symptoms_final["loc"][
        (list(only_symptoms_final["cogloss"] == 2)) and list((only_symptoms_final["eap"].isna()))] = 0
    only_symptoms_final["loc"][only_symptoms_final["loc"].isna()] = 0

    only_symptoms_final["ic"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["ic"].isna())] = -1
    only_symptoms_final["ic"][(list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["ic"].isna()))] = 0
    only_symptoms_final["ic"][only_symptoms_final["ic"].isna()] = 0

    only_symptoms_final["ii"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["ii"].isna())] = -1
    only_symptoms_final["ii"][(list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["ii"].isna()))] = 0
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
        (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["red_consciousness_confus"].isna()))] = 0
    only_symptoms_final["red_consciousness_confus"][only_symptoms_final["red_consciousness_confus"].isna()] = 0

    only_symptoms_final["agnosia"][(only_symptoms_final["cog"] == 1) == (only_symptoms_final["agnosia"].isna())] = -1
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
        (list(only_symptoms_final["cog"] == 2)) and list((only_symptoms_final["hallucinations_delusions"].isna()))] = 0
    only_symptoms_final["hallucinations_delusions"][only_symptoms_final["hallucinations_delusions"].isna()] = 0

    only_symptoms_final["sleep_disturbance"][only_symptoms_final["sleep_disturbance"].isna()] = -1

    only_symptoms_final["mab"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["mab"].isna())] = -1
    only_symptoms_final["mab"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["mab"].isna()))] = 0
    only_symptoms_final["mab"][only_symptoms_final["mab"].isna()] = 0

    only_symptoms_final["adh"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["adh"].isna())] = -1
    only_symptoms_final["adh"][
        (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["adh"].isna()))] = 0
    only_symptoms_final["adh"][only_symptoms_final["adh"].isna()] = 0

    # only_symptoms_final["dbfb"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["dbfb"].isna())] = -1
    # only_symptoms_final["dbfb"][(list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["dbfb"].isna()))] = 0
    # only_symptoms_final["dbfb"][only_symptoms_final["dbfb"].isna()] = 0

    only_symptoms_final["depr"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["depr"].isna())] = -1
    only_symptoms_final["depr"][
        (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["depr"].isna()))] = 0
    only_symptoms_final["depr"][only_symptoms_final["depr"].isna()] = 0

    only_symptoms_final["ma"][(only_symptoms_final["mab"] == 1) == (only_symptoms_final["ma"].isna())] = -1
    only_symptoms_final["ma"][(list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["ma"].isna()))] = 0
    only_symptoms_final["ma"][only_symptoms_final["ma"].isna()] = 0

    only_symptoms_final["personality"][
        (only_symptoms_final["mab"] == 1) == (only_symptoms_final["personality"].isna())] = -1
    only_symptoms_final["personality"][
        (list(only_symptoms_final["mab"] == 2)) and list((only_symptoms_final["personality"].isna()))] = 0
    only_symptoms_final["personality"][only_symptoms_final["personality"].isna()] = 0

    only_symptoms_final["s_e"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["s_e"].isna())] = -1
    only_symptoms_final["s_e"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["s_e"].isna()))] = 0
    only_symptoms_final["s_e"][only_symptoms_final["s_e"].isna()] = 0

    # only_symptoms_final["fs"][(only_symptoms_final["s_e"] == 1) == (only_symptoms_final["fs"].isna())] = -1
    # only_symptoms_final["fs"][(list(only_symptoms_final["s_e"] == 2)) and list((only_symptoms_final["fs"].isna()))] = 0
    # only_symptoms_final["fs"][only_symptoms_final["fs"].isna()] = 0

    only_symptoms_final["fs___2"][(only_symptoms_final["s_e"] == 1) == (only_symptoms_final["fs___2"].isna())] = -1
    only_symptoms_final["fs___2"][
        (list(only_symptoms_final["s_e"] == 2)) and list((only_symptoms_final["fs___2"].isna()))] = 0
    only_symptoms_final["fs___2"][only_symptoms_final["fs___2"].isna()] = 0

    # only_symptoms_final["gs"][(only_symptoms_final["s_e"] == 1) == (only_symptoms_final["gs"].isna())] = -1
    # only_symptoms_final["gs"][(list(only_symptoms_final["s_e"] == 2)) and list((only_symptoms_final["gs"].isna()))] = 0
    # only_symptoms_final["gs"][only_symptoms_final["gs"].isna()] = 0

    only_symptoms_final["emd"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["emd"].isna())] = -1
    only_symptoms_final["emd"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["emd"].isna()))] = 0
    only_symptoms_final["emd"][only_symptoms_final["emd"].isna()] = 0

    only_symptoms_final["diplopia"][(only_symptoms_final["emd"] == 1) == (only_symptoms_final["diplopia"].isna())] = -1
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

    only_symptoms_final["oculomot"][(only_symptoms_final["emd"] == 1) == (only_symptoms_final["oculomot"].isna())] = -1
    only_symptoms_final["oculomot"][
        (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["oculomot"].isna()))] = 0
    only_symptoms_final["oculomot"][only_symptoms_final["oculomot"].isna()] = 0

    only_symptoms_final["fourth_cranial_nerve_palsy"][
        (only_symptoms_final["emd"] == 1) == (only_symptoms_final["fourth_cranial_nerve_palsy"].isna())] = -1
    only_symptoms_final["fourth_cranial_nerve_palsy"][(list(only_symptoms_final["emd"] == 2)) and list(
        (only_symptoms_final["fourth_cranial_nerve_palsy"].isna()))] = 0
    only_symptoms_final["fourth_cranial_nerve_palsy"][only_symptoms_final["fourth_cranial_nerve_palsy"].isna()] = 0

    only_symptoms_final["abducens"][(only_symptoms_final["emd"] == 1) == (only_symptoms_final["abducens"].isna())] = -1
    only_symptoms_final["abducens"][
        (list(only_symptoms_final["emd"] == 2)) and list((only_symptoms_final["abducens"].isna()))] = 0
    only_symptoms_final["abducens"][only_symptoms_final["ino"].isna()] = 0

    only_symptoms_final["thy"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["thy"].isna())] = -1
    only_symptoms_final["thy"][
        (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["thy"].isna()))] = 0
    only_symptoms_final["thy"][only_symptoms_final["thy"].isna()] = 0

    only_symptoms_final["fp"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["fp"].isna())] = -1
    only_symptoms_final["fp"][(list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["fp"].isna()))] = 0
    only_symptoms_final["fp"][only_symptoms_final["fp"].isna()] = 0

    only_symptoms_final["od"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["od"].isna())] = -1
    only_symptoms_final["od"][(list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["od"].isna()))] = 0
    only_symptoms_final["od"][only_symptoms_final["od"].isna()] = 0

    only_symptoms_final["hi"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["hi"].isna())] = -1
    only_symptoms_final["hi"][(list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["hi"].isna()))] = 0
    only_symptoms_final["hi"][only_symptoms_final["hi"].isna()] = 0

    only_symptoms_final["hp"][(only_symptoms_final["crn"] == 1) == (only_symptoms_final["hp"].isna())] = -1
    only_symptoms_final["hp"][(list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["hp"].isna()))] = 0
    only_symptoms_final["hp"][only_symptoms_final["hp"].isna()] = 0

    only_symptoms_final["trig_neur"][
        (only_symptoms_final["crn"] == 1) == (only_symptoms_final["trig_neur"].isna())] = -1
    only_symptoms_final["trig_neur"][
        (list(only_symptoms_final["crn"] == 2)) and list((only_symptoms_final["trig_neur"].isna()))] = 0
    only_symptoms_final["trig_neur"][only_symptoms_final["trig_neur"].isna()] = 0

    only_symptoms_final["spsw"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["spsw"].isna())] = -1
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

    only_symptoms_final["dysphon"][(only_symptoms_final["spsw"] == 1) == (only_symptoms_final["dysphon"].isna())] = -1
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
        (list(only_symptoms_final["spsw"] == 2)) and list((only_symptoms_final["pseudobulbar_palsy"].isna()))] = 0
    only_symptoms_final["pseudobulbar_palsy"][only_symptoms_final["pseudobulbar_palsy"].isna()] = 0

    only_symptoms_final["dyp"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["dyp"].isna())] = -1
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

    # only_symptoms_final["visual_field_defect"][(only_symptoms_final["visi"] == 1) == (only_symptoms_final["visual_field_defect"].isna())] = -1
    # only_symptoms_final["visual_field_defect"][(list(only_symptoms_final["visi"] == 2)) and list((only_symptoms_final["visual_field_defect"].isna()))] = 0
    # only_symptoms_final["visual_field_defect"][only_symptoms_final["visual_field_defect"].isna()] = 0

    only_symptoms_final["sim"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["sim"].isna())] = -1
    only_symptoms_final["sim"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["sim"].isna()))] = 0
    only_symptoms_final["sim"][only_symptoms_final["sim"].isna()] = 0

    only_symptoms_final["sper"][(only_symptoms_final["sim"] == 1) == (only_symptoms_final["sper"].isna())] = -1
    only_symptoms_final["sper"][
        (list(only_symptoms_final["sim"] == 2)) and list((only_symptoms_final["sper"].isna()))] = 0
    only_symptoms_final["sper"][only_symptoms_final["sper"].isna()] = 0

    only_symptoms_final["vs"][(only_symptoms_final["sim"] == 1) == (only_symptoms_final["vs"].isna())] = -1
    only_symptoms_final["vs"][(list(only_symptoms_final["sim"] == 2)) and list((only_symptoms_final["vs"].isna()))] = 0
    only_symptoms_final["vs"][only_symptoms_final["vs"].isna()] = 0

    # only_symptoms_final["sensory_impa"][(only_symptoms_final["sim"] == 1) == (only_symptoms_final["sensory_impa"].isna())] = -1
    # only_symptoms_final["sensory_impa"][(list(only_symptoms_final["sim"] == 2)) and list((only_symptoms_final["sensory_impa"].isna()))] = 0
    # only_symptoms_final["sensory_impa"][only_symptoms_final["sensory_impa"].isna()] = 0

    only_symptoms_final["cersy"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["cersy"].isna())] = -1
    only_symptoms_final["cersy"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["cersy"].isna()))] = 0
    only_symptoms_final["cersy"][only_symptoms_final["cersy"].isna()] = 0

    only_symptoms_final["trem"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["trem"].isna())] = -1
    only_symptoms_final["trem"][
        (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["trem"].isna()))] = 0
    only_symptoms_final["trem"][only_symptoms_final["trem"].isna()] = 0

    # only_symptoms_final["hye"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["hye"].isna())] = -1
    # only_symptoms_final["hye"][(list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["hye"].isna()))] = 0
    # only_symptoms_final["hye"][only_symptoms_final["hye"].isna()] = 0

    only_symptoms_final["hyo"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["hyo"].isna())] = -1
    only_symptoms_final["hyo"][
        (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["hyo"].isna()))] = 0
    only_symptoms_final["hyo"][only_symptoms_final["hyo"].isna()] = 0

    only_symptoms_final["dyt"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["dyt"].isna())] = -1
    only_symptoms_final["dyt"][
        (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["dyt"].isna()))] = 0
    only_symptoms_final["dyt"][only_symptoms_final["dyt"].isna()] = 0

    # only_symptoms_final["dyskin"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["dyskin"].isna())] = -1
    # only_symptoms_final["dyskin"][(list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["dyskin"].isna()))] = 0
    # only_symptoms_final["dyskin"][only_symptoms_final["dyskin"].isna()] = 0

    only_symptoms_final["fmd"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["fmd"].isna())] = -1
    only_symptoms_final["fmd"][
        (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["fmd"].isna()))] = 0
    only_symptoms_final["fmd"][only_symptoms_final["fmd"].isna()] = 0

    only_symptoms_final["ataxia"][(only_symptoms_final["cersy"] == 1) == (only_symptoms_final["ataxia"].isna())] = -1
    only_symptoms_final["ataxia"][
        (list(only_symptoms_final["cersy"] == 2)) and list((only_symptoms_final["ataxia"].isna()))] = 0
    only_symptoms_final["ataxia"][only_symptoms_final["ataxia"].isna()] = 0

    only_symptoms_final["bd"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["bd"].isna())] = -1
    only_symptoms_final["bd"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["bd"].isna()))] = 0
    only_symptoms_final["bd"][only_symptoms_final["bd"].isna()] = 0

    only_symptoms_final["sexd"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["sexd"].isna())] = -1
    only_symptoms_final["sexd"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["sexd"].isna()))] = 0
    only_symptoms_final["sexd"][only_symptoms_final["sexd"].isna()] = 0

    only_symptoms_final["edy"][(only_symptoms_final["sexd"] == 1) == (only_symptoms_final["edy"].isna())] = -1
    only_symptoms_final["edy"][
        (list(only_symptoms_final["sexd"] == 2)) and list((only_symptoms_final["edy"].isna()))] = 0
    only_symptoms_final["edy"][only_symptoms_final["edy"].isna()] = 0

    only_symptoms_final["ll"][(only_symptoms_final["sexd"] == 1) == (only_symptoms_final["ll"].isna())] = -1
    only_symptoms_final["ll"][(list(only_symptoms_final["sexd"] == 2)) and list((only_symptoms_final["ll"].isna()))] = 0
    only_symptoms_final["ll"][only_symptoms_final["ll"].isna()] = 0

    only_symptoms_final["bi"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["bi"].isna())] = -1
    only_symptoms_final["bi"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["bi"].isna()))] = 0
    only_symptoms_final["bi"][only_symptoms_final["bi"].isna()] = 0

    only_symptoms_final["prs"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["prs"].isna())] = -1
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
        (list(only_symptoms_final["tprs"] == 2)) and list((only_symptoms_final["severity_of_paresis"].isna()))] = 0
    only_symptoms_final["severity_of_paresis"][only_symptoms_final["severity_of_paresis"].isna()] = 0

    only_symptoms_final["psi"][(only_symptoms_final["prs"] == 1) == (only_symptoms_final["psi"].isna())] = -1
    only_symptoms_final["psi"][
        (list(only_symptoms_final["prs"] == 2)) and list((only_symptoms_final["psi"].isna()))] = 0
    only_symptoms_final["psi"][only_symptoms_final["psi"].isna()] = 0

    only_symptoms_final["spas"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["spas"].isna())] = -1
    only_symptoms_final["spas"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["spas"].isna()))] = 0
    only_symptoms_final["spas"][only_symptoms_final["spas"].isna()] = 0

    only_symptoms_final["tspas"][(only_symptoms_final["spas"] == 1) == (only_symptoms_final["tspas"].isna())] = -1
    only_symptoms_final["tspas"][
        (list(only_symptoms_final["spas"] == 2)) and list((only_symptoms_final["tspas"].isna()))] = 0
    only_symptoms_final["tspas"][only_symptoms_final["tspas"].isna()] = 0

    # only_symptoms_final["pai"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["pai"].isna())] = -1
    # only_symptoms_final["pai"][(list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["pai"].isna()))] = 0
    # only_symptoms_final["pai"][only_symptoms_final["pai"].isna()] = 0
    #
    # only_symptoms_final["npa"][(only_symptoms_final["pai"] == 1) == (only_symptoms_final["npa"].isna())] = -1
    # only_symptoms_final["npa"][(list(only_symptoms_final["pai"] == 2)) and list((only_symptoms_final["npa"].isna()))] = 0
    # only_symptoms_final["npa"][only_symptoms_final["npa"].isna()] = 0
    #
    # only_symptoms_final["headache"][(only_symptoms_final["pai"] == 1) == (only_symptoms_final["headache"].isna())] = -1
    # only_symptoms_final["headache"][(list(only_symptoms_final["pai"] == 2)) and list((only_symptoms_final["headache"].isna()))] = 0
    # only_symptoms_final["headache"][only_symptoms_final["headache"].isna()] = 0

    only_symptoms_final["vertigo_dizziness"][
        (only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["vertigo_dizziness"].isna())] = -1
    only_symptoms_final["vertigo_dizziness"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["vertigo_dizziness"].isna()))] = 0
    only_symptoms_final["vertigo_dizziness"][only_symptoms_final["vertigo_dizziness"].isna()] = 0

    only_symptoms_final["type_of_dizziness"][
        (only_symptoms_final["vertigo_dizziness"] == 1) == (only_symptoms_final["type_of_dizziness"].isna())] = -1
    only_symptoms_final["type_of_dizziness"][(list(only_symptoms_final["vertigo_dizziness"] == 2)) and list(
        (only_symptoms_final["type_of_dizziness"].isna()))] = 0
    only_symptoms_final["type_of_dizziness"][only_symptoms_final["type_of_dizziness"].isna()] = 0

    only_symptoms_final["gdis"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["gdis"].isna())] = -1
    only_symptoms_final["gdis"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["gdis"].isna()))] = 0
    only_symptoms_final["gdis"][only_symptoms_final["gdis"].isna()] = 0

    # only_symptoms_final["gait_imbalance"][(only_symptoms_final["gdis"] == 1) == (only_symptoms_final["gait_imbalance"].isna())] = -1
    # only_symptoms_final["gait_imbalance"][(list(only_symptoms_final["gdis"] == 2)) and list((only_symptoms_final["gait_imbalance"].isna()))] = 0
    # only_symptoms_final["gait_imbalance"][only_symptoms_final["gait_imbalance"].isna()] = 0

    only_symptoms_final["exgdis"][(only_symptoms_final["gdis"] == 1) == (only_symptoms_final["exgdis"].isna())] = -1
    only_symptoms_final["exgdis"][
        (list(only_symptoms_final["gdis"] == 2)) and list((only_symptoms_final["exgdis"].isna()))] = 0
    only_symptoms_final["exgdis"][only_symptoms_final["exgdis"].isna()] = 0

    only_symptoms_final["nnsym"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["nnsym"].isna())] = -1
    only_symptoms_final["nnsym"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["nnsym"].isna()))] = 0
    only_symptoms_final["nnsym"][only_symptoms_final["nnsym"].isna()] = 0

    only_symptoms_final["addd"][(only_symptoms_final["nnsym"] == 1) == (only_symptoms_final["addd"].isna())] = -1
    only_symptoms_final["addd"][
        (list(only_symptoms_final["nnsym"] == 2)) and list((only_symptoms_final["addd"].isna()))] = 0
    only_symptoms_final["addd"][only_symptoms_final["addd"].isna()] = 0

    only_symptoms_final["hypogon"][(only_symptoms_final["nnsym"] == 1) == (only_symptoms_final["hypogon"].isna())] = -1
    only_symptoms_final["hypogon"][
        (list(only_symptoms_final["nnsym"] == 2)) and list((only_symptoms_final["hypogon"].isna()))] = 0
    only_symptoms_final["hypogon"][only_symptoms_final["hypogon"].isna()] = 0

    only_symptoms_final["pd"][(only_symptoms_final["visit1_fir"] == 1) == (only_symptoms_final["pd"].isna())] = -1
    only_symptoms_final["pd"][
        (list(only_symptoms_final["visit1_fir"] == 2)) and list((only_symptoms_final["pd"].isna()))] = 0
    only_symptoms_final["pd"][only_symptoms_final["pd"].isna()] = 0
except Exception as e:
    print("Block entry doesnt work")
    print(e)
only_symptoms_final.drop("visit1_fir", 1)
only_symptoms_final.drop("examination_data_use_new_sheet_for_every_visit_complete", 1)

only_symptoms_final.insert(loc=0, column='label', value=labels)

print("Data preparation is complete")


def quarter(x):
    return math.ceil(x * 4) / 4


def ANN(x, y, xt, yt):
    size = len(x)
    ##########################################
    # x = sklearn.preprocessing.normalize(x, norm="l1")
    # xt = sklearn.preprocessing.normalize([xt], norm="l1")
    # scaler_x = MinMaxScaler(feature_range=(0, 1))
    # x = pd. DataFrame(scaler_x.fit_transform(x))
    # xt = pd. DataFrame(scaler_x.fit_transform([xt]))
    # scaler_y = MinMaxScaler(feature_range=(0, 1))
    # y = pd. DataFrame(scaler_y.fit_transform([y]))
    # yt = pd. DataFrame(scaler_y.fit_transform([[yt]]))
    maxmin = []
    for i in range(0, 100):
        maxmin.append([0, 1])
    ##########################################
    inp = x  # .reshape(size,1)
    tar = y.reshape(size, 1)

    # Create network with 2 layers and random initialized
    net = nl.net.newff(maxmin, [20, 1])

    # Train network
    error = net.train(inp, tar, epochs=5000, show=100, goal=0.01)

    # Simulate network
    out = net.sim(inp)

    # Plot result

    # pl.subplot(211)
    # pl.plot(error)
    # pl.xlabel('Epoch number')
    # pl.ylabel('error (default SSE)')

    # x2 = xt#np.linspace(-6.0,6.0,150)
    ytt = net.sim([xt])
    return ytt
    ytt = np.round(ytt)
    yttn = []
    for item in ytt:
        if item[0] == 0:
            yttn.append(0)
        else:
            yttn.append(1)
    return len([a for a in np.isclose(yttn, yt) if (a)]) / len(yttn) * 100


imputer = MissForest(missing_values=-1)
data_real = only_symptoms_final
# del data_real[data_real.columns[0]]
data_imputed = imputer.fit_transform(data_real)
data = pd.DataFrame(data=data_imputed, columns=data_real.columns.values.tolist())

print("Data imputation is complete")

# data = pd.read_csv('./output.csv')
# normalize dataset with MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
# data = pd.DataFrame(scaler.fit_transform(realData))
linear_result = []
rbf_result = []
poly_result = []
sig_result = []
for testIndex in range(len(data)):
    # data_temp=data
    # train=data.drop([testIndex])
    # test=data.iloc[testIndex]
    data_x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    # y_class1=y
    """""
    y_class1=y.replace(2,1)
    y_class1=y_class1.replace(103,0)
    y_class1=y_class1.replace(7,0)
    y_class1=y_class1.replace(84,0)

    y_class2=y.replace(2,0)
    y_class2=y_class2.replace(103,1)
    y_class2=y_class2.replace(7,0)
    y_class2=y_class2.replace(84,0)
    """

    # 2, 84 and 103 realted to LD
    y_class = y.replace(2, 0)
    y_class = y_class.replace(103, 0)
    y_class = y_class.replace(1, 0)
    y_class = y_class.replace(84, 1)
    # 7 is related to MS
    y_class = y_class.replace(7, 0)

    y_class = y_class.replace(29, 0)
    y_class = y_class.replace(60, 0)
    """""
    y_class4=y.replace(2,0)
    y_class4=y_class4.replace(103,0)
    y_class4=y_class4.replace(7,0)
    y_class4=y_class4.replace(84,1)
    """
    sc = StandardScaler()
    x_scaled = pd.DataFrame(sc.fit_transform(data_x.values))
    xt_scaled = x_scaled.iloc[testIndex]
    x_scaled = x_scaled.drop([testIndex])

    # y_class1=y_class1.drop([testIndex])
    # y_class2=y_class2.drop([testIndex])
    y_class = y_class.drop([testIndex])
    # y_class4=y_class4.drop([testIndex])
    yt = data.iloc[testIndex].iloc[1]

    # y_class1= np.ravel(y_class1)
    # y_class2= np.ravel(y_class2)
    y_class = np.ravel(y_class)
    # y_class4= np.ravel(y_class4)
    # svm_class1 = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo', random_state=0)
    # svm_class2 = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo', random_state=0)
    svm_class_sigmoid = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo', random_state=0, probability=True)
    svm_class_linear = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=0, probability=True)
    svm_class_poly = SVC(kernel='poly', C=1, decision_function_shape='ovo', random_state=0, probability=True)
    svm_class_rbf = SVC(kernel='rbf', C=1, decision_function_shape='ovo', random_state=0, probability=True)
    # svm_class4 = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo', random_state=0)
    # svm_class1.fit(x_scaled, y_class1)
    # svm_class2.fit(x_scaled, y_class2)
    svm_class_sigmoid.fit(x_scaled, y_class)
    svm_class_linear.fit(x_scaled, y_class)
    svm_class_poly.fit(x_scaled, y_class)
    svm_class_rbf.fit(x_scaled, y_class)
    # svm_class4.fit(x_scaled, y_class4)
    # y_pre_class1 = svm_class1.predict([xt_scaled])
    # y_pre_class2 = svm_class2.predict([xt_scaled])
    y_pre_class_sigmoid = svm_class_sigmoid.predict([xt_scaled])
    y_pre_class_linear = svm_class_linear.predict([xt_scaled])
    y_pre_class_poly = svm_class_poly.predict([xt_scaled])
    y_pre_class_rbf = svm_class_rbf.predict([xt_scaled])
    # y_pre_class4 = svm_class4.predict([xt_scaled])
    # linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x, y)
    # rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(x, y)
    # poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(x, y)
    # sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(x, y)
    # linear_result.append([linear.predict([xt])[0],yt])
    if y_pre_class_sigmoid[0] == 1:
        sig_result.append([84, yt])
    else:
        sig_result.append([0, yt])

    if y_pre_class_linear[0] == 1:
        linear_result.append([84, yt])
    else:
        linear_result.append([0, yt])

    if y_pre_class_poly[0] == 1:
        poly_result.append([84, yt])
    else:
        poly_result.append([0, yt])

    if y_pre_class_rbf[0] == 1:
        rbf_result.append([84, yt])
    else:
        rbf_result.append([0, yt])

    #    if y_pre_class1[0]==0 and y_pre_class2[0]==1 and y_pre_class3[0]==0 and y_pre_class4[0]==0:
    #        SVM_result.append([103,yt])
    #    else:
    #         if y_pre_class1[0]==0 and y_pre_class2[0]==0 and y_pre_class3[0]==1 and y_pre_class4[0]==0:
    #             SVM_result.append([7,yt])
    #         else:
    #             if y_pre_class1[0]==0 and y_pre_class2[0]==0 and y_pre_class3[0]==0 and y_pre_class4[0]==1:
    #                 SVM_result.append([84,yt])
    #             else:
    #                 SVM_result.append([0,yt])
    # poly_result.append([poly.predict([xt])[0],yt])
    # sig_result.append([sig.predict([xt])[0],yt])
# Percent_SVM=0
# Percent_poly=0
# Percent_ANN=0
if not os.path.isdir(RESULT_PATH):
    raise RuntimeError(f'result folder {RESULT_PATH} dose not exist ')
pd.DataFrame(linear_result).to_csv(f'{RESULT_PATH}/binary_linear.csv')
pd.DataFrame(rbf_result).to_csv(f'{RESULT_PATH}/binary_rbf.csv')
pd.DataFrame(poly_result).to_csv(f'{RESULT_PATH}/binary_poly.csv')
pd.DataFrame(sig_result).to_csv(f'{RESULT_PATH}/binary_sig.csv')

Percent_linear = 0
Percent_poly = 0
Percent_rbf = 0
Percent_sig = 0
for index in range(len(data)):
    if linear_result[index][0] != linear_result[index][1]: Percent_linear = Percent_linear + 1
    if rbf_result[index][0] != rbf_result[index][1]: Percent_rbf = Percent_rbf + 1
    if poly_result[index][0] != poly_result[index][1]: Percent_poly = Percent_poly + 1
    if poly_result[index][0] != poly_result[index][1]: Percent_sig = Percent_sig + 1
Percent_linear = Percent_linear / len(data)
Percent_poly = Percent_poly / len(data)
Percent_rbf = Percent_rbf / len(data)
Percent_sig = Percent_sig / len(data)
print(Percent_linear)
print(Percent_poly)
print(Percent_rbf)
print(Percent_sig)
print("Model is finish")
print("SUCCESS")
"""""
data=pd.DataFrame( data[data.iloc[:, 1]!=84].values)
linear_result=[]
rbf_result=[]
poly_result=[]
sig_result=[]
#for index in range(len(data)):
#    if SVM_result[index][84]==SVM_result[index][1]:Percent_SVM=Percent_SVM+1
for testIndex in range( len(data)):
    train=data.drop([testIndex])
    test=data.iloc[testIndex]
    x = train.iloc[:, 2:].values
    y = train.iloc[:, 1].values
    xt = test.iloc[ 2:].values
    yt = test.iloc[ 1]
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x, y)
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(x, y)
    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(x, y)
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(x, y)
    linear_result.append([linear.predict([xt])[0],yt])
    rbf_result.append([rbf.predict([xt])[0],yt])
    poly_result.append([poly.predict([xt])[0],yt])
    sig_result.append([sig.predict([xt])[0],yt])
pd.DataFrame(linear_result).to_csv('multi_class_linear.csv')
pd.DataFrame(rbf_result).to_csv('multi_class_rbf.csv')
pd.DataFrame(poly_result).to_csv('multi_class_poly.csv')
pd.DataFrame(sig_result).to_csv('multi_class_sig.csv')

Percent_linear=Percent_SVM/len(data)
data.iloc[:,1]=data.iloc[:,1].replace(2,0)
data.iloc[:,1]=data.iloc[:,1].replace(103,0.25)
data.iloc[:,1]=data.iloc[:,1].replace(7,0.75)
data.iloc[:,1]=data.iloc[:,1].replace(84,1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = pd.DataFrame(scaler.fit_transform(data))
ANN_result=[]
for testIndex in range( len(data)):
    train=data.drop([testIndex])
    test=data.iloc[testIndex]
    x = train.iloc[:, 2:].values
    y = train.iloc[:, 1].values
    xt = test.iloc[ 2:].values
    yt = test.iloc[ 1]
    pyt= quarter(ANN(x,y,xt,yt))
    ANN_result.append([pyt,yt])
Percent_linear=0
Percent_poly=0
Percent_ANN=0
for index in range(len(data)):
    if linear_result[index][0]==linear_result[index][1]:Percent_linear=Percent_linear+1
    if poly_result[index][0]==poly_result[index][1]:Percent_poly=Percent_poly+1
    if ANN_result[index][0]==ANN_result[index][1]:Percent_ANN=Percent_ANN+1
Percent_linear=Percent_linear/len(data)
Percent_poly=Percent_poly/len(data)
Percent_ANN=Percent_ANN/len(data)
#2	103	7	84
class1=data.drop(np.where(data.iloc[:,1] != 2)[0])
class2=data.drop(np.where(data.iloc[:,1] != 103)[0])
class3=data.drop(np.where(data.iloc[:,1] != 7)[0])
class4=data.drop(np.where(data.iloc[:,1] != 84)[0])
"""
