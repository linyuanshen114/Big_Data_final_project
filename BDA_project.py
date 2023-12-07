import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
scaler = StandardScaler()

#todo:change hardcoding style
def random_forest_train_group(name,group):
    X = group.drop(['district_name', 'damage_grade','damage_level'], axis=1)
    y = group['damage_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)

    y_pred = rf_classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')
def random_forest_train_grade_1(name,group):
    X = group.drop(['district_name']+[x for x in group.columns if "grade" in x or "level" in x]
, axis=1)
    y = group['grade_1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

def random_forest_train_grade_2(name,group):
    X = group.drop(['district_name']+[x for x in group.columns if "grade" in x or "level" in x], axis=1)
    y = group['grade_2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

def random_forest_train_grade_3(name,group):
    X = group.drop(['district_name']+[x for x in group.columns if "grade" in x or "level" in x], axis=1)
    y = group['grade_3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

def random_forest_train_grade_4(name,group):
    X = group.drop(['district_name']+[x for x in group.columns if "grade" in x or "level" in x], axis=1)
    y = group['grade_4']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')
def random_forest_train_grade_5(name,group):
    X = group.drop(['district_name']+[x for x in group.columns if "grade" in x or "level" in x], axis=1)
    y = group['grade_5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

def histogram_grade_group(name,group):
    myfolder="F:\\BDA_datasets\\earthquake_plots\\histogram_damage_grade"
    filename=name+".png"
    file_path=os.path.join(myfolder,filename)
    y = group['damage_grade_raw']
    # plt.style.use('ggplot')
    plt.figure()
    sns.set()
    sns.histplot(y,label="Damage Grade",color="skyblue",edgecolor="salmon",stat="density",lw=2)
    # plt.grid()
    plt.xlabel("Grade")
    plt.ylabel("Density")
    plt.yticks(np.arange(0,1,0.1))
    plt.title(name+" with {} obs".format(group.shape[0]))
    plt.legend()
    plt.savefig(file_path)
    # plt.show()
def random_forest_train_raw(structure_data):
    X = structure_data.drop(['district_name'] + [x for x in structure_data.columns if "grade" in x or "level" in x],
                            axis=1)
    y = structure_data["damage_grade_raw"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=1000, max_depth=3, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy : {accuracy:.2f}')

def try_different_grades(grouped_dataframes):
    print("Classification for Grade 1 ONLY----------------------")
    for name, group in grouped_dataframes:
        random_forest_train_grade_1(name, group)

    print("Classification for Grade 2 ONLY----------------------")
    for name, group in grouped_dataframes:
        random_forest_train_grade_2(name, group)

    print("Classification for Grade 3 ONLY----------------------")
    for name, group in grouped_dataframes:
        random_forest_train_grade_3(name, group)

    print("Classification for Grade 4 ONLY----------------------")
    for name, group in grouped_dataframes:
        random_forest_train_grade_4(name, group)
    print("Classification for Grade 5 ONLY----------------------")
    for name, group in grouped_dataframes:
        random_forest_train_grade_5(name, group)

if __name__=="__main__":
    #edit data root here
    folder="F:\\BDA_datasets\\earthquake"
    filename = "csv_building_structure.csv"
    file_path = os.path.join(folder, filename)

    structure_data = pd.read_csv(file_path)

    # process data
    structure_data.dropna(inplace=True)
    distrct_map = pd.read_csv(os.path.join(folder,"ward_vdcmun_district_name_mapping.csv"))
    structure_data = pd.merge(structure_data, distrct_map, on=['ward_id', 'vdcmun_id', 'district_id'], how='left')
    structure_data = structure_data.drop(['building_id', 'district_id', 'vdcmun_id', 'ward_id', 'count_floors_post_eq',
        'height_ft_post_eq', 'condition_post_eq', 'technical_solution_proposed', 'vdcmun_name'], axis=1)

    #my process
    structure_data['damage_grade_raw']=structure_data['damage_grade']
    structure_data.sort_values(by=['damage_grade_raw'],inplace=True)

    severity_mapping = {'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5}
    structure_data['damage_grade'] = structure_data['damage_grade'].map(severity_mapping)
    columns_to_encode = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                          'other_floor_type', 'position', 'plan_configuration']

    structure_data = pd.get_dummies(structure_data, columns=columns_to_encode)

    #my process
    structure_data['damage_level']=structure_data['damage_grade'].apply(lambda x:1 if x>=3 else 0)
    structure_data['grade_1']=structure_data['damage_grade'].apply(lambda x:1 if x==1 else 0)
    structure_data['grade_2']=structure_data['damage_grade'].apply(lambda x:1 if x==2 else 0)
    structure_data['grade_3']=structure_data['damage_grade'].apply(lambda x:1 if x==3 else 0)
    structure_data['grade_4']=structure_data['damage_grade'].apply(lambda x:1 if x==4 else 0)
    structure_data['grade_5']=structure_data['damage_grade'].apply(lambda x:1 if x==5 else 0)

    grouped_dataframes = structure_data.groupby('district_name')
    try_different_grades(grouped_dataframes)





