# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 ```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

 ```
<img width="1657" height="523" alt="Screenshot 2025-10-07 151958" src="https://github.com/user-attachments/assets/9f70a8f8-73c6-4d4b-9eb5-3df853e8e65a" />

```
data.isnull().sum()

```
<img width="452" height="862" alt="Screenshot 2025-10-07 152034" src="https://github.com/user-attachments/assets/d6fd1569-d21d-44a4-8668-3375e740ca92" />

```
missing=data[data.isnull().any(axis=1)]
missing

```
<img width="1451" height="454" alt="Screenshot 2025-10-07 152056" src="https://github.com/user-attachments/assets/c59eb75f-ec39-429f-9d31-d1381f0ab0da" />

```
data2=data.dropna(axis=0)
data2

```
<img width="1453" height="441" alt="Screenshot 2025-10-07 152128" src="https://github.com/user-attachments/assets/b98d6790-7925-4a71-8f3a-f1d8bd0d4c62" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
<img width="1462" height="389" alt="Screenshot 2025-10-07 152145" src="https://github.com/user-attachments/assets/ef7384a5-51c6-4d22-bd7f-7b4d3d0dd7ab" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

```
<img width="638" height="737" alt="Screenshot 2025-10-07 152156" src="https://github.com/user-attachments/assets/1708222e-9822-4698-bcb6-e736258ea377" />

```
data2

```
<img width="1469" height="516" alt="Screenshot 2025-10-07 152216" src="https://github.com/user-attachments/assets/825603b8-57d3-4135-8723-0d5c25c84d2d" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data

```
<img width="1466" height="457" alt="Screenshot 2025-10-07 152235" src="https://github.com/user-attachments/assets/1b3de221-078d-4181-b118-39dea478398b" />

```
columns_list=list(new_data.columns)
print(columns_list)

```
<img width="1460" height="39" alt="Screenshot 2025-10-07 152330" src="https://github.com/user-attachments/assets/2294075a-6b10-4ed7-a1b6-8f1157434782" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)

```
<img width="1463" height="94" alt="Screenshot 2025-10-07 152346" src="https://github.com/user-attachments/assets/82166899-c9aa-474e-945e-58fd13ce4dd8" />

```
y=new_data['SalStat'].values
print(y)

```
<img width="1456" height="82" alt="Screenshot 2025-10-07 152400" src="https://github.com/user-attachments/assets/80f37aec-f95b-4b41-8a18-ba46474fee54" />

```
x=new_data[features].values
print(x)

```
<img width="1452" height="243" alt="Screenshot 2025-10-07 152604" src="https://github.com/user-attachments/assets/aa7e2b88-e145-4bc8-b509-06d427600b9e" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```
<img width="1459" height="155" alt="Screenshot 2025-10-07 152619" src="https://github.com/user-attachments/assets/2eb399e7-f140-436f-a848-820618e67592" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

```
<img width="1458" height="108" alt="Screenshot 2025-10-07 152630" src="https://github.com/user-attachments/assets/9fe3c70c-d9f3-47ef-96b3-50878aedff9f" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
<img width="1458" height="90" alt="Screenshot 2025-10-07 152645" src="https://github.com/user-attachments/assets/491ba920-0d3a-4e4c-ae35-ac4031a4376d" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```
<img width="1456" height="71" alt="Screenshot 2025-10-07 152700" src="https://github.com/user-attachments/assets/9ebb9995-5291-4bf1-8abb-5aae5cd4f24a" />


```
data.shape

```
<img width="1461" height="86" alt="Screenshot 2025-10-07 152715" src="https://github.com/user-attachments/assets/34ea415a-9f2e-47be-bc4a-b9b70fac80ce" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```
<img width="1459" height="99" alt="Screenshot 2025-10-07 152728" src="https://github.com/user-attachments/assets/b9be672d-a2dc-4a20-bb2f-1958ec36978f" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
<img width="1456" height="400" alt="Screenshot 2025-10-07 153814" src="https://github.com/user-attachments/assets/3cab2188-b019-479a-9657-f1a961b5377d" />

```
tips.time.unique()

```
<img width="1446" height="129" alt="Screenshot 2025-10-07 153828" src="https://github.com/user-attachments/assets/bb195d03-8f27-41df-849a-33c866278f9b" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```
<img width="1453" height="165" alt="Screenshot 2025-10-07 153848" src="https://github.com/user-attachments/assets/67c62cb9-1fb3-4924-9db4-30547d5007ff" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="1460" height="121" alt="Screenshot 2025-10-07 153902" src="https://github.com/user-attachments/assets/6f110508-4ad4-496d-b4b0-0402f2811bcc" />




# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
