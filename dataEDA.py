import pandas as pd

df1 = pd.read_csv('온도습도자외선지수자료.csv')
df2test = pd.read_csv('오존이산화질소자료.csv')
df1.drop(df1.columns[0], axis = 1, inplace=True)
df1
df1[df1.columns[1]] = round(df1[df1.columns[1]],2)
df1[df1.columns[2]] = round(df1[df1.columns[2]],2)
df1
df1.loc[[0]]
dffinal1 = pd.DataFrame()

for i in range(0,int(df1.shape[0]/2)):
    dffinal1 = pd.concat([dffinal1, df1.loc[[2*i]]])

dffinal1.reset_index(inplace=True)
dffinal1.drop('index', axis = 1, inplace=True)
dffinal1
dffinal1 = dffinal1.replace('사상', '사상구')
dffinal1 = dffinal1.replace('기장구', '기장군')
dffinal1
df2test
regions = ['영도구', '부산진구', '남구', '북구', '해운대구', '사하구', '금정구', '사상구', '기장군']

df2 = pd.DataFrame()

for i in regions:
    df = df2test[df2test['지점명'] == i]
    df2 = pd.concat([df2,df])

df2.reset_index(inplace=True)
df2.drop('index', axis=1, inplace=True)
df2
df2_morning = pd.DataFrame()

for i in range(0,int(df2.shape[0]/2)):
    df2_morning = pd.concat([df2_morning, df2.loc[[2*i]]])

df2_morning.reset_index(inplace=True)
df2_morning.drop('index', axis = 1, inplace=True)
df2_morning
df2_morning.rename(columns={'오존':'오전 오존 농도'}, inplace=True)
df2_morning
df2_after = pd.DataFrame()

for i in range(0,int(df2.shape[0]/2)):
    df2_after = pd.concat([df2_after, df2.loc[[2*i+1]]])

df2_after.reset_index(inplace=True)
df2_after.drop('index', axis = 1, inplace=True)
df2_after
df2_after.drop('이산화질소', axis = 1, inplace=True)
df2_after
df2_after.rename(columns={'오존':'오후 오존 농도'}, inplace=True)
df2_after
dffinal1
df2_after
df2_morning
df2_final = pd.concat([df2_morning,df2_after], axis=1)
df2_final.drop(['일시', '지점명'], axis = 1, inplace = True)
df2_final
regions = ['영도구', '부산진구', '남구', '북구', '해운대구', '사하구', '금정구', '사상구', '기장군']

df1_final = pd.DataFrame()

for i in regions:
    df = dffinal1[dffinal1['지점명'] == i]
    df1_final = pd.concat([df1_final,df])

df1_final.reset_index(inplace=True)
df1_final.drop('index', axis=1, inplace=True)
df1_final
df = pd.concat([df1_final, df2_final], axis = 1)
df
len(pd.date_range('2022-04-15', '2022-10-15', freq='D'))*9

realdf = pd.DataFrame()

for i in regions:
    x = df[df['지점명'] == i]
    x.drop('일시', axis = 1, inplace=True)
    x.set_index(pd.date_range('2022-04-15', '2022-10-15', freq='D'), inplace=True)
    realdf = pd.concat([realdf, x])

realdf

realdf.rename(columns={realdf.columns[0]:'오전 기온 평균', realdf.columns[1]:'오전 습도 평균', realdf.columns[5]:'오전 이산회질소 농도 평균'}, inplace=True)
realdf
realdf.to_csv('데이터최종.csv')