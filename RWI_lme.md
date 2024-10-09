

```R
library(dplR)
input_dir <- "E:/experiment/DTR_AI_growth/data/ITRDB/touwenjian_qingli_99"

output_dir <- "E:/experiment/DTR_AI_growth/data/ITRDB/FZS_STD"
files <- list.files(path = input_dir, pattern = "*.rwl", full.names = TRUE)
success_files <- c()
for(file in files) {
  try({

    ym3000 <- read.rwl(file)
    

    ym3000.sum <- summary(ym3000)
    ym3000.rwi <- detrend(rwl = ym3000, method = c('ModNegExp'))
    ym3000.crn <- chron.ars(ym3000.rwi)
    ym3000.ids <- read.ids(ym3000,stc = "auto")
    ym3000.stats <- rwi.stats(ym3000, ids=ym3000.ids)
    ym3000.crn.stats <- rwl.stats(ym3000.crn)
    filename <- basename(file)

    write.table(ym3000.sum, file = file.path(output_dir, paste0(filename, "_sum.csv")), sep=",")
    write.table(ym3000, file = file.path(output_dir, paste0(filename, ".csv")), sep=",")
    write.table(ym3000.crn, file = file.path(output_dir, paste0(filename, "crn.csv")), sep=",")
    write.table(ym3000.stats, file = file.path(output_dir, paste0(filename, ".stats.csv")), sep=",")
    write.table(ym3000.crn.stats, file = file.path(output_dir, paste0(filename, ".crn.stats.csv")), sep=",")
    
    png(file = file.path(output_dir, paste0(filename, ".png")), width = 800 * 300 / 72, height = 600 * 300 / 72, res = 300)
    plot(ym3000.crn, add.spline=TRUE, nyrs=16)
    dev.off()
    
    success_files <- c(success_files, filename)
  }, silent = TRUE)
}

write.table(success_files, file = file.path(output_dir, "success_files.csv"), sep=",", row.names = FALSE)
```

```python
import pandas as pd
import numpy as np
import os
success_files = pd.read_csv("E:\experiment\DTR_AI_growth\data\ITRDB\success_files.csv")
len(success_files)

csv_file_list = []
csv_folder_path = r'E:\experiment\DTR_AI_growth\data\ITRDB\FZS_STD_crn'
for root, dirs, files in os.walk(csv_folder_path):
    for file_name in files:
        if file_name.endswith(".rwlcrn.csv"):
            site = file_name.split('.rwlcrn.csv')[0]
            csv_file_list.append(site)

base_folder_path = r'E:\experiment\DTR_AI_growth\data\rwI'

subfolders = ["africa", "asia", "atlantic", "australia", "canada", "europe", "mexico", "southamerica", "usa"]

keywords_dict = {
    "Collection_Name:": "Collection_Name",
    "Northernmost_Latitude:": "Northernmost_Latitude",
    "Easternmost_Longitude:": "Easternmost_Longitude",
    "Tree_Species_Code:": "Tree_Species_Code",
    "Location:": "Location",
    "Elevation_m:": "Elevation",
    "First_Year:": "First_Year",
    "Last_Year:": "Last_Year"
}

all_extracted_data = []

for subfolder in subfolders:
    folder_path = os.path.join(base_folder_path, subfolder)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith("-rwl-noaa.txt"): 
                file_path = os.path.join(root, file_name)

                site = file_name.split('-')[0]


                if site not in csv_file_list:
                    continue

                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                data_dict = {}

                for line in lines:
                    for keyword, column_name in keywords_dict.items():
                        if keyword in line:
                            data = line.replace(keyword, "").strip()
                            data_dict[column_name] = data.strip('#').strip()
                            break
                    else:

                        if "Elevation:" in line:
                            data_dict["Elevation"] = line.replace("Elevation:", "").strip().strip('#').strip()
                        elif "Earliest_Year:" in line:
                            data_dict["First_Year"] = line.replace("Earliest_Year:", "").strip().strip('#').strip()
                        elif "Most_Recent_Year:" in line:
                            data_dict["Last_Year"] = line.replace("Most_Recent_Year:", "").strip().strip('#').strip()

                data_dict["Subfolder"] = subfolder
                data_dict["site"] = site 

                all_extracted_data.append(data_dict)
df = pd.DataFrame(all_extracted_data)
def extract_end_year(csv_file_path):
    try:
        df_csv = pd.read_csv(csv_file_path)
        last_index = df_csv.index[-1]
        return last_index
    except Exception as e:
        print(f"Error extracting end year from {csv_file_path}: {str(e)}")
        return None
df['end_year'] = df.apply(lambda row: extract_end_year(os.path.join(csv_folder_path, f"{row['site']}.rwlcrn.csv")), axis=1)
def extract_strat_year(csv_file_path):
    try:
        df_csv = pd.read_csv(csv_file_path)
        last_index = df_csv.index[0]
        return last_index
    except Exception as e:
        print(f"Error extracting first1 year from {csv_file_path}: {str(e)}")
        return None
df['strat_year'] = df.apply(lambda row: extract_strat_year(os.path.join(csv_folder_path, f"{row['site']}.rwlcrn.csv")), axis=1)
from tqdm import tqdm
folder_path = 'E:\\experiment\\DTR_AI_growth\\data\\ITRDB\\FZS_STD_EPS'
listeps = []

for file_name in tqdm(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        dfeps = pd.read_csv(file_path)
        dfeps.insert(0, 'site', os.path.splitext(file_name)[0].split('.')[0])
        listeps.append(dfeps)
merged_df = pd.concat(listeps, ignore_index=True)
df_ITRDB = pd.merge(df,merged_df,on='site')
df_cleaned = df_ITRDB.drop_duplicates(subset=['site'], keep='first')
df_cleaned[df_cleaned.duplicated(subset=['site'], keep=False)]
df_cleaned.to_csv('ITRDB_Descrip.csv')
```

```python
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

gdf = gpd.read_file("ITRDB_site_FZS.shp")
Itrdb_site =pd.read_csv("itrdb_dem_landconr_values.csv")
Climate_PRE= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.pre.dat.nc")
list_pre=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_PRE.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_pre.append(pd.DataFrame(df1.loc['pre',:]).T)
df_pre = pd.concat(list_pre)
df_pre1=df_pre.reset_index().drop('index',axis=1)
df_pre1.columns=pd.to_datetime(df_pre1.columns)
years = df_pre1.columns.year
grouped_by_year = df_pre1.groupby(years, axis=1)
df_pre_mean_summer = grouped_by_year.sum().reset_index().drop('index',axis=1)
df_sit_summer_pre=pd.concat([Itrdb_site,df_pre_mean_summer],axis=1)
df_sit_summer_pre1 = df_sit_summer_pre.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_pre1.to_csv('annual_pre.csv',index=False)
Climate_temp= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.tmp.dat.nc")
list_temp=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_temp.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_temp.append(pd.DataFrame(df1.loc['tmp',:]).T)
df_temp = pd.concat(list_temp)
df_temp1=df_temp.reset_index().drop('index',axis=1)
df_temp1.columns=pd.to_datetime(df_temp1.columns)
years = df_temp1.columns.year
grouped_by_year = df_temp1.groupby(years, axis=1)
df_temp_mean_summer = grouped_by_year.mean().reset_index().drop('index',axis=1)
df_sit_summer_temp=pd.concat([Itrdb_site,df_temp_mean_summer],axis=1)
df_sit_summer_temp1 = df_sit_summer_temp.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_temp1.to_csv('annual_temp.csv',index=False)
df_sit_summer_temp1
Climate_DTR= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.dtr.dat.nc")
list_DTR=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_DTR.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_DTR.append(pd.DataFrame(df1.loc['dtr',:]).T)
df_DTR = pd.concat(list_DTR)
df_DTR1=df_DTR.reset_index().drop('index',axis=1)
df_DTR1.columns=pd.to_datetime(df_DTR1.columns)
years = df_DTR1.columns.year
grouped_by_year = df_DTR1.groupby(years, axis=1)
df_DTR_mean_summer = grouped_by_year.mean().reset_index().drop('index',axis=1)
df_sit_summer_DTR=pd.concat([Itrdb_site,df_DTR_mean_summer],axis=1)
df_sit_summer_DTR1 = df_sit_summer_DTR.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_DTR1.to_csv('annual_dtr.csv',index=False)
df_sit_summer_DTR1
Climate_PDSI= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/scPDSI.cru_ts4.07early1.1901.2022.cal_1901_22.bams.2023.GLOBAL.IGBP.WHC.1901.2022.nc")
list_scPDSI=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_PDSI.sel(longitude=gdf["Longitude"].tolist()[i], latitude=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_scPDSI.append(pd.DataFrame(df1.loc['scpdsi',:]).T)
df_scPDSI = pd.concat(list_scPDSI)
df_scPDSI1=df_scPDSI.reset_index().drop('index',axis=1)
df_scPDSI1.columns=pd.to_datetime(df_scPDSI1.columns)
years = df_scPDSI1.columns.year
grouped_by_year = df_scPDSI1.groupby(years, axis=1)
df_scPDSI_mean_summer = grouped_by_year.mean().reset_index().drop('index',axis=1)
df_sit_summer_scPDSI=pd.concat([Itrdb_site,df_scPDSI_mean_summer],axis=1)
df_sit_summer_scPDSI1 = df_sit_summer_scPDSI.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_scPDSI1.to_csv('annual_scPDSI.csv',index=False)
df_sit_summer_scPDSI1
Climate_Tmax= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.tmx.dat.nc")
list_Tmax=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_Tmax.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_Tmax.append(pd.DataFrame(df1.loc['tmx',:]).T)
df_Tmax = pd.concat(list_Tmax)
df_Tmax1=df_Tmax.reset_index().drop('index',axis=1)
df_Tmax1.columns=pd.to_datetime(df_Tmax1.columns)
years = df_Tmax1.columns.year
grouped_by_year = df_Tmax1.groupby(years, axis=1)
df_Tmax_mean_summer = grouped_by_year.mean().reset_index().drop('index',axis=1)
df_sit_summer_Tmax=pd.concat([Itrdb_site,df_Tmax_mean_summer],axis=1)
df_sit_summer_Tmax1 = df_sit_summer_Tmax.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_Tmax1.to_csv('annual_Tmax.csv',index=False)
df_sit_summer_Tmax1
Climate_Tmin= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.tmn.dat.nc")
list_Tmin=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_Tmin.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_Tmin.append(pd.DataFrame(df1.loc['tmn',:]).T)
df_Tmin= pd.concat(list_Tmin)
df_Tmin1=df_Tmin.reset_index().drop('index',axis=1)
df_Tmin1.columns=pd.to_datetime(df_Tmin1.columns)
years = df_Tmin1.columns.year
grouped_by_year = df_Tmin1.groupby(years, axis=1)
df_Tmin_mean_summer = grouped_by_year.mean().reset_index().drop('index',axis=1)
df_sit_summer_Tmin=pd.concat([Itrdb_site,df_Tmin_mean_summer],axis=1)
df_sit_summer_Tmin1 = df_sit_summer_Tmin.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_Tmin1.to_csv('annual_Tmin.csv',index=False)
Climate_Tmin= xr.open_dataset("E:\experiment\DTR_AI_growth\data\Climate_metric/cru_ts4.07.1901.2022.pet.dat.nc")
list_Tmin=[]
for i in tqdm(range(len(gdf["Longitude"].tolist())), desc="Processing Points"):
    df = Climate_Tmin.sel(lon=gdf["Longitude"].tolist()[i], lat=gdf["Latitude"].tolist()[i], method="nearest")
    df1=df.to_dataframe().T
    list_Tmin.append(pd.DataFrame(df1.loc['pet',:]).T)
df_Tmin= pd.concat(list_Tmin)
df_Tmin1=df_Tmin.reset_index().drop('index',axis=1)
df_Tmin1.columns=pd.to_datetime(df_Tmin1.columns)
years = df_Tmin1.columns.year
grouped_by_year = df_Tmin1.groupby(years, axis=1)
df_Tmin_mean_summer = grouped_by_year.sum().reset_index().drop('index',axis=1)
df_sit_summer_Tmin=pd.concat([Itrdb_site,df_Tmin_mean_summer],axis=1)
df_sit_summer_Tmin1 = df_sit_summer_Tmin.loc[:,'site':].drop(['snr','strat_year'],axis=1)
df_sit_summer_Tmin1.to_csv('annual_Pet.csv',index = False)
```





```R
library(lme4)
library(car)
library(Metrics)
library(MuMIn) 
df <- read.csv("E:/experiment/DTR_AI_growth/code5_droghtloss/AridGL.csv")
df$Tree.age <- scale(df$Tree.age)
lm_model <- lmer(Growth_loss ~ Pre + scPDSI + DTR + VPD + Pre_CV + Tmax_CV + Tmin_CV + DTR_CV
                 + Pet_CV + VPD_CV + Temp_CV +(1 | Tree_Speci_encoded) + (1 | Tree.age) , data = df)
fixed_effects <- summary(lm_model)$coefficients
write.csv(fixed_effects, "E:/experiment/DTR_AI_growth/code5_droghtloss/AridGLfixed_effects.csv")
summary(lm_model)
aic_value <- AIC(lm_model)
bic_value <- BIC(lm_model)
r_squared_marginal <- r.squaredGLMM(lm_model)[1]
r_squared_conditional <- r.squaredGLMM(lm_model)[2]

print(paste("Marginal R-squared:", r_squared_marginal))
print(paste("Conditional R-squared:", r_squared_conditional))
print(paste("AIC:", aic_value))
print(paste("BIC:", bic_value))
df <- read.csv("E:/experiment/DTR_AI_growth/code5_droghtloss/HumidGL.csv")
df$Tree.age <- scale(df$Tree.age)
lm_model <- lmer(Growth_loss ~ Pre + scPDSI + DTR + VPD + Pre_CV + Tmax_CV + Tmin_CV + DTR_CV
                 + Pet_CV + VPD_CV + Temp_CV +(1 | Tree_Speci_encoded) + (1 | Tree.age) , data = df)
fixed_effects <- summary(lm_model)$coefficients
write.csv(fixed_effects, "E:/experiment/DTR_AI_growth/code5_droghtloss/HumidGLfixed_effects.csv")
summary(lm_model)
aic_value <- AIC(lm_model)
bic_value <- BIC(lm_model)

r_squared_marginal <- r.squaredGLMM(lm_model)[1]
r_squared_conditional <- r.squaredGLMM(lm_model)[2]

print(paste("Marginal R-squared:", r_squared_marginal))
print(paste("Conditional R-squared:", r_squared_conditional))
print(paste("AIC:", aic_value))
print(paste("BIC:", bic_value))
```

