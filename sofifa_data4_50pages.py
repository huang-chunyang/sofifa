# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from tqdm import tqdm

# %%
data_4 = []
#for train_name in tqdm(train_names, desc='读取文件及处理'):
for i in tqdm(range(50), desc="爬取页面"):
    offset = i * 60
    standing_url = "https://sofifa.com/?offset=" + str(offset)
    links_data = requests.get(standing_url)
    links_soup = BeautifulSoup(links_data.text)
    select_tbody = links_soup.select("body > div.center > div > div.col.col-12 > div > table > tbody")[0]
    athlete_links_a_unselected = select_tbody.find_all("a")
    # 利用if 选择运动员link，获得60个运动员的数据链接
    athlete_links = ["https://sofifa.com" + link_a.get("href") for link_a in athlete_links_a_unselected if link_a.get("data-tip-pos") == "top"]
    for athlete_link in athlete_links:
        time.sleep(1)
        # 爬取运动员页面的姓名和前四项'overall_rating', 'potential', 'value', 'wage'
        # 数据准备
        athlete_data = requests.get(athlete_link)
        athlete_soup = BeautifulSoup(athlete_data.text)
        name_list = ["name", "overall_rating", "potential", "value", "wage"]
        data_list = list(range(len(name_list)))
        i = 0

        # 姓名
        athlete_name = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > div > h1")
        athlete_name = str(athlete_name[0])
        name_re = re.match(".+>(.+)<", athlete_name)
        name = name_re.group(1)
        data_list[i] = name
        i += 1

        # overall rating 
        overall_rating = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(1) > div > span")
        overall_rating_data = overall_rating[0].string
        data_list[i] = int(overall_rating_data)
        i += 1

        # potential
        potential = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(2) > div > span")
        potential_data = potential[0].string
        potential_data
        data_list[i] = int(potential_data)
        i += 1

        #Value
        value = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(3) > div")
        value_data = str(value[0])
        value_re = re.findall("(\d+\.?\d*)([MK])", value_data)
        if value_re:
            value_num = value_re[0][0]
            value_unit = value_re[0][1]
            if value_unit == "M":
                data_list[i] = (float(value_num)*10**6)
            elif value_unit == "K":
                data_list[i] = (float(value_num*10**3))
            else:
                print("clawer value failed.")
        else:
            # print('regular match value failed:', value_data)
            data_list[i] = 0
        i += 1

        # Wage
        wage = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(4)")
        wage_data = str(wage[0])
        wage_re = re.findall("(\d+\.?\d*)([MK])", wage_data)
        if wage_re:
            wage_num = wage_re[0][0]
            wage_unit = wage_re[0][1]
            if wage_unit == "M":
                data_list[i] = (float(wage_num)*10**6)
            elif wage_unit == "K":
                data_list[i] = (float(wage_num)*10**3)
            else:
                print("clawer value failed")
        else:
            # print('regular match wage failed:', wage_data)
            data_list[i] = 0
        
        data_4.append(data_list)
        time.sleep(3)


# %% [markdown]
# 开始爬取运动员页面，例如，https://sofifa.com/player/268421/mathys-tel/230005/

# %%
len(data_4) # 330 * 60 

# %%
frame_data_4 = pd.DataFrame(data_4, columns=name_list)
frame_data_4.to_csv("./data_4/50pages.csv")

# %%


# %%



# %%



