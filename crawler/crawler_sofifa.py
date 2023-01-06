import time
from tqdm import tqdm
import random
import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

print("处理命令行...")
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num", dest="number", type=int, help="number of csv")
args = parser.parse_args()
num = args.number

# for num in tqdm(range(100), desc="爬取页面"):
data = []
time.sleep(3 + random.random())
offset = num * 60
standing_url = "https://sofifa.com/?offset=" + str(offset)

links_data = requests.get(standing_url)
links_soup = BeautifulSoup(links_data.text)
select_tbody = links_soup.select("body > div.center > div > div.col.col-12 > div > table > tbody")[0]
athlete_links_a_unselected = select_tbody.find_all("a")
# 利用if 选择运动员link，获得60个运动员的数据链接
athlete_links = ["https://sofifa.com" + link_a.get("href") for link_a in athlete_links_a_unselected if link_a.get("data-tip-pos") == "top"]
for link_num, athlete_link in tqdm(enumerate(athlete_links), desc="爬取运动员数据"):
    try:
        time.sleep(1 + random.random())
        # 爬取运动员页面的姓名和前四项'overall_rating', 'potential', 'value', 'wage'
        # 数据准备
        athlete_data = requests.get(athlete_link)
        athlete_soup = BeautifulSoup(athlete_data.text)
        name_list = ["name", "overall_rating", "potential", "value", "wage", "Crossing", "Finishing", "Heading_Accuracy", "Short_Passing", "Volleys", "Dribbling", "Curve", "FK_Accuracy", "Long_Passing", "Ball_Control", "Acceleration", "Sprint_Speed", "Agility", "Reactions", "Balance", "Shot_Power", "Jumping", "Stamina", "Strength", "Long_Shots", "Aggression", "Interceptions", "Positioning", "Vision", "Penalties", "Composure", "Defensive_Awareness", "Standing_Tackle", "Sliding_Tackle", "GK_Diving", "GK_Handling", "GK_Kicking", "GK_Positioning", "GK_Reflexes"]
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
        data_list[i] = int(potential_data)
        i += 1

        #Value
        value = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(3) > div")
        value_data = str(value[0])
        value_re = re.findall("(\d+\.?\d*)([MK]?)", value_data)
        if value_re:
            value_num = value_re[0][0]
            value_unit = value_re[0][1]
            if value_unit == "M":
                data_list[i] = float(value_num)*10**6
            elif value_unit == "K":
                data_list[i] = float(value_num)*10**3
            elif value_unit =="":
                data_list[i] = float(value_num)
            else:
                print("clawer value failed.")
        else:
            data_list[i] = 0
        i += 1

        # Wage
        wage = athlete_soup.select("#body > div:nth-child(5) > div > div.col.col-12 > div.bp3-card.player > section > div:nth-child(4)")
        wage_data = str(wage[0])
        wage_re = re.findall("(\d+\.?\d*)([MK]?)", wage_data)
        if wage_re:
            wage_num = wage_re[0][0]
            wage_unit = wage_re[0][1]
            if wage_unit == "M":
                data_list[i] = float(wage_num)*10**6
            elif wage_unit == "K":
                data_list[i] = float(wage_num)*10**3
            elif wage_unit == "":
                data_list[i] = float(wage_num)
            else:
                print("clawer value failed")
        else:
            data_list[i] = 0
        i += 1

        # ATTACKING
        for j in range(5):
            ATTACKING = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(3) > div > ul > li:nth-child({line})".format(line=j + 1))
            ATTACKING_data = re.match(".+>(\d+)<", str(ATTACKING[0])).group(1)
            data_list[i] = int(ATTACKING_data)
            i += 1

        ## SKILL
        for j in range(5):
            # 1 - Dribbling
            SKILL = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(4) > div > ul > li:nth-child({line})".format(line=j + 1))
            SKILL_data = re.match(".+>(\d+)<", str(SKILL[0])).group(1)
            data_list[i] = int(SKILL_data)
            i += 1

        # MOVEMENT
        for j in range(5):
            MOVEMENT = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(5) > div > ul > li:nth-child({line})".format(line=j + 1))
            MOVEMENT_data = re.match(".+>(\d+)<", str(MOVEMENT[0])).group(1)
            data_list[i] = int(MOVEMENT_data)
            i += 1

        # POWER
        for j in range(5):
            POWER = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(6) > div > ul > li:nth-child({line})".format(line=j + 1))
            POWER_data = re.match(".+>(\d+)<", str(POWER[0])).group(1)
            data_list[i] = int(POWER_data)
            i += 1

        # MENTALITY
        for j in range(6):
            MENTALITY = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(7) > div > ul > li:nth-child({line})".format(line=j + 1))
            MENTALITY_data = re.match(".+>(\d+)<", str(MENTALITY[0])).group(1)
            data_list[i] = int(MENTALITY_data)
            i += 1

        # DEFENDING
        for j in range(3):
            DEFENDING = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(8) > div > ul > li:nth-child({line})".format(line=j + 1))
            DEFENDING_data = re.match(".+>(\d+)<", str(DEFENDING[0])).group(1)
            data_list[i] = int(DEFENDING_data)
            i += 1

        # GOALKEEPING
        for j in range(5):
            GOALKEEPING = athlete_soup.select("#body > div:nth-child(6) > div > div.col.col-12 > div:nth-child(9) > div > ul > li:nth-child({line})".format(line=j + 1))
            GOALKEEPING_data = re.match(".+>(\d+)<", str(GOALKEEPING[0])).group(1)
            data_list[i] = int(GOALKEEPING_data)
            i += 1
        data.append(data_list)
    except:
        print("the {n} page {link_n} is error".format(n=num, link_n=link_num))

frame_data = pd.DataFrame(data, columns=name_list)
frame_data.to_csv("../data_200/{index}page.csv".format(index=num))