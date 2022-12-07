1. 爬取https://sofifa.com/ 中运动员的数据

crawler 中，是爬取数据的脚步，*.py 脚本是尝试用的，sofifa_term.ipynb 是爬取一个页面的尝试脚本，尝试成功后在sofifa.ipynb 中爬取50个pages。共3000名球员
反思：在论文《融合XGBoost与SHA...动员身价预测及特征分析方法_廖彬》中，他们的样本量为16000个，是我现在的5倍多，可能需要多爬取一些数据。同时他还爬取在各个位置（如中场...,守门等位置）的评分，和身高，年龄，知名度，最佳位置等数据，可能需要为脚本加一些代码，多爬取一些特征。

data_50 中为sofifa.ipynb 爬取的50 个pages 的数据，每爬取一个页面就保存到一个.csv 文件中。

raw_data 是对data_50 中50个csv 的初步处理：
    1. 将50 个csv 合并为一个文件 raw_data.csv 
    2. 进行数据清洗，将含有inf 的数据清除
    3. 按是否为守门员进行分类，part1.csv 为非守门员的数据，part2 为守门员的数据
raw_data 的处理靠 dealdata 中的脚本完成.可能有些简单的代码被后面写的代码覆盖了

trypic 中的图像为我对数据集粗略的观看
    1. distribution.pdf 是看各因素的分布，part1_distributions.pdf 是看非守门员各因素的分布
    2. try_pictures.pdf 是看各因素与vlue, wage 的关系， part1_try_pictures.pdf是看非守门员各因素与vlue, wage 的关系
靠dealdata/plot_try.ipynb 完成的