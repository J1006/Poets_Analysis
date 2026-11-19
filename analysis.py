# -*- coding: utf-8 -*-
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# 解决 Mac 中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

li_bai_texts = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "众鸟高飞尽，孤云独去闲。相看两不厌，只有敬亭山。",
    "危楼高百尺，手可摘星辰。不敢高声语，恐惊天上人。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "谁家玉笛暗飞声，散入春风满洛城。此夜曲中闻折柳，何人不起故园情。",
    "兰陵美酒郁金香，玉碗盛来琥珀光。但使主人能醉客，不知何处是他乡。"
]

du_fu_texts = [
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "戍鼓断人行，边秋一雁声。露从今夜白，月是故乡明。有弟皆分散，无家问死生。寄书长不达，况乃未休兵。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。野径云俱黑，江船火独明。晓看红湿处，花重锦官城。",
    "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。",
    "丞相祠堂何处寻，锦官城外柏森森。映阶碧草自春色，隔叶黄鹂空好音。三顾频烦天下计，两朝开济老臣心。出师未捷身先死，长使英雄泪满襟。",
    "剑外忽传收蓟北，初闻涕泪满衣裳。却看妻子愁何在，漫卷诗书喜欲狂。白日放歌须纵酒，青春作伴好还乡。即从巴峡穿巫峡，便下襄阳向洛阳。",
    "舍南舍北皆春水，但见群鸥日日来。花径不曾缘客扫，蓬门今始为君开。盘飧市远无兼味，樽酒家贫只旧醅。肯与邻翁相对饮，隔篱呼取尽余杯。",
    "黄四娘家花满蹊，千朵万朵压枝低。留连戏蝶时时舞，自在娇莺恰恰啼。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "岱宗夫如何？齐鲁青未了。造化钟神秀，阴阳割昏晓。荡胸生曾云，决眦入归鸟。会当凌绝顶，一览众山小。"
]

custom_words = ["明月", "故乡", "敬亭山", "庐山", "瀑布", "银河", "白帝", "江陵", "猿声", "轻舟", "万重山", "汪伦", "黄鹤楼", "扬州", "孤帆", "碧空", "天门山", "楚江", "洛城", "玉笛", "故园", "国破", "草木", "烽火", "家书", "白头", "戍鼓", "雁声", "死生", "锦官城", "登台", "霜鬓", "浊酒", "祠堂", "蜀相", "英雄", "蓟北", "衣裳", "青春", "巴峡", "巫峡", "襄阳", "洛阳", "群鸥", "花径", "蓬门", "邻翁", "黄四娘", "戏蝶", "娇莺", "黄鹂", "白鹭", "西岭", "东吴", "岱宗", "齐鲁", "昏晓", "绝顶"]
for word in custom_words: jieba.add_word(word)

stop_words = ["，", "。", "？", "！", "、", "之", "其", "而", "以", "于", "所", "不", "无", "何", "谁", "来", "去", "是", "有", "在"]

def preprocess(text):
    words = jieba.lcut(text)
    return " ".join([w for w in words if w not in stop_words and w.strip() != ""])

def get_top_tfidf(texts, top_n=15):
    processed_texts = [preprocess(t) for t in texts]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0) 
    word_weights = list(zip(feature_names, mean_tfidf))
    word_weights.sort(key=lambda x: x[1], reverse=True)
    return word_weights[:top_n]

top_li_bai = get_top_tfidf(li_bai_texts)
top_du_fu = get_top_tfidf(du_fu_texts)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle('李白与杜甫高权重词 (TF-IDF) 对照分析', fontsize=16)
words_lb, scores_lb = zip(*top_li_bai[::-1])
words_df, scores_df = zip(*top_du_fu[::-1])

axes[0].barh(words_lb, scores_lb, color='#C0392B', alpha=0.8)
axes[0].set_title('李白（飘逸/天象山水）', fontsize=14)
axes[0].set_xlabel('TF-IDF 权重')

axes[1].barh(words_df, scores_df, color='#2980B9', alpha=0.8)
axes[1].set_title('杜甫（沉郁/身世家国）', fontsize=14)
axes[1].set_xlabel('TF-IDF 权重')

plt.tight_layout()
plt.savefig('李杜对比图.png', dpi=300, bbox_inches='tight') 
plt.show()

print("="*40)
print(f"{'李白 Top 15':<20} | {'杜甫 Top 15':<20}")
print("-" * 40)
for i in range(15):
    lb_str = f"{top_li_bai[i][0]} ({top_li_bai[i][1]:.4f})"
    df_str = f"{top_du_fu[i][0]} ({top_du_fu[i][1]:.4f})"
    print(f"{lb_str:<20} | {df_str:<20}")
print("="*40)