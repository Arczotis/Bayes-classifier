import time
import requests
import urllib

page = input("请输入要爬取多少页：")
page = int(page) + 1  # 确保其至少是一页，因为 输入值可以是 0
header = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Mobile Safari/537.36 Edg/101.0.1210.32'
}
n = 1  # 图片的前缀 如 1.png
pn = 1  # pn是从第几张图片获取 百度图片下滑时默认一次性显示30张
for m in range(1, page):
    url = 'https://image.baidu.com/search/acjson?'
    param = {
        'tn': 'resultjson_com',
        'logid': '8648945621491868697',
        'ipn': 'rj',
        'ct': '201326592',
        'is': '',
        'fp': 'result',
        'fr': '',
        'word': '青岛天主教堂',
        'queryWord': '青岛天主教堂',
        'cl': '2',
        'lm': '-1',
        'ie': 'utf-8',
        'oe': 'utf-8',
        'adpicid': '',
        'st': '-1',
        'z': '',
        'ic': '',
        'hd': '',
        'latest': '',
        'copyright': '',
        's': '',
        'se': '',
        'tab': '',
        'width': '',
        'height': '',
        'face': '0',
        'istype': '2',
        'qc': '',
        'nc': '1',
        'expermode': '',
        'nojc': '',
        'isAsync': '',
        'pn': pn,
        'rn': '',
        'gsm': '3c',
    }
    page_info = requests.get(url=url, headers=header, params=param)
    page_info.encoding = 'utf-8'  # 确保解析的格式是utf-8的
    page_info = page_info.json()  # 转化为json格式在后面可以遍历字典获取其值
    info_list = page_info['data']  # 观察发现data中存在 需要用到的url地址
    del info_list[-1]  # 每一页的图片30张，下标是从 0 开始 29结束 ，那么请求的数据要删除第30个即 29为下标结束点
    img_path_list = []
    for i in info_list:
        img_path_list.append(i['thumbURL'])
    for index in range(len(img_path_list)):
        print(img_path_list[index])  # 所有的图片的访问地址
        urllib.request.urlretrieve(img_path_list[index], "C:/Users/10722/Desktop/青岛地表建筑/天主教堂/" + str(n) + '.jpg')
        n = n + 1
    pn += 30
