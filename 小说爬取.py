import requests
from lxml import etree
import os
import pandas
import json
def makedir(path):
    if os.path.exists(path):
        print(path + '文件夹已有')
    else:
        os.mkdir(path)
        print(path + '文件夹创建成功')
if __name__ == "__main__":
    headers = {  # todo 注意是User-Agent 不是 s
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.26'
    }
    #  番茄小说爬虫网址
    url = 'http://www.ybsf.org/ybsf/13234/'
    response = requests.get(url=url, headers=headers)
    response.encoding = 'utf-8'
    page_text = response.text

    tree = etree.HTML(page_text)
    title = tree.xpath('//*[@id="info"]/h1/text()')[0]

    detail_names = tree.xpath('//*[@id="list"]/dl/dd/a/text()')
    detail_urls = tree.xpath('//*[@id="list"]/dl/dd/a/@href')
    dir_name = 'E:\python_files\west2online_last_test\data' + '\\' + title
    makedir(dir_name)
    print(title)
    print(detail_names)
    print(detail_urls)
    for i in range(len(detail_names)):
        chapter_name = '%03d' % (i+1) + detail_names[i]
        chapter_name = chapter_name.replace('？','').replace('?','')
        chapter_url = 'http:'+detail_urls[i]
        response = requests.get(url=chapter_url, headers=headers)
        response.encoding = 'utf-8'
        page_text = response.text
        tree = etree.HTML(page_text)
        content = tree.xpath('//*[@id="content"]//text()')
        with open(dir_name + '\\' + chapter_name + '.txt', "a", encoding="utf-8") as f:
            for j in range(len(content)):
                f.write(content[j].replace(r'\xa0\xa0\xa0\xa0','\n').replace('\n',''))
            print(title + '\t\t' + chapter_name + '\t\t已经爬取完成\t\t')

        #print(content)
    #print(page_text)
    # E:\python_files\west2online_last_test\data

