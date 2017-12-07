# -*- coding:utf-8 -*-

from urllib import request
import re
from bs4 import BeautifulSoup

response = request.urlopen("http://baike.baidu.com/view/284853.htm")
html = response.read()#.decode(u'utf-8')

#html = urllib.request.urlopen('http://baike.baidu.com/view/284853.htm')
bs_obj = BeautifulSoup(html,"html.parser")

#findAll(tag, attributes, recursive, text, limit, keywords)
#find(tag, attributes, recursive, text, keywords)
#recursive=False表示只搜索直接儿子，否则搜索整个子树，默认为True。
#findAll(“a”）
#findAll(“a”, href=“”)
#findAll(“div”, class=“”)
#findAll(“button”, id=“”)

#a_list = bs_obj.findAll("a")
a_list = bs_obj.findAll("a",href=re.compile("\.baidu\.com\w?"))
print(a_list)

for aa in a_list:
    if not aa.find("img"):
        if aa.attrs.get('href'):
            print(aa.text, aa.attrs['href'])
        