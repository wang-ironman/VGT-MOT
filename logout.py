import requests

url = 'http://ipgw.neu.edu.cn/srun_portal_pc.php?ac_id=1&'

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Length': '98',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Host': 'ipgw.neu.edu.cn',
    'Origin': 'http://ipgw.neu.edu.cn',
    'Referer': 'http://ipgw.neu.edu.cn/srun_portal_pc.php?ac_id=1&',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}

data = {
    'action': 'logout',
    'ac_id': '1',
    'username': '1971627',  # username
    'password': 'liweixi123',  # password
    'save_me': '0'
}

r = requests.post(url, data=data, headers=headers)
print(r.text)