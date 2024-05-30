import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
}

i = 1
for page in range(1,4):
    url = f'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=8611828923164059627&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E4%BC%AF%E6%9B%BC%E7%8C%AB&queryWord=%E4%BC%AF%E6%9B%BC%E7%8C%AB&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&expermode=&nojc=&isAsync=&pn=90&rn=30&gsm=5a&1695050021259='
    response = requests.get(url,headers = headers)
    json_data = response.json()
    # print(json_data)
    data_list = json_data['data']
    for data in data_list[:-1]:
        pic_url = data['hoverURL']
        image_data = requests.get(pic_url).content
        with open(f'img/伯曼猫/{i}.jpg','wb') as f:
            f.write(image_data)
        i+=1


# i = 1
# for url in pic_urls:
#    time.sleep(1)
#    try:
#        image_data = requests.get(url).content
#        with open(f'cat/{i}.jpg','wb') as f:
#            f.write(image_data)
#        i+=1
#    except:
#        continue