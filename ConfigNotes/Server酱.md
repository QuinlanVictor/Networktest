SCKEY

SCU131802Tef40bc6617c6e29c898cfdc99dbcbcc55fc655fa91537

    import requests
    sckey = 'your sckey'#在发送消息页面可以找到
    url = 'https://sc.ftqq.com/%s.send?text=程序完成了'%sckey
    #text为推送的title,desp为推送的描述
    url = 'https://sc.ftqq.com/%s.send?text=程序完成了&desp=好玩吧'%sckey
    requests.get(url)
