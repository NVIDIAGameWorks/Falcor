import urllib.parse
import urllib.request
import getpass
import argparse
import xml.etree.ElementTree as ET

server_url = 'http://teamcity.nvidia.com:80/'
project_url =  server_url + 'app/rest/builds/?locator=project:Falcor,count:1000'

def connect(username, password = ''):
    if not username:
        raise ConnectError('No username provided for teamcity connection')
    
    if not password:
        getPassPrompt = 'Enter teamcity password for user ' + username + ':'
        password = getpass.getpass(prompt=getPassPrompt )
    
    pword_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pword_mgr.add_password(None, server_url, username, password) # None param is a realm btw
    
    handler = urllib.request.HTTPBasicAuthHandler(pword_mgr)
    opener = urllib.request.build_opener(handler)
    opener.open(server_url)
    urllib.request.install_opener(opener)

